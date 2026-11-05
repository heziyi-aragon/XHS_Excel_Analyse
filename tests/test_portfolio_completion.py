"""连续报价回合、执行前规划和下游退回接口的离线功能验收。"""
from copy import deepcopy
import json
import sys
from uuid import uuid4

import pytest
from langgraph.types import Command

from agent.portfolio import build_portfolio_plan
from agent.nodes.portfolio import make_portfolio_handoff_node, make_portfolio_node
from agent.nodes.portfolio_review import build_repricing_graph
from tests.test_portfolio import _record, _state, _targets


def _working_state():
    state = _state([_record("a", price_image=100, price_video=200),
                    _record("b", price_image=80, price_video=160)], budget=10000)
    state["pricing_candidates"][0]["nickname"] = "小桃"
    state["pricing_candidates"][0]["talent"]["nickname"] = "小桃"
    return state


def test_a_second_round_quantity_change_keeps_the_first_round_discount_and_history():
    state = _working_state()
    original = deepcopy(state)
    first = build_portfolio_plan(state, _targets(exposure=10000),
                                 feedback={"feedback_id": "discount", "comments": "九折"})
    assert first["configuration"]["price_multiplier"] == 0.9
    state["portfolio_plan"] = first
    second = build_portfolio_plan(state, feedback={"feedback_id": "quantity", "comments": "每人2篇"})
    assert second["configuration"]["price_multiplier"] == 0.9
    assert second["post_caps"]["default"] == 2
    assert sum(line["quantity"] for line in second["lines"]) == 4
    assert second["revision"] == first["revision"] + 1
    assert second["revision_history"][-1]["revision"] == first["revision"]
    assert second["revision_history"][-1]["configuration"]["price_multiplier"] == 0.9
    for line in second["lines"]:
        assert line["unit_price_suggested"] == pytest.approx(line["platform_reference_price"] * 0.9)
    assert {key: value for key, value in state.items() if key != "portfolio_plan"} == original


def test_nested_patches_preserve_other_targets_and_other_creators_settings():
    state = _working_state()
    config = {**_targets(exposure=10000, interactions=1000, cpe=25),
              "per_creator_max_posts": {"a": 1, "b": 2},
              "per_creator_allowed_formats": {"a": ["image"], "b": ["image"]}}
    first = build_portfolio_plan(state, config)
    state["portfolio_plan"] = first
    second = build_portfolio_plan(state, {"targets": {"cpe": {"value": 20}},
                                          "per_creator_max_posts": {"a": 2}},
                                  feedback={"id": "formats", "config_overrides": {
                                      "per_creator_allowed_formats": {"a": ["video"]}}})
    cfg = second["configuration"]
    assert cfg["targets"]["exposure"] == first["configuration"]["targets"]["exposure"]
    assert cfg["targets"]["interactions"] == first["configuration"]["targets"]["interactions"]
    assert cfg["targets"]["cpe"]["value"] == 20
    assert cfg["targets"]["cpe"]["source"] == first["configuration"]["targets"]["cpe"]["source"]
    assert cfg["per_creator_max_posts"] == {"a": 2, "b": 2}
    assert cfg["per_creator_allowed_formats"] == {"a": ["video"], "b": ["image"]}


@pytest.mark.parametrize("id_key", ["feedback_id", "review_id", "id"])
def test_repeated_feedback_identity_does_not_repeat_revision_history_or_model_calls(id_key):
    class Model:
        calls = 0

        def complete_json(self, *_args, **_kwargs):
            self.calls += 1
            return {"proposals": []}

    client = Model()
    state = _working_state()
    feedback = {id_key: "same-message", "comments": "九折"}
    first = build_portfolio_plan(state, llm_client=client, feedback=feedback)
    state["portfolio_plan"] = deepcopy(first)
    calls = client.calls
    repeated = build_portfolio_plan(state, llm_client=client, feedback=feedback)
    assert repeated == first
    assert client.calls == calls
    repeated["configuration"]["price_multiplier"] = 0.5
    assert state["portfolio_plan"] == first


def test_a_changed_input_cannot_reuse_a_previous_plan_just_because_feedback_id_matches():
    state = _working_state()
    feedback = {"feedback_id": "same-id", "comments": "九折"}
    first = build_portfolio_plan(state, feedback=feedback)
    state["portfolio_plan"] = first
    state["pricing_candidates"][0]["talent"]["price_image"] = 110
    second = build_portfolio_plan(state, feedback=feedback)
    assert second["revision"] == first["revision"] + 1
    assert second["input_fingerprint"] != first["input_fingerprint"]
    line = next(line for line in second["lines"] if line["xhs_id"] == "a")
    assert line["platform_reference_price"] == 110
    assert line["unit_price_suggested"] == 99


def test_changed_audit_summary_cannot_reuse_a_plan_with_stale_review_evidence():
    state = _working_state()
    feedback = {"feedback_id": "same-id", "comments": "九折"}
    first = build_portfolio_plan(state, feedback=feedback)
    state["portfolio_plan"] = first
    state["audit_summary"]["policy_changed_after_approval"] = True
    second = build_portfolio_plan(state, feedback=feedback)
    assert second["revision"] == first["revision"] + 1
    assert second["input_fingerprint"] != first["input_fingerprint"]
    assert second["upstream_risks"]["audit_summary"]["policy_changed_after_approval"] is True


def test_node_old_config_snapshot_cannot_reset_changes_from_previous_feedback():
    state = _working_state()
    state["portfolio_config"] = {"max_posts_per_creator": 1, **_targets(exposure=10000)}
    state["portfolio_feedback"] = {"id": "first", "comments": "九折"}
    node = make_portfolio_node()
    state.update(node(state))
    first = deepcopy(state["portfolio_plan"])
    state["portfolio_feedback"] = {"id": "second", "comments": "每人2篇"}
    state.update(node(state))
    assert state["portfolio_config"]["max_posts_per_creator"] == 1
    assert state["portfolio_plan"]["configuration"]["max_posts_per_creator"] == 2
    assert state["portfolio_plan"]["configuration"]["price_multiplier"] == 0.9
    assert state["portfolio_plan"]["revision"] == first["revision"] + 1
    assert all(row["attempt"] == 2 for row in state["pricing_candidates"])


def test_an_explicit_node_patch_can_reset_a_prior_discount_without_losing_other_settings():
    state = _working_state()
    state["portfolio_config"] = {"max_posts_per_creator": 2}
    state["portfolio_feedback"] = {"id": "discount", "comments": "九折"}
    node = make_portfolio_node()
    state.update(node(state))
    state["portfolio_feedback"] = None
    state["portfolio_config_patch"] = {"price_multiplier": 1}
    state.update(node(state))
    assert state["portfolio_plan"]["configuration"]["price_multiplier"] == 1
    assert state["portfolio_plan"]["configuration"]["max_posts_per_creator"] == 2
    assert all(line["unit_price_suggested"] == line["platform_reference_price"]
               for line in state["portfolio_plan"]["lines"])


def test_correcting_an_unresolved_review_keeps_the_last_valid_revision_history():
    state = _working_state()
    state["portfolio_feedback"] = {"id": "first", "comments": "九折"}
    node = make_portfolio_handoff_node()
    state.update(node(state))
    first = deepcopy(state["portfolio_plan"])
    state["portfolio_feedback"] = {"id": "unresolved", "comments": "重新定价"}
    state.update(node(state))
    assert state["portfolio_plan"] is None
    assert state["portfolio_error"]["type"] == "needs_clarification"
    assert state["portfolio_previous_plan"] == first
    state["portfolio_feedback"] = {"id": "still-unresolved", "comments": "更有性价比"}
    state.update(node(state))
    assert state["portfolio_plan"] is None
    assert state["portfolio_previous_plan"] == first
    state["portfolio_feedback"] = {"id": "resolved", "comments": "每人2篇"}
    state.update(node(state))
    assert state["portfolio_error"] is None
    assert state["portfolio_plan"]["revision"] == first["revision"] + 1
    assert state["portfolio_plan"]["revision_history"][-1]["revision"] == first["revision"]
    assert state["portfolio_plan"]["pricing_policy"]["price_multiplier"] == 0.9
    assert state["portfolio_plan"]["post_caps"]["default"] == 2


def test_natural_language_format_is_executed_before_the_solver_uses_video_prices():
    state = _working_state()
    original = deepcopy(state)
    plan = build_portfolio_plan(state, feedback={"id": "video", "comments": "小桃改成视频"})
    line = next(line for line in plan["lines"] if line["xhs_id"] == "a")
    assert line["format"] == "video"
    assert line["platform_reference_price"] == line["unit_price_suggested"] == 200
    assert any(item["path"].startswith("per_creator_allowed_formats.") for item in plan["planning"]["accepted"])
    assert state == original


def test_model_constraint_is_verified_and_applied_before_solver_instead_of_only_annotating():
    class Model:
        calls = 0

        def complete_json(self, *_args, **_kwargs):
            self.calls += 1
            return {"proposals": [{"path": "max_posts_per_creator", "value": 2,
                                   "source_quote": "每人2篇", "source_kind": "feedback"}]}

    model = Model()
    plan = build_portfolio_plan(_working_state(), _targets(exposure=10000), llm_client=model,
                                feedback={"id": "quantity", "comments": "每人2篇"})
    assert model.calls == 1
    assert plan["configuration"]["max_posts_per_creator"] == 2
    assert sum(line["quantity"] for line in plan["lines"]) == 4
    assert plan["llm_annotations"]["execution_phase"] == "before_solver"
    assert plan["planning"]["model_used"] is True


def test_benchmark_valuation_reaches_the_solver_and_the_budget_uses_its_high_bound():
    state = _working_state()
    state["pricing_candidates"] = state["pricing_candidates"][:1]
    config = {"pricing_as_of": "2026-11-04", "industry_benchmarks": [
        {"id": identity, "creator_id": identity, "content_type": "Beauty", "tier": "T3",
         "format": "image", "price": price, "source": "构造的行业证据", "as_of": "2026-11-01"}
        for identity, price in (("peer1", 90), ("peer2", 100), ("peer3", 110))
    ]}
    plan = build_portfolio_plan(state, config)
    line = plan["lines"][0]
    assert line["valuation"]["benchmark"]["adequate_samples"] is True
    assert line["valuation"]["status"] == "industry_reference"
    assert line["unit_price_low"] > 80
    assert line["unit_price_high"] < 120
    assert plan["budget"]["total_upper"] == line["line_cost_high"]
    assert plan["industry_benchmark_status"] != "not_provided"


def test_downstream_rejection_loops_only_through_pricing_and_preserves_the_audit(monkeypatch):
    from agent.nodes.audit import AuditNodes

    state = _working_state()
    state["portfolio_config"] = _targets(exposure=10000)
    original = deepcopy(state)
    decisions = iter([
        {"action": "revise", "review_id": "discount", "feedback": "九折"},
        {"action": "revise", "review_id": "quantity", "feedback": "每人2篇"},
        {"action": "continue", "review_id": "continue"},
    ])
    seen_revisions = []

    def review(current):
        seen_revisions.append(current["portfolio_plan"]["revision"])
        return {"portfolio_review": next(decisions)}

    def unexpected_audit(*_args, **_kwargs):
        raise AssertionError("报价回退不能重新运行上游审核")

    monkeypatch.setattr(AuditNodes, "audit_reasons", unexpected_audit)
    result = build_repricing_graph(review).invoke(
        state, config={"configurable": {"thread_id": uuid4().hex}, "recursion_limit": 32})
    assert seen_revisions == [1, 2, 3]
    assert result["portfolio_plan"]["revision"] == 3
    assert result["portfolio_plan"]["configuration"]["price_multiplier"] == 0.9
    assert result["portfolio_plan"]["post_caps"]["default"] == 2
    assert result["portfolio_review_status"] == "accepted_for_handoff"
    assert result["portfolio_plan"]["approval_status"] == "pending_final_human_review"
    assert result["next_step"]["name"] == "write_suggestions"
    for field in ("pricing_candidates", "audited_talents", "audit_pending", "supplement_results",
                  "audit_unverified_checks", "audit_summary"):
        assert result[field] == original[field]
    assert state == original


def test_repricing_an_existing_plan_does_not_create_an_extra_revision_before_review():
    state = _working_state()
    state.update(make_portfolio_node()(state))
    seen = []

    def review(current):
        seen.append(current["portfolio_plan"]["revision"])
        if len(seen) == 1:
            return {"action": "revise", "feedback": "九折", "review_id": "discount"}
        return {"action": "continue"}

    result = build_repricing_graph(review).invoke(
        state, config={"configurable": {"thread_id": uuid4().hex}})
    assert seen == [1, 2]
    assert result["portfolio_plan"]["revision"] == 2
    assert result["portfolio_plan"]["pricing_policy"]["price_multiplier"] == 0.9


def test_duplicate_downstream_review_id_stops_without_repeating_the_same_revision():
    calls = []

    def review(current):
        calls.append(current["portfolio_plan"]["revision"])
        return {"action": "revise", "feedback": "九折", "review_id": "same-message"}

    result = build_repricing_graph(review).invoke(
        _working_state(), config={"configurable": {"thread_id": uuid4().hex}, "recursion_limit": 32})
    assert calls == [1, 2]
    assert result["portfolio_review_status"] == "duplicate_feedback_ignored"
    assert result["portfolio_plan"]["revision"] == 2


def test_revision_limit_keeps_the_latest_proposal_and_pending_human_status():
    def review(current):
        return {"action": "revise", "feedback": "九折", "review_id": str(current["portfolio_plan"]["revision"])}

    result = build_repricing_graph(review, max_revisions=2).invoke(
        _working_state(), config={"configurable": {"thread_id": uuid4().hex}, "recursion_limit": 32})
    assert result["portfolio_plan"]["revision"] == 2
    assert result["workflow_status"] == "portfolio_revision_limit_reached"
    assert result["portfolio_review_status"] == "revision_limit_reached"
    assert result["portfolio_plan"]["approval_status"] == "pending_final_human_review"
    assert result["audited_talents"] and result["pricing_candidates"]


def test_standalone_feedback_interrupt_resumes_to_reprice_then_handoff():
    graph = build_repricing_graph()
    config = {"configurable": {"thread_id": uuid4().hex}, "recursion_limit": 32}
    first = graph.invoke(_working_state(), config=config)
    assert first["__interrupt__"][0].value["type"] == "portfolio_feedback"
    assert first["portfolio_plan"]["revision"] == 1
    second = graph.invoke(Command(resume={"action": "revise", "review_id": "discount", "feedback": "九折"}),
                          config=config)
    assert second["__interrupt__"][0].value["type"] == "portfolio_feedback"
    assert second["portfolio_plan"]["revision"] == 2
    assert second["portfolio_plan"]["pricing_policy"]["price_multiplier"] == 0.9
    final = graph.invoke(Command(resume={"action": "continue"}), config=config)
    assert not final.get("__interrupt__")
    assert final["portfolio_review_status"] == "accepted_for_handoff"
    assert final["portfolio_plan"]["approval_status"] == "pending_final_human_review"


def test_review_callback_only_receives_a_copy_of_upstream_evidence():
    state = _working_state()
    original = deepcopy(state)

    def review(current):
        current["audited_talents"][0]["attempt"] = 999
        current["pricing_candidates"].clear()
        return {"action": "approve"}

    result = build_repricing_graph(review).invoke(
        state, config={"configurable": {"thread_id": uuid4().hex}})
    assert result["audited_talents"] == original["audited_talents"]
    assert result["pricing_candidates"] == original["pricing_candidates"]
    assert result["portfolio_review_status"] == "accepted_for_handoff"
    assert result["portfolio_plan"]["approval_status"] == "pending_final_human_review"
    assert state == original


@pytest.mark.parametrize("envelope", [False, True])
def test_repricing_cli_accepts_saved_states_and_preserves_multi_round_changes(tmp_path, monkeypatch, envelope):
    from scripts.reprice_portfolio import main

    state = _working_state()
    state.update(make_portfolio_handoff_node()(state))
    source = tmp_path / "original.json"
    payload = {"upstream_state": _working_state(), "portfolio_plan": state["portfolio_plan"]} if envelope else state
    source.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    original_bytes = source.read_bytes()
    output, report = tmp_path / "revised.json", tmp_path / "revised.md"
    answers = iter(["九折", "每人2篇", "继续"])
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    monkeypatch.setattr(sys, "argv", ["reprice_portfolio", str(source), "--output", str(output), "--report", str(report)])
    assert main() == 0
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["portfolio_plan"]["revision"] == 3
    assert saved["portfolio_plan"]["configuration"]["price_multiplier"] == 0.9
    assert saved["portfolio_plan"]["configuration"]["max_posts_per_creator"] == 2
    assert saved["portfolio_plan"]["approval_status"] == "pending_final_human_review"
    assert saved["audited_talents"] == state["audited_talents"]
    assert saved["pricing_candidates"] == state["pricing_candidates"]
    assert source.read_bytes() == original_bytes
    assert report.exists()


def test_repricing_cli_blank_reply_preserves_the_original_without_writing_a_result(tmp_path, monkeypatch):
    from scripts.reprice_portfolio import main

    state = _working_state()
    state.update(make_portfolio_handoff_node()(state))
    source, output = tmp_path / "original.json", tmp_path / "revised.json"
    source.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    before = source.read_bytes()
    monkeypatch.setattr("builtins.input", lambda *_: "")
    monkeypatch.setattr(sys, "argv", ["reprice_portfolio", str(source), "--output", str(output)])
    assert main() == 0
    assert source.read_bytes() == before
    assert not output.exists()


def test_repricing_cli_rejects_a_report_path_that_would_overwrite_its_new_json(tmp_path, monkeypatch):
    from scripts.reprice_portfolio import main

    source, output = tmp_path / "original.json", tmp_path / "revised.json"
    source.write_text(json.dumps(_working_state(), ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["reprice_portfolio", str(source), "--output", str(output), "--report", str(output)])
    with pytest.raises(SystemExit) as rejected:
        main()
    assert rejected.value.code == 2
    assert not output.exists()


def test_repricing_cli_can_clarify_a_saved_error_and_keep_the_last_valid_discount(tmp_path, monkeypatch):
    from scripts.reprice_portfolio import main

    state = _working_state()
    state["portfolio_feedback"] = {"id": "discount", "comments": "九折"}
    node = make_portfolio_handoff_node()
    state.update(node(state))
    first = deepcopy(state["portfolio_plan"])
    state["portfolio_feedback"] = {"id": "unknown", "comments": "更有性价比"}
    state.update(node(state))
    assert state["portfolio_plan"] is None
    source, output = tmp_path / "error.json", tmp_path / "clarified.json"
    source.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    before = source.read_bytes()
    answers = iter(["每人2篇", "继续"])
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    monkeypatch.setattr(sys, "argv", ["reprice_portfolio", str(source), "--output", str(output)])
    assert main() == 0
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["portfolio_error"] is None
    assert saved["portfolio_plan"]["revision"] == first["revision"] + 1
    assert saved["portfolio_plan"]["pricing_policy"]["price_multiplier"] == 0.9
    assert saved["portfolio_plan"]["post_caps"]["default"] == 2
    assert saved["portfolio_review_status"] == "accepted_for_handoff"
    assert source.read_bytes() == before


def test_continue_cannot_turn_an_unresolved_quote_error_into_an_approved_handoff():
    state = _working_state()
    node = make_portfolio_handoff_node()
    state.update(node(state))
    state["portfolio_feedback"] = {"id": "unknown", "comments": "重新定价"}
    state.update(node(state))
    assert state["portfolio_plan"] is None and state["portfolio_previous_plan"]
    result = build_repricing_graph(lambda _: {"action": "continue"}).invoke(
        state, config={"configurable": {"thread_id": uuid4().hex}})
    assert result["portfolio_error"]
    assert result["portfolio_plan"] is None
    assert result["portfolio_previous_plan"] == state["portfolio_previous_plan"]
    assert result["portfolio_review_status"] != "accepted_for_handoff"
    assert result["workflow_status"] == "portfolio_needs_input"


def test_a_failed_model_request_is_reported_unavailable_while_offline_rules_still_execute():
    class UnavailableModel:
        def complete_json(self, *_args, **_kwargs):
            raise RuntimeError("offline service failure")

    plan = build_portfolio_plan(_working_state(), llm_client=UnavailableModel(),
                                feedback={"id": "discount", "comments": "九折"})
    assert plan["configuration"]["price_multiplier"] == 0.9
    assert plan["planning"]["status"] == "ready"
    assert plan["planning"]["model_requested"] is True
    assert plan["planning"]["model_used"] is False
    assert plan["llm_annotations"]["status"] == "unavailable"
    assert all(line["unit_price_suggested"] == pytest.approx(line["platform_reference_price"] * 0.9)
               for line in plan["lines"])
