"""共享主图第9步接线与CLI验收，保留原审核阶段单元测试的职责。"""
from copy import deepcopy
import json
from uuid import uuid4

import pytest
from langgraph.types import Command

from agent.graph import build_graph
from agent.nodes.portfolio import make_portfolio_handoff_node
from agent.requirements import LLMClient
import main as main_module
from tests.test_requirements import DATA, ResponseSequence, complete_profile, reason_draft, valid_policy


_AUDIT_FIELDS = (
    "brief", "talents", "issues", "requirement_profile", "screening_policy", "approved_policy",
    "active_policy", "screen_relax_count", "screen_log", "excluded_talents", "deduped_talents",
    "evaluated_talents", "scored_talents", "reason_results", "audit_results", "audit_history",
    "audit_summary", "audit_unverified_checks", "audited_talents", "audit_pending",
    "supplement_results", "supplement_history", "manual_review_queue", "pricing_candidates",
)


def _run_graph(*, enabled=True, portfolio_config=None, feedback=None, annotation_client=None,
               client_answers=None, review_node=None):
    sequence = ResponseSequence([complete_profile(), valid_policy()], default=reason_draft())
    kwargs = {"audit_llm_client": LLMClient(transport=lambda *_: {"verdict": "pass", "issues": []})}
    if not enabled:
        kwargs["enable_portfolio"] = False
    if annotation_client is not None:
        kwargs["portfolio_llm_client"] = annotation_client
    if review_node is not None:
        kwargs["portfolio_review_node"] = review_node
    graph = build_graph(LLMClient(transport=sequence), **kwargs)
    invocation = {"configurable": {"thread_id": "portfolio-integration-" + uuid4().hex}, "recursion_limit": 64}
    initial_state = {
        "brief_path": str(DATA / "甲方投放需求单_屿光精华.xlsx"),
        "talent_path": str(DATA / "蒲公英达人数据_屿光精华.xlsx"),
    }
    if portfolio_config is not None:
        initial_state["portfolio_config"] = deepcopy(portfolio_config)
    if feedback is not None:
        initial_state["portfolio_feedback"] = deepcopy(feedback)
    if client_answers is not None:
        initial_state["client_answers"] = client_answers
    initial = graph.invoke(initial_state, config=invocation)
    assert len(initial["__interrupt__"]) == 1
    assert initial["__interrupt__"][0].value["type"] == "policy_review"
    assert initial.get("portfolio_plan") is None
    completed = graph.invoke(Command(resume={"action": "approve"}), config=invocation)
    assert not completed.get("__interrupt__")
    return completed, sequence


@pytest.fixture(scope="module")
def graph_modes():
    audit, audit_sequence = _run_graph(enabled=False)
    integrated, integrated_sequence = _run_graph()
    return audit, integrated, audit_sequence, integrated_sequence


def test_default_main_graph_completes_portfolio_and_advances_to_suggestions(graph_modes):
    audit, result, _, sequence = graph_modes
    plan = result["portfolio_plan"]
    assert plan["lines"]
    assert result["workflow_status"] == "portfolio_" + plan["status"]
    assert result["next_step"]["name"] == "write_suggestions"
    assert result["next_step"]["implemented"] is False
    assert result["next_step"]["status"] == "interface_ready"
    assert result["audit_workflow_status"] == audit["workflow_status"]
    assert result["audit_next_step"] == audit["next_step"]
    assert plan["approval_status"] == "pending_final_human_review"
    assert plan["budget"]["total_upper"] <= result["brief"]["total_budget"]
    assert plan["llm_annotations"]["status"] == "not_requested"
    assert not any("proposal" in payload for payload in sequence.payloads)


def test_disable_flag_retains_the_original_audit_terminal_contract(graph_modes):
    audit, _, _, _ = graph_modes
    assert audit.get("portfolio_plan") is None
    assert audit.get("portfolio_error") is None
    assert audit["workflow_status"] == "audit_complete_with_incomplete_data"
    assert audit["next_step"]["name"] == "build_portfolio"
    assert audit["next_step"]["implemented"] is False
    assert audit["pricing_candidates"]


def test_enabling_portfolio_preserves_every_upstream_assessment_and_audit_risk(graph_modes):
    audit, result, audit_sequence, integrated_sequence = graph_modes
    for field in _AUDIT_FIELDS:
        assert result.get(field) == audit.get(field), field
    assert integrated_sequence.calls == audit_sequence.calls
    risks = result["portfolio_plan"]["upstream_risks"]
    for field in ("audited_talents", "audit_pending", "supplement_results", "audit_unverified_checks", "audit_summary"):
        assert risks[field] == result[field]
    assert result["supplement_results"]
    assert result["audit_unverified_checks"]
    assert all(check["status"] == "unverified" for check in risks["audit_unverified_checks"])


def test_portfolio_config_and_feedback_survive_shared_graph_state_without_changing_attempts(graph_modes):
    audit, _, _, _ = graph_modes
    config = {"price_multiplier": 0.9, "max_posts_per_creator": 2}
    feedback = {"comments": "只将建议价调为85折", "config_overrides": {"price_multiplier": 0.85}}
    result, _ = _run_graph(portfolio_config=config, feedback=feedback)
    assert result["portfolio_config"] == config
    assert result["portfolio_feedback"] == feedback
    assert result["portfolio_plan"]["pricing_policy"]["price_multiplier"] == 0.85
    assert result["portfolio_plan"]["post_caps"]["default"] == 2
    assert result["portfolio_plan"]["feedback"] == feedback
    assert result["evaluated_talents"] == audit["evaluated_talents"]
    for line in result["portfolio_plan"]["lines"]:
        assert line["unit_price_suggested"] == pytest.approx(line["platform_reference_price"] * 0.85)


def test_optional_portfolio_client_is_explicit_and_cannot_change_the_numeric_plan(graph_modes):
    calls = []

    def annotate(_system, payload):
        calls.append(deepcopy(payload))
        return {"proposals": []}

    _, original, _, _ = graph_modes
    result, _ = _run_graph(annotation_client=LLMClient(transport=annotate))
    assert len(calls) == 1 and "sources" in calls[0] and "candidate_aliases" in calls[0]
    assert result["portfolio_plan"]["llm_annotations"]["status"] == "generated_for_review"
    assert result["portfolio_plan"]["llm_annotations"]["execution_phase"] == "before_solver"
    for field in ("lines", "budget", "kpi_checks", "upstream_risks"):
        assert result["portfolio_plan"][field] == original["portfolio_plan"][field]


@pytest.mark.parametrize("invalid_schema", [False, True])
def test_unusable_advisory_output_still_hands_a_valid_plan_to_suggestions(
        graph_modes, invalid_schema, tmp_path, monkeypatch, capsys):
    audit, baseline, _, _ = graph_modes

    def model(_system, payload):
        if invalid_schema:
            return {"unexpected": "unusable model response"}
        brief = payload["brief_fields"]
        return {"proposals": [
            {"path": "budget", "value": 350000,
             "source_quote": brief["raw"]["总预算"], "source_kind": "brief"},
            {"path": "price_multiplier", "value": 1.2,
             "source_quote": brief["notes"], "source_kind": "brief"},
        ]}

    result, _ = _run_graph(annotation_client=LLMClient(transport=model))
    assert result["portfolio_error"] is None
    assert result["next_step"]["name"] == "write_suggestions"
    assert result["next_step"]["status"] == "interface_ready"
    assert result["portfolio_plan"]["planning"]["rejected"]
    assert result["portfolio_plan"]["planning"]["pending"] == []
    for field in ("lines", "budget", "kpi_checks", "upstream_risks"):
        assert result["portfolio_plan"][field] == baseline["portfolio_plan"][field]
    for field in _AUDIT_FIELDS:
        assert result.get(field) == audit.get(field), field
    original = deepcopy(result)
    monkeypatch.setattr(main_module, "BASE", tmp_path)
    saved = json.loads(main_module.save_result(result).read_text(encoding="utf-8"))
    assert saved["portfolio_plan"]["planning"] == result["portfolio_plan"]["planning"]
    assert saved["portfolio_error"] is None
    main_module.render_shortlist(result)
    rendered = capsys.readouterr().out
    assert "部分模型建议未通过核验" in rendered
    assert "下一步：写建议" in rendered
    assert "unverified_model_proposals_require_review" not in rendered
    assert result == original


@pytest.mark.parametrize("with_previous", [False, True])
def test_real_clarification_has_readable_output_and_preserves_diagnostics(
        graph_modes, with_previous, capsys):
    audit, baseline, _, _ = graph_modes
    state = deepcopy(audit)
    if with_previous:
        state["portfolio_plan"] = deepcopy(baseline["portfolio_plan"])
    state["portfolio_feedback"] = "更有性价比"
    state.update(make_portfolio_handoff_node()(state))
    assert state["portfolio_plan"] is None
    assert state["workflow_status"] == "portfolio_needs_input"
    assert state["portfolio_error"]["planning"]["pending"]
    assert state["portfolio_previous_plan"] == (baseline["portfolio_plan"] if with_previous else None)
    original = deepcopy(state)
    main_module.render_shortlist(state)
    rendered = capsys.readouterr().out
    assert rendered.count("更有性价比") == 1
    assert "请明确具体折扣、篇数或内容形式" in rendered
    assert "source_kind" not in rendered
    assert "request_has_no_supported_executable_value" not in rendered
    assert ("上一次有效报价已单独保留" in rendered) == with_previous
    assert state == original


def test_save_step9_preserves_existing_step8_snapshot_and_all_audit_evidence(tmp_path, monkeypatch, graph_modes):
    audit, integrated, _, _ = graph_modes
    monkeypatch.setattr(main_module, "BASE", tmp_path)
    audit_path = main_module.save_result(deepcopy(audit))
    assert audit_path.name == "step8_audit.json"
    old_bytes = audit_path.read_bytes()
    state = deepcopy(integrated)
    state["portfolio_config"] = {"max_posts_per_creator": 1}
    state["portfolio_feedback"] = "集成验收备注"
    original = deepcopy(state)
    portfolio_path = main_module.save_result(state)
    assert portfolio_path.name == "step9_portfolio.json"
    assert audit_path.read_bytes() == old_bytes
    assert state == original
    saved = json.loads(portfolio_path.read_text(encoding="utf-8"))
    for field in (*_AUDIT_FIELDS, "portfolio_plan", "portfolio_config", "portfolio_feedback",
                  "audit_workflow_status", "audit_next_step", "workflow_status", "next_step"):
        assert saved.get(field) == state.get(field), field
    assert "portfolio_plan" not in json.loads(old_bytes)


def test_save_step9_does_not_create_or_overwrite_a_separate_audit_snapshot(tmp_path, monkeypatch, graph_modes):
    _, integrated, _, _ = graph_modes
    monkeypatch.setattr(main_module, "BASE", tmp_path)
    saved = main_module.save_result(deepcopy(integrated))
    assert saved.name == "step9_portfolio.json"
    assert not (tmp_path / "output" / "step8_audit.json").exists()


def test_invalid_portfolio_input_is_recoverable_without_discarding_the_audit(graph_modes):
    audit, _, _, _ = graph_modes
    state = deepcopy(audit)
    state["portfolio_config"] = {"price_multiplier": -1}
    original = deepcopy(state)
    update = make_portfolio_handoff_node()(state)
    assert state == original
    assert update["portfolio_plan"] is None
    assert update["portfolio_error"]["type"] == "input_validation"
    assert update["workflow_status"] == "portfolio_needs_input"
    assert update["next_step"]["name"] == "build_portfolio"
    assert update["next_step"]["implemented"] is True
    assert update["next_step"]["status"] == "needs_input"
    assert update["audit_workflow_status"] == audit["workflow_status"]
    assert update["audit_next_step"] == audit["next_step"]
    assert not any(field in update for field in _AUDIT_FIELDS)


def test_a_main_graph_input_error_still_returns_complete_audit_results():
    result, _ = _run_graph(portfolio_config={"price_multiplier": -1})
    assert result["workflow_status"] == "portfolio_needs_input"
    assert result["portfolio_plan"] is None
    assert result["portfolio_error"]["type"] == "input_validation"
    assert result["audited_talents"] and result["pricing_candidates"]
    assert result["next_step"]["name"] == "build_portfolio"
    assert result["audit_next_step"]["name"] == "build_portfolio"


def test_a_clear_customer_budget_revision_is_automatically_applied_with_its_source():
    result, _ = _run_graph(client_answers="本轮总预算调整为100元，其他要求不变。")
    assert result["brief"]["total_budget"] == 350000
    assert result["portfolio_error"] is None
    plan = result["portfolio_plan"]
    assert result["workflow_status"] == "portfolio_" + plan["status"]
    assert plan["budget"]["limit"] == 100
    assert plan["budget"]["total_upper"] <= 100
    assert "client_answers" in plan["budget"]["source"]
    assert result["audited_talents"] and result["pricing_candidates"]
    assert result["audit_next_step"]["name"] == "build_portfolio"


def test_confirmed_budget_override_uses_its_source_without_mutating_the_original_brief():
    answer = "本轮总预算调整为100元，其他要求不变。"
    config = {"budget": {"value": 100, "source": "客户补答：" + answer}}
    result, _ = _run_graph(client_answers=answer, portfolio_config=config)
    plan = result["portfolio_plan"]
    assert result["brief"]["total_budget"] == 350000
    assert result["portfolio_error"] is None
    assert plan["budget"]["limit"] == 100
    assert plan["budget"]["source"] == config["budget"]["source"]
    assert plan["budget"]["total_upper"] <= 100
    assert plan["budget"]["within_budget"] is True
    assert result["audited_talents"] and result["pricing_candidates"]
    assert result["next_step"]["status"] == "no_portfolio"


def test_nonmapping_kpi_overrides_are_an_input_error_and_keep_completed_audit():
    result, _ = _run_graph(client_answers="合作次数以近3个月为准。",
                           portfolio_config={"targets": ["invalid-target"]})
    assert result["workflow_status"] == "portfolio_needs_input"
    assert result["portfolio_plan"] is None
    assert result["portfolio_error"]["type"] == "input_validation"
    assert result["audited_talents"] and result["pricing_candidates"]


def test_recoverable_portfolio_error_saves_to_step9_without_overwriting_audit(tmp_path, monkeypatch, graph_modes):
    audit, _, _, _ = graph_modes
    state = deepcopy(audit)
    state["portfolio_config"] = {"price_multiplier": -1}
    state.update(make_portfolio_handoff_node()(state))
    monkeypatch.setattr(main_module, "BASE", tmp_path)
    audit_path = main_module.save_result(deepcopy(audit))
    original = audit_path.read_bytes()
    path = main_module.save_result(state)
    assert path.name == "step9_portfolio.json"
    assert audit_path.read_bytes() == original
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["portfolio_error"] == state["portfolio_error"]
    assert saved["portfolio_plan"] is None
    assert saved["audited_talents"] == audit["audited_talents"]
    assert saved["audit_pending"] == audit["audit_pending"]


def test_internal_solver_errors_are_not_misrepresented_as_customer_input_errors(monkeypatch, graph_modes):
    import agent.nodes.portfolio as portfolio_nodes

    def internal_failure(*_args, **_kwargs):
        raise RuntimeError("internal budget invariant failed")

    monkeypatch.setattr(portfolio_nodes, "build_portfolio_plan", internal_failure)
    with pytest.raises(RuntimeError, match="internal budget invariant failed"):
        make_portfolio_handoff_node()(deepcopy(graph_modes[0]))


def test_cli_displays_audit_portfolio_costs_and_unmet_kpis_without_claiming_approval(capsys, graph_modes):
    _, result, _, _ = graph_modes
    original = deepcopy(result)
    main_module.render_shortlist(result)
    output = capsys.readouterr().out
    assert result == original
    assert "审核结果" in output
    assert "投放组合" in output
    assert "建议总价" in output
    assert "区间上限" in output
    assert "曝光" in output and "互动" in output and "缺口" in output
    assert "未核实" in output
    assert "人工审核" in output
    assert "下一步：写建议" in output
    assert "投放组合尚未实现" not in output
    assert "人工审核已通过" not in output
    assert "全部KPI已达标" not in output
    plan = result["portfolio_plan"]
    for key in ("total_suggested", "total_upper"):
        assert f"{plan['budget'][key]:,.0f}" in output


def test_cli_reports_input_error_while_preserving_audit_and_risk_display(capsys, graph_modes):
    result = deepcopy(graph_modes[0])
    result["portfolio_config"] = {"price_multiplier": -1}
    result.update(make_portfolio_handoff_node()(result))
    main_module.render_shortlist(result)
    output = capsys.readouterr().out
    assert "审核结果" in output
    assert "投放组合" in output
    assert "price_multiplier" in output or "输入" in output or "参数" in output
    assert "未核实" in output
    assert "人工审核已通过" not in output


def test_shared_main_graph_can_use_the_downstream_return_hook_without_rerunning_upstream(graph_modes):
    seen = []

    def review(state):
        seen.append(state["portfolio_plan"]["revision"])
        return ({"action": "revise", "review_id": "discount", "feedback": "九折"}
                if len(seen) == 1 else {"action": "continue"})

    audit, _, baseline_calls, _ = graph_modes
    result, sequence = _run_graph(review_node=review)
    assert seen == [1, 2]
    assert result["portfolio_plan"]["revision"] == 2
    assert result["portfolio_plan"]["pricing_policy"]["price_multiplier"] == 0.9
    assert result["portfolio_review_status"] == "accepted_for_handoff"
    assert result["portfolio_plan"]["approval_status"] == "pending_final_human_review"
    assert sequence.calls == baseline_calls.calls
    for field in _AUDIT_FIELDS:
        assert result.get(field) == audit.get(field), field
