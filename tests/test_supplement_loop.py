"""补数成功回评估：来源不变、整批标尺一致、旧理由隔离和有限循环。"""
from copy import deepcopy
import json

import pytest
from langgraph.types import Command

from agent.graph import build_graph
from agent.nodes.audit_handoff import finalize_audit
from agent.nodes.evaluation import collect_reasons
from agent.nodes.rewrite import route_after_audit
from agent.nodes.supplement import route_after_supplement, supplement_data
from agent.requirements import LLMClient
from agent.screening import clean_talents, score_candidates
from main import render_shortlist, save_result
from tests.test_audit_nodes import _nodes
from tests.test_requirements import DATA, complete_profile, reason_draft, valid_policy


def _fixture_state(*, partial=False, at_limit=False):
    raw = [
        {"xhs_id": "target", "nickname": "待补", "cpuv": 5, "visit_store_med": None if partial else 50},
        {"xhs_id": "low", "nickname": "低", "cpuv": 10, "visit_store_med": 0},
        {"xhs_id": "high", "nickname": "高", "cpuv": 20, "visit_store_med": 100},
    ]
    work = deepcopy(raw)
    work[0]["cpuv"] = None
    policy = {"criteria": [
        {"field": "cpuv", "operator": "minimize", "weight": 0.4, "hard": False},
        {"field": "visit_store_med", "operator": "maximize", "weight": 0.6, "hard": False},
    ]}
    pool = clean_talents(work)
    scored = score_candidates(pool, policy)
    if at_limit:
        scored[0]["attempt"] = 2
    drafts = [{"xhs_id": row["xhs_id"], "nickname": row["nickname"], "attempt": row["attempt"],
               "data_revision": 0, "error": None, "source": "test", **reason_draft()}
              for row in scored if row["band"] == "mid"]
    state = {
        "talent_path": "test-fixture", "talents": raw, "cleaned_talents": deepcopy(pool),
        "screened_talents": pool, "scored_talents": scored, "reason_results": drafts,
        "approved_policy": policy, "requirement_profile": {}, "brief": {},
    }
    state.update(collect_reasons(state))
    state.update(_nodes().audit_reasons(state))
    return state


def test_recovered_metric_rescores_the_same_pool_without_mutating_source_or_reusing_old_reason():
    state = _fixture_state()
    original = deepcopy(state)
    assert route_after_audit(state) == "supplement"
    updated = {**state, **supplement_data(state)}
    assert state == original
    assert updated["talents"] == original["talents"]
    assert updated["scored_talents"][0]["talent"]["cpuv"] == 5
    assert updated["scored_talents"][1]["score"] != original["scored_talents"][1]["score"]
    assert updated["scored_talents"][0]["data_revision"] == 1
    assert updated["supplement_history"][0]["values"] == {"cpuv": 5}
    assert updated["supplement_history"][0]["input_check"]["source_path"] == "test-fixture"
    updated["reason_results"].append({
        "xhs_id": "target", "nickname": "待补", "attempt": 2, "data_revision": 0,
        "stance": "recommend", "tag": "进店好", "sentence": "旧数据版本不能覆盖新数据。", "error": None,
    })
    merged = collect_reasons(updated)
    target = next(row for row in merged["evaluated_talents"] if row["xhs_id"] == "target")
    assert target["error"] and target["sentence"] is None
    assert target["talent"]["cpuv"] == 5  # 尚未运行新评估，不能借用旧理由冒充已评估。


def test_whitespace_missing_metric_can_be_restored_without_overwriting_any_real_value():
    state = _fixture_state()
    for collection in ("evaluated_talents", "audited_talents", "scored_talents"):
        next(row for row in state[collection] if row["xhs_id"] == "target")["talent"]["cpuv"] = "  "
    original = deepcopy(state)
    result = supplement_data(state)
    assert result["supplement_history"][0]["values"] == {"cpuv": 5}
    assert next(row for row in result["scored_talents"] if row["xhs_id"] == "target")["talent"]["cpuv"] == 5
    assert state == original


def test_partial_recovery_keeps_remaining_missing_fields_and_does_not_query_forever():
    state = _fixture_state(partial=True)
    state.update(supplement_data(state))
    assert state["supplement_history"][0]["status"] == "partial"
    assert state["scored_talents"][0]["talent"]["visit_store_med"] is None
    # 这位达人补值后落入明确档，无需再调用模型，仍必须检查剩余必要缺数。
    assert state["scored_talents"][0]["band"] == "clear_yes"
    state.update(collect_reasons(state))
    state.update(_nodes().audit_reasons(state))
    assert route_after_audit(state) == "supplement"
    state.update(supplement_data(state))
    assert route_after_supplement(state) == "finalize_audit"
    assert route_after_audit(state) == "finish"
    state.update(finalize_audit(state))
    target = next(row for row in state["audited_talents"] if row["xhs_id"] == "target")
    assert target["missing_data_fields"] == ["visit_store_med"]
    assert target["talent"]["cpuv"] == 5 and target["release_status"] == "flagged_release"
    assert [row["status"] for row in state["supplement_results"]] == ["partial", "failed"]
    assert len(state["supplement_history"]) == 2


def test_recovery_at_the_existing_limit_holds_the_record_instead_of_using_stale_reason():
    state = _fixture_state(at_limit=True)
    state.update(supplement_data(state))
    state.update(collect_reasons(state))
    state.update(_nodes().audit_reasons(state))
    assert route_after_audit(state) == "finish"
    state.update(finalize_audit(state))
    target = next(row for row in state["audited_talents"] if row["xhs_id"] == "target")
    assert target["attempt"] == 2 and target["data_revision"] == 1
    assert target["release_status"] == "held_for_review"
    assert target["talent"]["cpuv"] == 5 and target["error"]
    assert all(row["xhs_id"] != "target" for row in state["pricing_candidates"])
    assert state["supplement_results"][0]["status"] == "success"
    assert state["next_step"]["name"] == "build_portfolio"


def test_recovered_values_cannot_bypass_the_existing_hard_conditions():
    state = _fixture_state()
    state["approved_policy"]["criteria"].append({"field": "cpuv", "operator": "gte", "value": 8, "hard": True, "weight": 0})
    state.update(supplement_data(state))
    state.update(collect_reasons(state))
    state.update(_nodes().audit_reasons(state))
    repeated = _nodes().audit_reasons(state)
    assert len(repeated["audit_history"]) == len(state["audit_history"])
    state.update(finalize_audit(state))
    assert state["audit_pending"][0]["reason_code"] == "post_supplement_policy_conflict"
    assert all(row["xhs_id"] != "target" for row in state["pricing_candidates"])


@pytest.mark.parametrize("change", ["missing_baseline", "wrong_identity", "existing_value"])
def test_invalid_recovery_cannot_silently_overwrite_the_scoring_baseline(change):
    state = _fixture_state()
    if change == "missing_baseline":
        state.pop("scored_talents")
    elif change == "wrong_identity":
        state["scored_talents"][0]["xhs_id"] = "other"
    else:
        next(row for row in state["audited_talents"] if row["xhs_id"] == "target")["talent"]["cpuv"] = 7
    with pytest.raises(ValueError):
        supplement_data(state)


def test_success_report_is_saved_and_not_printed_as_a_failed_lookup(tmp_path, monkeypatch, capsys):
    import main

    state = _fixture_state(at_limit=True)
    state.update(supplement_data(state))
    state.update(collect_reasons(state))
    state.update(_nodes().audit_reasons(state))
    state.update(finalize_audit(state))
    monkeypatch.setattr(main, "BASE", tmp_path)
    saved = json.loads(save_result(state).read_text(encoding="utf-8"))
    assert saved["supplement_history"][0]["values"] == {"cpuv": 5}
    render_shortlist(state)
    output = capsys.readouterr().out
    assert "已查回的缺失字段" in output
    assert "补数失败，标记放行" not in output


@pytest.mark.parametrize("already_rewritten", [False, True])
def test_main_graph_recovers_real_file_values_from_an_in_memory_hole(monkeypatch, already_rewritten):
    import agent.nodes.screening as screening_nodes

    path = DATA / "蒲公英达人数据_屿光精华.xlsx"
    original_file = path.read_bytes()
    original_clean = screening_nodes.clean_talents
    target = "成分党阿澈"

    def clean_with_test_hole(raw):
        rows = deepcopy(raw)
        next(row for row in rows if row["nickname"] == target)["cpuv"] = None
        return original_clean(rows)

    monkeypatch.setattr(screening_nodes, "clean_talents", clean_with_test_hole)
    writes, audits = [], []

    def evaluation_transport(_system, payload):
        if "structured_brief" in payload:
            return complete_profile()
        if "candidate" not in payload:
            return valid_policy()
        candidate = payload["candidate"]
        writes.append(deepcopy(candidate))
        if already_rewritten and candidate["nickname"] == target and not candidate.get("review_feedback"):
            return {"stance": "recommend", "tag": "进店好", "sentence": "进店中位数99999，但成本缺失。"}
        return reason_draft()

    def audit_transport(_system, payload):
        candidate = payload["candidate"]
        audits.append(deepcopy(candidate))
        if "99999" in candidate["assessment"]["sentence"]:
            return {"verdict": "reject", "issues": [{"code": "data_conflict", "message": "99999与原值不符。", "fields": ["visit_store_med"]}]}
        return {"verdict": "pass", "issues": []}

    graph = build_graph(LLMClient(transport=evaluation_transport), audit_llm_client=LLMClient(transport=audit_transport))
    config = {"configurable": {"thread_id": "supplement-real-file-" + str(already_rewritten)}}
    initial = graph.invoke({"brief_path": str(DATA / "甲方投放需求单_屿光精华.xlsx"), "talent_path": str(path)}, config=config)
    assert initial["__interrupt__"][0].value["type"] == "policy_review"
    result = graph.invoke(Command(resume={"action": "approve"}), config=config)
    source = next(row for row in result["talents"] if row["nickname"] == target)
    latest = next(row for row in result["audited_talents"] if row["nickname"] == target)
    found = next(row for row in result["supplement_results"] if row["nickname"] == target)
    assert found["status"] == "success" and found["values"] == {"cpuv": source["cpuv"]}
    assert source["cpuv"] is not None and latest["talent"]["cpuv"] == source["cpuv"]
    assert found["input_check"]["source_path"] == str(path)
    assert latest["data_revision"] == 1 and latest["attempt"] == 2
    assert result["next_step"]["name"] == "write_suggestions"
    assert result["audit_next_step"]["name"] == "build_portfolio"
    assert result["workflow_status"] == "portfolio_" + result["portfolio_plan"]["status"]
    assert len([row for row in writes if row["nickname"] == target]) == 2
    assert all(sum(row["nickname"] == name for row in writes) == 1 for name in {row["nickname"] for row in writes} - {target})
    assert all(sum(row["nickname"] == name for row in audits) == 1 for name in {row["nickname"] for row in audits} - {target})
    if already_rewritten:
        assert latest["release_status"] == "held_for_review" and latest["error"]
    else:
        assert latest["audit_status"] == "passed" and not latest["error"]
        assert not any(issue["code"] == "data_missing" for issue in latest["audit_issues"])
    assert path.read_bytes() == original_file
