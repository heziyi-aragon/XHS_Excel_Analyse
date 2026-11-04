"""首轮审核接线：档位分流、待处理任务、下游候选和 CLI 导出。"""
from copy import deepcopy
import json

import pytest
from langgraph.types import Command

from agent.graph import build_graph
from agent.nodes.audit import AuditNodes
from agent.requirements import LLMClient, LLMServiceError
from main import render_shortlist, save_result
from tests.test_requirements import DATA, complete_profile, reason_draft, valid_policy


def _record(nickname, band="mid", **updates):
    return {
        "xhs_id": None, "nickname": nickname, "attempt": 1, "band": band,
        "stance": "reject" if band == "clear_no" else "recommend",
        "tag": "数据差" if band == "clear_no" else "进店好",
        "sentence": "进店量较高，但成本还有优化空间。", "error": None,
        "talent": {"visit_store_med": 800, "cpuv": 10},
        "score_parts": [], **updates,
    }


def _state(records):
    return {
        "evaluated_talents": records,
        "brief": {"raw": {"达人要求": "图文优先"}},
        "requirement_profile": {"soft_preferences": ["图文优先"]},
        "approved_policy": {"criteria": [{"field": "cpuv", "operator": "minimize"}]},
    }


def _nodes(transport=None):
    return AuditNodes(LLMClient(transport=transport or (lambda *_: {"verdict": "pass", "issues": []})))


def test_only_mid_band_calls_audit_and_only_ready_recommendations_reach_pricing():
    seen = []

    def transport(_system, payload):
        seen.append(payload["candidate"]["nickname"])
        return {"verdict": "pass", "issues": []}

    state = _state([_record("高", "clear_yes"), _record("中"), _record("低", "clear_no")])
    original = deepcopy(state)
    result = _nodes(transport).audit_reasons(state)
    assert seen == ["中"]
    assert [row["audit_status"] for row in result["audited_talents"]] == ["skipped_clear_band", "passed", "skipped_clear_band"]
    assert [row["nickname"] for row in result["audit_results"]] == ["中"]
    assert {row["nickname"] for row in result["pricing_candidates"]} == {"高", "中"}
    assert result["workflow_status"] == "audit_complete"
    assert result["next_step"]["name"] == "build_portfolio"
    assert result["next_step"]["implemented"] is False
    assert state == original


def test_client_withdrawal_reaudits_and_clears_old_exclusion_task_without_changing_the_reason():
    record = _record("补答后复审", talent={"cpuv": 10, "main_format": "纯vlog生活号"})
    state = _state([record])
    state["requirement_profile"]["exclusions"] = ["纯vlog生活号"]
    seen = []

    def transport(_system, payload):
        answer = payload["context"]["client_answers"]
        seen.append(answer)
        if answer:
            return {"verdict": "pass", "issues": []}
        return {"verdict": "reject", "issues": [{
            "code": "client_exclusion", "message": "原要求排除纯vlog生活号。", "fields": ["main_format"],
        }]}

    nodes = _nodes(transport)
    state.update(nodes.audit_reasons(state))
    assert state["audit_pending"][0]["action"] == "rewrite_reason"
    state["client_answers"] = "本轮撤回纯vlog生活号的排除限制，其他要求不变。"
    state.update(nodes.audit_reasons(state))
    assert seen == ["", state["client_answers"]]
    assert state["audit_pending"] == []
    assert state["audit_results"][0]["verdict"] == "pass"
    assert [item["result"]["verdict"] for item in state["audit_history"]] == ["reject", "pass"]
    assert state["evaluated_talents"] == [record]
    assert state["pricing_candidates"][0]["sentence"] == record["sentence"]


@pytest.mark.parametrize("band", ["clear_yes", "mid", "clear_no"])
def test_missing_required_metric_in_any_band_creates_a_refill_task(band):
    def transport(*_args):
        assert band == "mid", "明确档只检查必要数据，不调用模型理由审核"
        return {"verdict": "pass", "issues": []}

    record = _record("缺数", band, talent={"cpuv": None})
    result = _nodes(transport).audit_reasons(_state([record]))
    task = result["audit_pending"][0]
    assert task["action"] == "supplement_data"
    assert task["fields"] == ["cpuv"]
    assert result["pricing_candidates"] == []
    assert result["workflow_status"] == "audit_needs_resolution"
    assert result["next_step"]["name"] == "supplement_data"


def test_a_reject_assessment_that_passes_audit_is_not_a_pricing_candidate():
    result = _nodes().audit_reasons(_state([_record("不推荐", stance="reject", tag="价格贵")]))
    assert result["audit_results"][0]["verdict"] == "pass"
    assert result["pricing_candidates"] == []
    assert result["next_step"]["status"] == "no_candidates"


@pytest.mark.parametrize("attempt,action", [(1, "rewrite_reason"), (2, "rewrite_limit_reached")])
def test_reason_rewrite_task_respects_the_existing_evaluation_round_limit(attempt, action):
    result = _nodes().audit_reasons(_state([_record("标签错误", tag="价格贵", attempt=attempt)]))
    assert result["audit_pending"][0]["action"] == action
    assert result["audit_pending"][0]["attempt"] == attempt


def test_an_unsupported_external_claim_is_returned_to_evaluation_for_correction():
    def transport(*_args):
        return {"verdict": "reject", "issues": [{
            "code": "data_conflict", "message": "未提供品牌合作次数，无法核实未合作的断言。", "fields": [],
        }]}

    result = _nodes(transport).audit_reasons(_state([_record("缺证据")]))
    assert result["audit_pending"][0]["action"] == "rewrite_reason"


def test_invalid_audit_answers_keep_an_execution_error_without_cancelling_other_people():
    def transport(_system, payload):
        if payload["candidate"]["nickname"] == "失败":
            return {"verdict": "reject", "issues": []}
        return {"verdict": "pass", "issues": []}

    result = _nodes(transport).audit_reasons(_state([_record("失败"), _record("正常")]))
    assert [item["nickname"] for item in result["audit_results"]] == ["正常"]
    assert result["audited_talents"][0]["audit_status"] == "execution_error"
    assert result["audit_pending"][0]["action"] == "retry_audit"
    assert result["workflow_status"] == "audit_needs_resolution"
    assert result["next_step"]["name"] != "build_portfolio"


def test_service_connection_failure_still_propagates_to_the_existing_cli_handler():
    def transport(*_args):
        raise LLMServiceError("模型暂时不可用")

    with pytest.raises(LLMServiceError):
        _nodes(transport).audit_reasons(_state([_record("连接失败")]))


def test_empty_shortlist_does_not_call_a_model_or_claim_to_have_pricing_candidates():
    def transport(*_args):
        pytest.fail("空名单不应请求模型")

    result = _nodes(transport).audit_reasons(_state([]))
    assert result["audit_results"] == []
    assert result["pricing_candidates"] == []
    assert result["next_step"]["status"] == "no_candidates"


@pytest.mark.parametrize("records", [
    [_record("重复"), _record("重复")],
    [_record("未知", band="invented")],
    [_record("轮次丢失", attempt=None)],
])
def test_unmatchable_or_invalid_input_cannot_silently_produce_a_completed_audit(records):
    with pytest.raises(ValueError):
        _nodes().audit_reasons(_state(records))


def test_export_and_console_include_unresolved_audit_tasks(tmp_path, monkeypatch, capsys):
    import main

    state = _state([_record("缺数", "clear_yes", talent={"cpuv": None})])
    result = {**state, **_nodes().audit_reasons(state)}
    monkeypatch.setattr(main, "BASE", tmp_path)
    path = save_result(result)
    assert path.name == "step8_audit.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["evaluated_talents"] == state["evaluated_talents"]
    assert saved["audit_pending"][0]["fields"] == ["cpuv"]
    assert saved["audited_talents"][0]["audit_status"] == "needs_resolution"
    render_shortlist(result)
    printed = capsys.readouterr().out
    assert "补充数据（进店成本）" in printed
    assert "单纯补数失败时标记放行" in printed


def test_full_graph_can_reuse_one_client_while_keeping_audit_requests_separate():
    audit_names = []

    def transport(_system, payload):
        if "structured_brief" in payload:
            return complete_profile()
        if "candidate" in payload:
            if "assessment" in payload["candidate"]:
                audit_names.append(payload["candidate"]["nickname"])
                return {"verdict": "pass", "issues": []}
            return reason_draft()
        return valid_policy()

    graph = build_graph(LLMClient(transport=transport))
    config = {"configurable": {"thread_id": "audit-shared-client"}}
    initial = graph.invoke({
        "brief_path": str(DATA / "甲方投放需求单_屿光精华.xlsx"),
        "talent_path": str(DATA / "蒲公英达人数据_屿光精华.xlsx"),
    }, config=config)
    assert initial["__interrupt__"][0].value["type"] == "policy_review"
    result = graph.invoke(Command(resume={"action": "approve"}), config=config)
    mid_names = {row["nickname"] for row in result["evaluated_talents"] if row["band"] == "mid"}
    assert audit_names and set(audit_names).issubset(mid_names)
    assert len(result["audited_talents"]) == len(result["evaluated_talents"])
    assert result["audit_workflow_status"] == "audit_complete_with_incomplete_data"
    assert result["workflow_status"] == "portfolio_" + result["portfolio_plan"]["status"]
    assert result["manual_review_queue"] == result["audit_pending"] == []
    assert result["supplement_results"]
    assert result["next_step"]["name"] == "write_suggestions"
    assert result["audit_next_step"]["name"] == "build_portfolio"
    flagged_recommendations = {
        row["nickname"] for row in result["audited_talents"]
        if row.get("data_incomplete") and row["stance"] == "recommend"
    }
    assert flagged_recommendations
    assert flagged_recommendations.issubset({row["nickname"] for row in result["pricing_candidates"]})
    assert all(row["stance"] == "recommend" for row in result["pricing_candidates"])
