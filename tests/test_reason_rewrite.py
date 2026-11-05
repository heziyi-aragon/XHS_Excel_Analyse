"""重写回路：携带审核意见、两轮上限、单条暂存与复审复用边界。"""
from collections import Counter
from copy import deepcopy

import pytest
from langgraph.types import Command

from agent.graph import build_graph
from agent.nodes.audit_handoff import finalize_audit
from agent.nodes.evaluation import EvaluationNodes, collect_reasons, route_reasons
from agent.nodes.rewrite import prepare_reason_rewrites, route_after_audit
from agent.requirements import LLMClient
from tests.test_audit_nodes import _nodes, _record, _state
from tests.test_requirements import DATA, complete_profile, reason_draft, valid_policy


def _task(record, action="rewrite_reason"):
    return {
        "xhs_id": record.get("xhs_id"), "nickname": record["nickname"],
        "attempt": record["attempt"], "action": action, "fields": [],
        "issues": [{"code": "data_conflict", "message": "理由中的进店量与原指标不一致。", "fields": ["visit_store_med"]}],
    }


def test_prepare_only_queues_target_people_and_preserves_the_reviewed_version():
    target, other = _record("待改"), _record("已通过")
    state = {"evaluated_talents": [target, other], "audit_pending": [_task(target), _task(other, "rewrite_limit_reached")]}
    original = deepcopy(state)
    result = prepare_reason_rewrites(state)
    queue = result["reason_queue"]
    assert len(queue) == 1 and queue[0]["nickname"] == "待改"
    assert queue[0]["attempt"] == 2
    assert queue[0]["review_feedback"]["previous_assessment"]["sentence"] == target["sentence"]
    assert queue[0]["review_feedback"]["issues"][0]["fields"] == ["visit_store_med"]
    queue[0]["talent"]["cpuv"] = 999
    assert state == original


@pytest.mark.parametrize("action", ["rewrite_limit_reached", "retry_audit"])
def test_non_rewrite_tasks_do_not_create_an_automatic_loop(action):
    assert route_after_audit({"audit_pending": [_task(_record("任务"), action)]}) == "finish"


@pytest.mark.parametrize("case", ["cap", "stale", "clear", "duplicate", "missing"])
def test_invalid_rewrite_targets_cannot_spend_an_extra_evaluation_round(case):
    record = _record("目标", attempt=2 if case == "cap" else 1)
    task = _task(record)
    if case == "stale":
        task["attempt"] = 2
    elif case == "clear":
        record["band"] = "clear_yes"
    elif case == "missing":
        task["nickname"] = "不存在"
    tasks = [task, deepcopy(task)] if case == "duplicate" else [task]
    with pytest.raises(ValueError):
        prepare_reason_rewrites({"evaluated_talents": [record], "audit_pending": tasks})


@pytest.mark.parametrize("change", ["none", "metric", "client_answer"])
def test_audit_reuse_requires_both_unchanged_assessment_and_unchanged_context(change):
    calls = []

    def transport(*_args):
        calls.append(1)
        return {"verdict": "pass", "issues": []}

    node = _nodes(transport)
    state = _state([_record("候选")])
    first = node.audit_reasons(state)
    second_state = deepcopy({**state, **first})
    if change == "metric":
        second_state["evaluated_talents"][0]["talent"]["cpuv"] = 20
    elif change == "client_answer":
        second_state["client_answers"] = "客户修改了本轮受众要求。"
    second = node.audit_reasons(second_state)
    assert len(calls) == (1 if change == "none" else 2)
    assert len(second["audit_history"]) == len(calls)
    assert len(second["audit_results"]) == 1


@pytest.mark.parametrize("problem", ["invalid_tag", "data_conflict", "client_exclusion"])
def test_mixed_missing_data_and_other_problems_are_rewritten_before_flagged_release(problem):
    target = _record(
        "待改且缺数", tag="发明标签" if problem == "invalid_tag" else "进店好",
        sentence="进店中位数99999，但缺少进店成本。",
        talent={"visit_store_med": 800, "cpuv": None, "main_format": "vlog"},
    )
    healthy = _record("正常明确档", "clear_yes")
    state = _state([target, healthy])
    state.update(
        scored_talents=deepcopy([target, healthy]), reason_results=[deepcopy(target)],
        talents=[{"nickname": target["nickname"], **target["talent"]}],
    )
    original_baseline = deepcopy(state["scored_talents"])
    audit_calls, rewrite_calls = [], []

    def audit_transport(_system, payload):
        candidate = payload["candidate"]
        audit_calls.append(deepcopy(candidate))
        if "99999" in candidate["assessment"]["sentence"]:
            return {"verdict": "reject", "issues": [{
                "code": problem, "message": "理由的数字错误或推荐违反排除项，需修正结论。",
                "fields": ["visit_store_med"] if problem == "data_conflict" else ["main_format"],
            }]}
        return {"verdict": "pass", "issues": []}

    nodes = _nodes(audit_transport)
    state.update(nodes.audit_reasons(state))
    assert route_after_audit(state) == "rewrite"
    task = state["audit_pending"][0]
    assert {issue["code"] for issue in task["issues"]} == {problem, "data_missing"}
    assert state["pricing_candidates"][0]["nickname"] == "正常明确档"
    assert all(not row.get("release_status") for row in state["audited_talents"])
    state.update(prepare_reason_rewrites(state))

    def write_transport(_system, payload):
        rewrite_calls.append(deepcopy(payload))
        feedback = payload["candidate"]["review_feedback"]
        assert len(feedback["issues"]) == 2
        return {"stance": "reject", "tag": "数据不稳", "sentence": "进店中位数为800，但缺少进店成本，暂不推荐。"}

    writer = EvaluationNodes(LLMClient(transport=write_transport))
    sends = route_reasons(state)
    assert len(sends) == 1
    state["reason_results"] += writer.write_one_reason(sends[0].arg)["reason_results"]
    state.update(collect_reasons(state))
    state.update(nodes.audit_reasons(state))
    assert state["audit_pending"][0]["action"] == "supplement_data"
    state.update(finalize_audit(state))
    flagged = next(row for row in state["audited_talents"] if row["nickname"] == target["nickname"])
    assert flagged["attempt"] == 2 and flagged["release_status"] == "flagged_release"
    assert {issue["code"] for issue in flagged["audit_issues"]} == {"data_missing"}
    assert flagged["talent"]["cpuv"] is None and "99999" not in flagged["sentence"]
    assert state["workflow_status"] == "audit_complete_with_incomplete_data"
    assert state["next_step"]["name"] == "build_portfolio"
    assert state["manual_review_queue"] == state["audit_pending"] == []
    assert len(rewrite_calls) == 1 and len(audit_calls) == (1 if problem == "invalid_tag" else 2)
    assert state["scored_talents"] == original_baseline
    assert [row["nickname"] for row in state["pricing_candidates"]] == ["正常明确档"]


@pytest.mark.parametrize("mode", ["fixed", "still_wrong", "invalid_output"])
def test_real_graph_rewrites_once_reaudits_only_changed_person_and_preserves_history(mode):
    target = "早八小满"
    evaluations, audits = Counter(), Counter()
    feedback_seen = []

    def evaluation_transport(_system, payload):
        if "structured_brief" in payload:
            return complete_profile()
        if "candidate" not in payload:
            return valid_policy()
        candidate = payload["candidate"]
        nickname = candidate["nickname"]
        evaluations[nickname] += 1
        feedback = candidate.get("review_feedback")
        if feedback:
            assert nickname == target
            assert "99999" in feedback["previous_assessment"]["sentence"]
            assert feedback["issues"][0]["metrics"] == ["进店中位数"]
            feedback_seen.append(deepcopy(feedback))
            if mode == "fixed":
                return {"stance": "reject", "tag": "价格贵", "sentence": "进店量尚可，但进店成本偏高，暂不推荐。"}
            if mode == "invalid_output":
                return {"stance": "recommend", "tag": "发明标签", "sentence": "没有按规则修正。"}
        if nickname == target:
            return {"stance": "recommend", "tag": "进店好", "sentence": "进店中位数达到99999，但成本还有优化空间。"}
        return reason_draft()

    def audit_transport(_system, payload):
        candidate = payload["candidate"]
        nickname = candidate["nickname"]
        audits[nickname] += 1
        if "99999" in candidate["assessment"]["sentence"]:
            return {"verdict": "reject", "issues": [{
                "code": "data_conflict", "message": "理由中的99999与原始进店中位数不一致。", "fields": ["visit_store_med"],
            }]}
        return {"verdict": "pass", "issues": []}

    graph = build_graph(
        LLMClient(transport=evaluation_transport),
        audit_llm_client=LLMClient(transport=audit_transport),
    )
    config = {"configurable": {"thread_id": "rewrite-" + mode}}
    initial = graph.invoke({
        "brief_path": str(DATA / "甲方投放需求单_屿光精华.xlsx"),
        "talent_path": str(DATA / "蒲公英达人数据_屿光精华.xlsx"),
    }, config=config)
    assert initial["__interrupt__"][0].value["type"] == "policy_review"
    result = graph.invoke(Command(resume={"action": "approve"}), config=config)
    latest = next(row for row in result["evaluated_talents"] if row["nickname"] == target)
    versions = [row for row in result["reason_results"] if row["nickname"] == target]
    history = [entry for entry in result["audit_history"] if entry["result"]["nickname"] == target]
    assert latest["attempt"] == 2
    assert [row["attempt"] for row in versions] == [1, 2]
    assert [entry["result"]["attempt"] for entry in history] == [1, 2]
    assert "99999" in history[0]["assessment"]["sentence"]
    assert evaluations[target] == (3 if mode == "invalid_output" else 2)
    assert audits[target] == (1 if mode == "invalid_output" else 2)
    assert feedback_seen
    assert all(count == 1 for name, count in evaluations.items() if name != target)
    assert all(count == 1 for name, count in audits.items() if name != target)
    assert all(row["nickname"] != target for row in result["pricing_candidates"])
    if mode == "fixed":
        assert latest["stance"] == "reject"
        assert history[-1]["result"]["verdict"] == "pass"
        assert all(task["nickname"] != target for task in result["audit_pending"])
    else:
        task = next(task for task in result["audit_pending"] if task["nickname"] == target)
        assert task["action"] == "rewrite_limit_reached"
        assert history[-1]["result"]["verdict"] == "reject"
        if mode == "invalid_output":
            assert latest["error"] and latest["tag"] is None
    # 这份模拟表另有缺数达人：按 workflow 标记放行，不提前卡到人工节点。
    assert result["supplement_results"]
    assert result["workflow_status"] == "portfolio_" + result["portfolio_plan"]["status"]
    assert result["next_step"]["name"] == "write_suggestions"
    if mode == "fixed":
        assert result["audit_workflow_status"] == "audit_complete_with_incomplete_data"
        assert result["audit_next_step"]["name"] == "build_portfolio"
        assert result["manual_review_queue"] == []
    else:
        assert result["audit_workflow_status"] == "audit_complete_with_pending"
        assert result["audit_next_step"]["name"] == "build_portfolio"
        assert result["audit_next_step"]["status"] == "interface_ready"
        assert result["pricing_candidates"]
        assert next(row for row in result["audited_talents"] if row["nickname"] == target)["release_status"] == "held_for_review"
        assert {task["nickname"] for task in result["audit_pending"]} == {target}
        assert result["manual_review_queue"] == []
    assert all(task["action"] != "rewrite_reason" for task in result["audit_pending"])
