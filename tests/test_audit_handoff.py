"""审核交接遵循 workflow：补数失败标记放行，保留原评估及其他审核问题。"""
from copy import deepcopy
import json

import pytest

from agent.nodes.audit_handoff import finalize_audit
from main import render_shortlist, save_result
from tests.test_audit_nodes import _nodes, _record, _state


def _missing_state(raw_rows=None):
    state = _state([_record("缺数达人", "clear_yes", xhs_id="kol-1", talent={"cpuv": None})])
    state["talent_path"] = "static/current_input.xlsx"
    state["talents"] = raw_rows if raw_rows is not None else [{"xhs_id": "kol-1", "nickname": "缺数达人", "cpuv": None}]
    return {**state, **_nodes().audit_reasons(state)}


def test_missing_value_in_the_existing_input_is_flagged_and_released_to_portfolio():
    state = _missing_state()
    original = deepcopy(state)
    result = finalize_audit(state)
    case = result["supplement_results"][0]
    assert result["workflow_status"] == "audit_complete_with_incomplete_data"
    assert result["next_step"]["name"] == "build_portfolio"
    assert result["manual_review_queue"] == result["audit_pending"] == []
    candidate = result["pricing_candidates"][0]
    assert candidate["data_incomplete"] is True
    assert candidate["missing_data_fields"] == ["cpuv"]
    assert candidate["release_status"] == "flagged_release"
    assert candidate["audit_status"] == "needs_resolution"
    assert candidate["talent"]["cpuv"] is None
    assert case["status"] == "failed"
    assert case["action"] == "flagged_release" and case["original_action"] == "supplement_data"
    assert case["data_incomplete"] is True
    assert case["fields"] == ["cpuv"]
    assert case["reason_code"] == "required_data_unavailable"
    assert case["input_check"] == {
        "source_path": "static/current_input.xlsx", "match_field": "xhs_id", "status": "checked",
        "available_values": {}, "unavailable_fields": ["cpuv"], "supplementary_source_status": "not_connected",
    }
    assert case["assessment"]["stance"] == "recommend"
    assert state == original  # 原指标、分数、推荐倾向、审核结果及原任务全部保留


def test_inconsistent_loaded_input_holds_only_that_record_for_later_review():
    state = _missing_state([{"xhs_id": "kol-1", "nickname": "原昵称", "cpuv": 0}])
    result = finalize_audit(state)
    case = result["audit_pending"][0]
    assert case["reason_code"] == "input_and_assessment_disagree"
    assert case["input_check"]["available_values"] == {"cpuv": 0}
    assert case["input_check"]["unavailable_fields"] == []
    assert state["audited_talents"][0]["talent"]["cpuv"] is None
    assert result["pricing_candidates"] == []
    assert case["action"] == "restore_input_data"
    assert result["workflow_status"] == "audit_complete_with_pending"
    assert result["audited_talents"][0]["release_status"] == "held_for_review"
    assert result["next_step"]["name"] == "build_portfolio"
    assert result["next_step"]["status"] == "no_candidates"
    assert result["manual_review_queue"] == []


@pytest.mark.parametrize("rows,status", [
    ([{"xhs_id": "other", "nickname": "缺数达人", "cpuv": 8}], "record_not_found"),
    ([{"xhs_id": "kol-1", "cpuv": 8}, {"xhs_id": "kol-1", "cpuv": 9}], "ambiguous_record"),
])
def test_unknown_or_duplicate_identity_does_not_select_the_first_plausible_value(rows, status):
    case = finalize_audit(_missing_state(rows))["supplement_results"][0]
    assert case["input_check"]["status"] == status
    assert case["input_check"]["available_values"] == {}


@pytest.mark.parametrize("value", [None, "-", "/", "—", "N/A", float("nan")])
def test_placeholder_and_nonfinite_source_values_are_not_offered_as_usable_data(value):
    case = finalize_audit(_missing_state([{"xhs_id": "kol-1", "cpuv": value}]))["supplement_results"][0]
    assert case["input_check"]["available_values"] == {}


def test_missing_data_candidates_are_kept_alongside_healthy_candidates():
    state = _missing_state()
    healthy = _record("正常达人", "clear_yes", xhs_id="kol-2")
    healthy["audit_status"] = "skipped_clear_band"
    healthy["audit_issues"] = []
    state["audited_talents"].append(healthy)
    state["pricing_candidates"] = deepcopy(state["audited_talents"])
    result = finalize_audit(state)
    assert [row["nickname"] for row in result["pricing_candidates"]] == ["缺数达人", "正常达人"]
    assert result["pricing_candidates"][0]["data_incomplete"] is True
    assert "data_incomplete" not in result["pricing_candidates"][1]
    assert result["manual_review_queue"] == []


@pytest.mark.parametrize("band", ["clear_yes", "mid", "clear_no"])
@pytest.mark.parametrize("attempt", [1, 2])
def test_flagged_release_applies_to_all_bands_without_changing_stance_or_audit_verdict(band, attempt):
    record = _record("缺数", band, attempt=attempt, talent={"cpuv": None})
    state = _state([record])
    audited = _nodes().audit_reasons(state)
    result = finalize_audit({**state, **audited})
    flagged = result["audited_talents"][0]
    assert flagged["release_status"] == "flagged_release"
    assert flagged["stance"] == record["stance"]
    assert flagged["audit_issues"] == audited["audited_talents"][0]["audit_issues"]
    assert len(result["pricing_candidates"]) == (0 if band == "clear_no" else 1)
    assert result["next_step"]["name"] == "build_portfolio"
    assert result["manual_review_queue"] == []
    if band == "mid":
        assert audited["audit_results"][0]["verdict"] == "reject"


def test_missing_data_does_not_release_a_record_with_an_additional_invalid_tag():
    state = _state([_record("缺数且标签错误", tag="价格贵", talent={"cpuv": None})])
    audited = _nodes().audit_reasons(state)
    assert audited["audit_pending"][0]["action"] == "rewrite_reason"
    assert audited["next_step"]["name"] == "prepare_reason_rewrites"
    with pytest.raises(ValueError):
        finalize_audit({**state, **audited})  # 不能绕过尚可执行的重写，直接补数放行。


def test_clean_audit_goes_to_the_existing_portfolio_interface_without_manual_cases():
    state = _state([_record("正常", "clear_yes")])
    result = finalize_audit({**state, **_nodes().audit_reasons(state)})
    assert result["manual_review_queue"] == []
    assert result["workflow_status"] == "audit_complete"
    assert result["next_step"]["name"] == "build_portfolio"


def test_rewrite_limit_preserves_the_record_without_pretending_it_passed():
    state = _state([_record("重写耗尽", tag="价格贵", attempt=2)])
    result = finalize_audit({**state, **_nodes().audit_reasons(state)})
    case = result["audit_pending"][0]
    assert case["reason_code"] == "revision_limit_reached"
    assert case["data_incomplete"] is False
    assert case["assessment"]["tag"] == "价格贵"
    assert case["action"] == "rewrite_limit_reached"
    assert result["workflow_status"] == "audit_complete_with_pending"
    assert result["audited_talents"][0]["release_status"] == "held_for_review"
    assert result["next_step"]["name"] == "build_portfolio"
    assert result["next_step"]["status"] == "no_candidates"
    assert result["manual_review_queue"] == result["pricing_candidates"] == []


def test_execution_error_stays_unfinished_without_a_fake_verdict_or_manual_node():
    state = _state([_record("审核失败")])
    node = _nodes(lambda *_: {"verdict": "reject", "issues": []})
    result = finalize_audit({**state, **node.audit_reasons(state)})
    case = result["audit_pending"][0]
    assert case["reason_code"] == "audit_execution_error"
    assert case["original_action"] == "retry_audit" and case["error"]
    assert case["data_incomplete"] is False
    assert result["workflow_status"] == "audit_complete_with_pending"
    assert result["audited_talents"][0]["audit_status"] == "execution_error"
    assert result["audited_talents"][0]["release_status"] == "held_for_review"
    assert result["next_step"]["status"] == "no_candidates"
    assert result["manual_review_queue"] == result["pricing_candidates"] == []


@pytest.mark.parametrize("problem", ["rewrite_limit", "invalid_answer", "inconsistent_input"])
def test_a_held_record_does_not_block_healthy_or_flagged_candidates(problem):
    held = _record("暂存达人", xhs_id="held")
    if problem == "rewrite_limit":
        held.update(tag="价格贵", attempt=2)
    elif problem == "inconsistent_input":
        held["talent"]["cpuv"] = None
    records = [
        _record("正常达人", "clear_yes", xhs_id="healthy"),
        _record("缺数放行", "clear_yes", xhs_id="missing", talent={"cpuv": None}),
        held,
        _record("明确不推荐", "clear_no", xhs_id="rejected"),
    ]
    state = _state(records)
    state["talents"] = [{"xhs_id": row["xhs_id"], "nickname": row["nickname"], **row["talent"]} for row in records]
    if problem == "inconsistent_input":
        state["talents"][2]["cpuv"] = 0

    def transport(*_args):
        return {"verdict": "reject", "issues": []} if problem == "invalid_answer" else {"verdict": "pass", "issues": []}

    state.update(_nodes(transport).audit_reasons(state))
    state["pricing_candidates"] = deepcopy(records)  # 即使旧名单错误地混入未通过者，也要重建。
    original = deepcopy(state)
    result = finalize_audit(state)
    assert state == original
    assert result["workflow_status"] == "audit_complete_with_pending"
    assert result["next_step"]["name"] == "build_portfolio"
    assert result["next_step"]["status"] == "interface_ready"
    assert {row["xhs_id"] for row in result["pricing_candidates"]} == {"healthy", "missing"}
    assert {row["xhs_id"] for row in result["audit_pending"]} == {"held"}
    assert {row["xhs_id"] for row in result["audited_talents"]} == {"healthy", "missing", "held", "rejected"}
    saved_held = next(row for row in result["audited_talents"] if row["xhs_id"] == "held")
    assert saved_held["release_status"] == "held_for_review"
    assert saved_held["audit_status"] != "passed"
    assert {name: saved_held[name] for name in held} == held
    assert result["supplement_results"][0]["xhs_id"] == "missing"
    assert set(result["next_step"]["required_state"]) >= {
        "audited_talents", "audit_pending", "pricing_candidates", "supplement_results",
    }


def test_held_problems_and_usable_candidates_are_exported_and_shown_together(tmp_path, monkeypatch, capsys):
    import main

    state = _state([_record("待复核", tag="价格贵", attempt=2), _record("正常", "clear_yes")])
    state.update(_nodes().audit_reasons(state))
    state.update(finalize_audit(state))
    monkeypatch.setattr(main, "BASE", tmp_path)
    saved = json.loads(save_result(state).read_text(encoding="utf-8"))
    assert saved["workflow_status"] == "audit_complete_with_pending"
    assert len(saved["audit_pending"]) == 1 and len(saved["audited_talents"]) == 2
    assert [row["nickname"] for row in saved["pricing_candidates"]] == ["正常"]
    assert saved["next_step"]["name"] == "build_portfolio"
    render_shortlist(state)
    printed = capsys.readouterr().out
    assert "暂存待复核，暂不参与投放组合" in printed
    assert "其余候选继续" in printed
    assert "下一步：投放组合，拟定价格" in printed


@pytest.mark.parametrize("change", ["stale_attempt", "unknown_person", "rewrite_not_done"])
def test_stale_or_unfinished_tasks_cannot_silently_be_marked_handled(change):
    state = _missing_state()
    task = state["audit_pending"][0]
    if change == "stale_attempt":
        task["attempt"] = 2
    elif change == "unknown_person":
        task["xhs_id"] = "unknown"
    else:
        task["action"] = "rewrite_reason"
    with pytest.raises(ValueError):
        finalize_audit(state)


def test_flagged_release_and_missing_fields_are_saved_and_visible_to_downstream_users(tmp_path, monkeypatch, capsys):
    import main

    state = _missing_state()
    result = {**state, **finalize_audit(state)}
    monkeypatch.setattr(main, "BASE", tmp_path)
    path = save_result(result)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["manual_review_queue"] == saved["audit_pending"] == []
    assert saved["supplement_results"][0]["data_incomplete"] is True
    assert saved["pricing_candidates"][0]["release_status"] == "flagged_release"
    assert saved["pricing_candidates"][0]["talent"]["cpuv"] is None
    render_shortlist(result)
    printed = capsys.readouterr().out
    assert "数据不全（进店成本）" in printed
    assert "保留空值" in printed
    assert "补充数据（进店成本）" not in printed
