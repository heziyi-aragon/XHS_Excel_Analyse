"""审核覆盖不能冒充人工调查完成，模型通过不能冒充推荐。"""
from copy import deepcopy
import json

from agent.audit_reporting import audit_report_fields
from agent.nodes.audit_handoff import finalize_audit
from main import render_shortlist, save_result
from tests.test_audit_nodes import _nodes, _record, _state


def test_manual_checks_survive_handoff_and_save_without_blocking_or_claiming_completion(tmp_path, monkeypatch, capsys):
    import main

    state = _state([_record("明确", "clear_yes"), _record("模型不推荐", stance="reject", tag="价格贵")])
    state["approved_policy"]["manual_checks"] = ["核实账号主体", "查看敏感肌内容", "核实账号主体"]
    state.update(_nodes().audit_reasons(state))
    original = deepcopy(state)
    result = finalize_audit(state)
    assert state == original
    assert result["audit_pending"] == [] and result["workflow_status"] == "audit_complete"
    assert result["audit_summary"]["model_reviewed_count"] == 1
    assert result["audit_summary"]["skipped_reason_review_count"] == 1
    assert result["audit_summary"]["rejected_count"] == 1
    assert result["audit_summary"]["usable_recommendations"] == 1
    assert result["audit_summary"]["unverified_check_count"] == 2
    assert all(item["status"] == "unverified" for item in result["audit_unverified_checks"])
    assert "audit_unverified_checks" in result["next_step"]["required_state"]
    merged = {**state, **result}
    monkeypatch.setattr(main, "BASE", tmp_path)
    saved = json.loads(save_result(merged).read_text(encoding="utf-8"))
    assert saved["audit_summary"] == result["audit_summary"]
    assert saved["audit_unverified_checks"] == result["audit_unverified_checks"]
    render_shortlist(merged)
    output = capsys.readouterr().out
    assert "评估倾向：不推荐" in output and "已处理不推荐 1 人" in output
    assert "未核实：核实账号主体" in output and "按流程跳过模型理由审校" in output


def test_code_rejection_is_not_counted_as_a_model_review_and_history_remains_visible(capsys):
    state = _state([_record("非法标签", tag="发明标签", attempt=2)])
    state.update(_nodes(lambda *_: (_ for _ in ()).throw(AssertionError("不能调用模型"))).audit_reasons(state))
    state.update(finalize_audit(state))
    assert state["audit_summary"]["model_reviewed_count"] == 0
    assert state["audit_summary"]["held_count"] == 1
    render_shortlist(state)
    assert "第 2 轮被打回" in capsys.readouterr().out


def test_old_saved_report_can_be_reviewed_read_only_without_using_an_api(tmp_path, monkeypatch, capsys):
    from scripts.audit_report import main
    from agent.requirements import LLMClient

    state = _state([_record("明确", "clear_yes")])
    state.update(_nodes().audit_reasons(state))
    state.update(finalize_audit(state))
    state.pop("audit_summary")
    state.pop("audit_unverified_checks")
    state["active_policy"] = {"criteria": []}
    path = tmp_path / "step8.json"
    path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    before = path.read_bytes()
    monkeypatch.setattr("sys.argv", ["audit_report", str(path)])
    monkeypatch.setattr(LLMClient, "from_env", lambda: (_ for _ in ()).throw(AssertionError("不能访问API")))
    assert main() == 0
    assert "模型理由审校按流程跳过" in capsys.readouterr().out
    assert path.read_bytes() == before
    assert audit_report_fields(state)["audit_summary"]["policy_changed_after_approval"]
