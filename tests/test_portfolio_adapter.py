"""新节点、独立运行器及重定价CLI的离线验收。"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from agent.nodes.portfolio import build_portfolio, build_portfolio_graph, make_portfolio_node
from scripts.build_portfolio import main, read_json, report_markdown, run


def _handoff():
    row = {
        "xhs_id": "test-kol", "nickname": "离线达人", "stance": "recommend",
        "score": 0.9, "band": "clear_yes", "attempt": 2, "data_revision": 1,
        "tag": "数据好", "sentence": "历史互动可供参考。",
        "audit_status": "skipped_clear_band", "audit_issues": [],
        "talent": {"xhs_id": "test-kol", "nickname": "离线达人",
                   "price_image": 100, "price_video": None,
                   "exposure_med": 1000, "read_med": 500, "interact_med": 100,
                   "main_format": "图文为主"},
    }
    return {
        "brief": {"total_budget": 1000, "notes": "平台参考价上下浮动20%以内",
                  "kpi": "曝光≥1000；互动≥100；整体CPE≤2", "content_direction": "图文为主"},
        "requirement_profile": {}, "active_policy": {"criteria": []},
        "pricing_candidates": [row], "audited_talents": [deepcopy(row)],
        "audit_pending": [], "supplement_results": [], "audit_unverified_checks": [],
        "audit_summary": {"entered_count": 1}, "workflow_status": "audit_complete",
        "next_step": {"name": "build_portfolio", "implemented": False},
    }


def _write_source(tmp_path, state=None):
    source = tmp_path / "step8.json"
    source.write_text(json.dumps(state if state is not None else _handoff(), ensure_ascii=False), encoding="utf-8")
    return source


def _run_main(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["build_portfolio", *map(str, args)])
    return main()


@pytest.mark.parametrize("field", [
    "brief", "requirement_profile", "active_policy", "pricing_candidates", "audited_talents",
    "audit_pending", "supplement_results", "audit_unverified_checks", "audit_summary",
])
def test_node_rejects_an_incomplete_handoff_without_altering_it(field):
    state = _handoff()
    del state[field]
    original = deepcopy(state)
    with pytest.raises(ValueError, match=field):
        build_portfolio(state)
    assert state == original


@pytest.mark.parametrize("field,invalid", [
    ("brief", []), ("requirement_profile", "text"), ("active_policy", None),
    ("pricing_candidates", {}), ("audited_talents", {}), ("audit_pending", {}),
    ("supplement_results", "text"), ("audit_unverified_checks", {}), ("audit_summary", []),
])
def test_node_rejects_wrong_handoff_collection_types(field, invalid):
    state = _handoff()
    state[field] = invalid
    with pytest.raises(ValueError, match=field):
        make_portfolio_node()(state)


def test_node_only_returns_a_new_plan_and_preserves_all_upstream_fields():
    state = _handoff()
    state["portfolio_config"] = {"price_multiplier": 0.9}
    state["portfolio_feedback"] = "测试九折"
    original = deepcopy(state)
    result = build_portfolio(state)
    assert set(result) == {"portfolio_plan"}
    assert result["portfolio_plan"]["pricing_policy"]["price_multiplier"] == 0.9
    assert result["portfolio_plan"]["feedback"] == "测试九折"
    assert state == original
    assert state["next_step"]["implemented"] is False
    assert state["workflow_status"] == "audit_complete"


def test_run_writes_a_new_envelope_and_keeps_upstream_payload_and_file_exactly(tmp_path):
    source = _write_source(tmp_path)
    original_bytes = source.read_bytes()
    expected = read_json(source)
    output = tmp_path / "new" / "step9.json"
    report = tmp_path / "new" / "step9.md"
    result = run(source, output, config={"price_multiplier": 0.9},
                 feedback={"comments": "第一次建议价"}, report_path=report)
    assert source.read_bytes() == original_bytes
    assert result["upstream_state"] == expected
    assert read_json(output) == result
    assert result["portfolio_plan"]["revision"] == 1
    assert result["portfolio_config"]["price_multiplier"] == 0.9
    assert report.read_text(encoding="utf-8") == report_markdown(result["portfolio_plan"])


def test_run_rejects_using_the_input_as_output(tmp_path):
    source = _write_source(tmp_path)
    original = source.read_bytes()
    with pytest.raises(ValueError):
        run(source, source)
    assert source.read_bytes() == original


@pytest.mark.parametrize("existing", ["output", "report"])
def test_run_rejects_existing_output_or_report_without_overwriting_either(tmp_path, existing):
    source = _write_source(tmp_path)
    output, report = tmp_path / "step9.json", tmp_path / "step9.md"
    protected = output if existing == "output" else report
    protected.write_text("existing evidence", encoding="utf-8")
    with pytest.raises(ValueError):
        run(source, output, report_path=report)
    assert protected.read_text(encoding="utf-8") == "existing evidence"
    assert not (report if existing == "output" else output).exists()


def test_run_rejects_a_report_target_that_is_the_source_input(tmp_path):
    source = _write_source(tmp_path)
    original = source.read_bytes()
    output = tmp_path / "step9.json"
    with pytest.raises(ValueError):
        run(source, output, report_path=source)
    assert source.read_bytes() == original
    assert not output.exists()


def test_run_rejects_colliding_new_json_and_markdown_paths(tmp_path):
    source = _write_source(tmp_path)
    collision = tmp_path / "new_result"
    with pytest.raises(ValueError):
        run(source, collision, report_path=collision)
    assert not collision.exists()


def test_cli_previous_increments_only_portfolio_revision_and_preserves_inputs(tmp_path, monkeypatch):
    source = _write_source(tmp_path)
    first_path, second_path = tmp_path / "v1.json", tmp_path / "v2.json"
    first = run(source, first_path)
    source_bytes, previous_bytes = source.read_bytes(), first_path.read_bytes()
    assert _run_main(monkeypatch, source, "--previous", first_path, "--output", second_path) == 0
    second = read_json(second_path)
    assert second["portfolio_plan"]["revision"] == first["portfolio_plan"]["revision"] + 1
    assert second["upstream_state"] == read_json(source)
    assert second["upstream_state"]["pricing_candidates"][0]["attempt"] == 2
    assert second["upstream_state"]["pricing_candidates"][0]["data_revision"] == 1
    assert source.read_bytes() == source_bytes
    assert first_path.read_bytes() == previous_bytes


def test_cli_previous_inherits_prior_configuration_when_no_new_config_is_supplied(tmp_path, monkeypatch):
    source = _write_source(tmp_path)
    previous_path, output = tmp_path / "v1.json", tmp_path / "v2.json"
    config = {"price_multiplier": 0.9, "max_posts_per_creator": 2, "max_search_nodes": 50}
    run(source, previous_path, config=config)
    assert _run_main(monkeypatch, source, "--previous", previous_path, "--output", output) == 0
    result = read_json(output)
    assert result["portfolio_config"] == config
    assert result["portfolio_plan"]["pricing_policy"]["price_multiplier"] == 0.9
    assert result["portfolio_plan"]["post_caps"]["default"] == 2
    assert result["portfolio_plan"]["max_search_nodes"] == 50


def test_cli_feedback_overrides_only_the_requested_setting_and_keeps_prior_config(tmp_path, monkeypatch):
    source = _write_source(tmp_path)
    previous_path, output = tmp_path / "v1.json", tmp_path / "v2.json"
    config = {"price_multiplier": 0.9, "max_posts_per_creator": 2, "max_search_nodes": 50}
    run(source, previous_path, config=config)
    feedback = tmp_path / "feedback.json"
    feedback.write_text(json.dumps({"config_overrides": {"price_multiplier": 0.85},
                                    "comments": "仅修改为85折"}, ensure_ascii=False), encoding="utf-8")
    assert _run_main(monkeypatch, source, "--previous", previous_path, "--feedback", feedback,
                     "--output", output) == 0
    result = read_json(output)
    assert result["portfolio_plan"]["revision"] == 2
    assert result["portfolio_plan"]["pricing_policy"]["price_multiplier"] == 0.85
    assert result["portfolio_plan"]["post_caps"]["default"] == 2
    assert result["portfolio_plan"]["max_search_nodes"] == 50
    assert result["portfolio_config"]["price_multiplier"] == 0.85
    assert result["portfolio_plan"]["feedback"]["comments"] == "仅修改为85折"


def test_cli_previous_rejects_colliding_output_and_report_paths(tmp_path, monkeypatch):
    source = _write_source(tmp_path)
    previous = tmp_path / "v1.json"
    run(source, previous)
    output = tmp_path / "collision"
    with pytest.raises(SystemExit) as caught:
        _run_main(monkeypatch, source, "--previous", previous, "--output", output, "--report", output)
    assert caught.value.code == 2
    assert not output.exists()


def test_report_uses_plan_costs_kpi_results_and_preserved_risk_counts(tmp_path):
    state = _handoff()
    state["pricing_candidates"][0]["nickname"] = "达人|甲"
    state["audit_pending"] = [{"xhs_id": "held", "action": "retry_audit"}]
    state["audit_unverified_checks"] = [{"description": "核实合作次数"}, {"description": "核实排期"}]
    unknown = deepcopy(state["pricing_candidates"][0])
    unknown["xhs_id"], unknown["nickname"] = "unpriced", "待核价达人"
    unknown["talent"].update(xhs_id="unpriced", price_image=None, price_video=None)
    state["pricing_candidates"].append(unknown)
    source = _write_source(tmp_path, state)
    result = run(source, tmp_path / "out.json")
    plan = result["portfolio_plan"]
    text = report_markdown(plan)
    assert f"建议总价：¥{plan['budget']['total_suggested']:,.2f}" in text
    assert f"区间上限总价：¥{plan['budget']['total_upper']:,.2f}" in text
    assert "达人/甲" in text
    assert "待核价达人" in text
    assert "暂存问题 1 项；待核实检查 2 项" in text
    assert f"版本：{plan['revision']}" in text
    assert plan["solver_status"] in text
    assert "历史中位数用于估算，不代表实际投放承诺" in text
    assert "整体CPE（上限费用口径）" in text
    for check in plan["kpi_checks"].values():
        assert check["status"] in text


def test_json_reader_accepts_utf8_bom_but_rejects_a_nonobject_payload(tmp_path):
    source = tmp_path / "bom.json"
    source.write_text(json.dumps(_handoff(), ensure_ascii=False), encoding="utf-8-sig")
    assert read_json(source) == _handoff()
    source.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        read_json(source)


@pytest.mark.skipif(importlib.util.find_spec("langgraph") is None, reason="LangGraph not installed in this interpreter")
def test_optional_subgraph_retains_every_known_upstream_field_and_adds_only_plan():
    state = _handoff()
    original = deepcopy(state)
    result = build_portfolio_graph().invoke(state)
    for key, value in original.items():
        assert result[key] == value
    assert result["portfolio_plan"]["revision"] == 1
    assert state == original
