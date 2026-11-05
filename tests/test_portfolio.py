"""投放组合的独立离线验收，不启动图、不调用真实模型、不改上游记录。"""
from copy import deepcopy
import math
from pathlib import Path

import pytest

from agent.portfolio import build_portfolio_plan


def _record(identity="kol-1", *, price_image=100, price_video=None,
            exposure=1000, interactions=100, score=0.9, **changes):
    row = {
        "xhs_id": identity,
        "nickname": f"达人{identity}",
        "stance": "recommend",
        "band": "clear_yes",
        "score": score,
        "tag": "数据好",
        "sentence": "历史互动数据可供方案比较。",
        "attempt": 2,
        "data_revision": 3,
        "audit_status": "skipped_clear_band",
        "audit_issues": [],
        "talent": {
            "xhs_id": identity,
            "nickname": f"达人{identity}",
            "price_image": price_image,
            "price_video": price_video,
            "exposure_med": exposure,
            "read_med": 500,
            "interact_med": interactions,
            "cpe": 999,
            "main_format": "图文为主",
            "content_type": "Beauty",
            "tier": "T3",
        },
    }
    row.update(changes)
    return row


def _state(rows=None, *, budget=10000):
    records = deepcopy(rows if rows is not None else [_record()])
    return {
        "brief": {
            "total_budget": budget,
            "notes": "达人报价以平台参考价为基准上下浮动20%以内，超出需单独说明理由",
            "kpi": "",
            "content_direction": "图文种草为主、视频为辅",
        },
        "requirement_profile": {"consultant_assumptions": [], "downstream_dependencies": []},
        "active_policy": {"criteria": [], "manual_checks": []},
        "pricing_candidates": records,
        "audited_talents": deepcopy(records),
        "audit_pending": [],
        "supplement_results": [],
        "audit_unverified_checks": [],
        "audit_summary": {"pricing_candidates_count": len(records)},
        "workflow_status": "audit_complete",
    }


def _targets(**values):
    return {"targets": {name: {"value": value, "source": "离线验收构造需求"}
                        for name, value in values.items()}}


def _total(plan, key):
    return sum(line[key] for line in plan["lines"])


def test_nested_platform_price_drives_default_range_and_exact_cost_totals():
    state = _state([_record(price_image=100, price_video=200)])
    # 同名顶层字段不能覆盖原始档案内的报价。
    state["pricing_candidates"][0]["price_image"] = 1
    plan = build_portfolio_plan(state)
    assert plan["schema_version"] == "1.0"
    assert len(plan["lines"]) == 1
    line = plan["lines"][0]
    reference = state["pricing_candidates"][0]["talent"][
        "price_image" if line["format"] == "image" else "price_video"]
    assert line["platform_reference_price"] == reference
    assert line["unit_price_low"] == pytest.approx(reference * 0.8)
    assert line["unit_price_suggested"] == reference
    assert line["unit_price_high"] == pytest.approx(reference * 1.2)
    assert line["quantity"] == 1
    for suffix in ("low", "suggested", "high"):
        assert line[f"line_cost_{suffix}"] == pytest.approx(
            line[f"unit_price_{suffix}"] * line["quantity"])
    assert plan["budget"]["total_low"] == pytest.approx(_total(plan, "line_cost_low"))
    assert plan["budget"]["total_suggested"] == pytest.approx(_total(plan, "line_cost_suggested"))
    assert plan["budget"]["total_upper"] == pytest.approx(_total(plan, "line_cost_high"))


def test_selection_reserves_upper_bound_instead_of_only_suggested_price():
    state = _state([_record("expensive", price_image=100, score=1),
                    _record("affordable", price_image=50, score=0.8)], budget=100)
    plan = build_portfolio_plan(state)
    assert plan["budget"]["basis"] == "upper_bound"
    assert plan["budget"]["total_upper"] <= 100
    assert plan["budget"]["remaining_upper"] == pytest.approx(100 - plan["budget"]["total_upper"])
    assert {line["xhs_id"] for line in plan["lines"]} == {"affordable"}
    assert plan["budget"]["within_budget"] is True


def test_price_at_budget_is_not_feasible_if_range_upper_exceeds_budget():
    plan = build_portfolio_plan(_state([_record(price_image=100)], budget=100))
    assert plan["lines"] == []
    assert plan["status"] != "ready"


def test_currency_rounding_keeps_an_exact_budget_boundary_feasible():
    plan = build_portfolio_plan(
        _state([_record(price_image=0.1)], budget=0.36),
        {**_targets(exposure=3000), "max_posts_per_creator": 3},
    )
    assert sum(line["quantity"] for line in plan["lines"]) == 3
    assert plan["budget"]["total_upper"] == 0.36
    assert plan["budget"]["remaining_upper"] == 0
    assert plan["budget"]["within_budget"] is True


@pytest.mark.parametrize("price", [None, 0, -100, True, float("nan"), float("inf")])
def test_unusable_prices_never_create_a_free_or_nonfinite_placement(price):
    plan = build_portfolio_plan(_state([_record(price_image=price, price_video=None)]))
    assert plan["lines"] == []
    assert plan["status"] == "needs_pricing"
    assert plan["conditional_candidates"]
    assert math.isfinite(plan["budget"]["total_upper"])


def test_missing_video_does_not_block_a_known_image_price():
    plan = build_portfolio_plan(_state([_record(price_image=100, price_video=None)]))
    assert len(plan["lines"]) == 1
    assert plan["lines"][0]["format"] == "image"
    assert plan["lines"][0]["unit_price_suggested"] == 100


def test_flagged_release_keeps_upstream_missing_data_and_audit_markers():
    row = _record(data_incomplete=True, missing_data_fields=["cpuv"],
                  release_status="flagged_release", audit_status="needs_resolution",
                  audit_issues=[{"code": "data_missing", "message": "缺少进店成本", "fields": ["cpuv"]}])
    state = _state([row])
    state["supplement_results"] = [{"xhs_id": row["xhs_id"], "status": "failed", "fields": ["cpuv"]}]
    plan = build_portfolio_plan(state)
    assert len(plan["lines"]) == 1
    flags = plan["lines"][0]["risk_flags"]
    for key in ("data_incomplete", "missing_data_fields", "release_status", "audit_status", "audit_issues"):
        assert flags[key] == row[key]
    assert plan["upstream_risks"]["supplement_results"] == state["supplement_results"]


def test_held_and_rejected_records_cannot_reenter_a_portfolio():
    state = _state([_record("valid"),
                    _record("held", release_status="held_for_review"),
                    _record("rejected", stance="reject", band="clear_no")])
    plan = build_portfolio_plan(state)
    assert {line["xhs_id"] for line in plan["lines"]} == {"valid"}


def test_input_and_all_upstream_exception_lists_are_left_unchanged():
    state = _state()
    state["audit_pending"] = [{"xhs_id": "held", "action": "retry_audit", "issues": [{"code": "execution_error"}]}]
    state["audit_unverified_checks"] = [{"description": "合作次数需人工核实", "status": "unverified"}]
    original = deepcopy(state)
    plan = build_portfolio_plan(state)
    assert state == original
    assert plan["upstream_risks"]["audit_pending"] == original["audit_pending"]
    assert plan["upstream_risks"]["audit_unverified_checks"] == original["audit_unverified_checks"]
    # 返回内容也不应与上游嵌套列表共用可变对象。
    plan["upstream_risks"]["audit_pending"][0]["issues"].append({"code": "new"})
    assert state == original


def test_no_candidates_is_reported_without_recreating_rejected_creators():
    state = _state([])
    state["audited_talents"] = [_record("rejected", stance="reject")]
    plan = build_portfolio_plan(state)
    assert plan["status"] == "no_candidates"
    assert plan["solver_status"] == "not_run"
    assert plan["lines"] == []


def test_duplicate_identity_is_an_error_instead_of_double_booking():
    with pytest.raises(ValueError):
        build_portfolio_plan(_state([_record("same"), _record("same")]))


def test_an_entirely_blank_identity_is_an_error():
    row = _record("")
    row["nickname"] = ""
    row["talent"]["nickname"] = ""
    with pytest.raises(ValueError):
        build_portfolio_plan(_state([row]))


@pytest.mark.parametrize("budget", [None, True, -1, float("nan"), float("inf")])
def test_invalid_hard_budget_never_becomes_an_unbounded_plan(budget):
    with pytest.raises(ValueError):
        build_portfolio_plan(_state(budget=budget))


def test_missing_effect_data_is_not_zero_or_a_successful_kpi():
    state = _state([_record(exposure=None, interactions=None)])
    plan = build_portfolio_plan(state, _targets(exposure=1, interactions=1, cpe=25, brand_search_lift=0.3))
    assert plan["lines"]
    assert plan["lines"][0]["effect_proxy"]["exposure"] is None
    assert plan["lines"][0]["effect_proxy"]["interactions"] is None
    for name in ("exposure", "interactions", "cpe", "brand_search_lift"):
        assert plan["kpi_checks"][name]["status"] == "not_evaluable"
        assert plan["kpi_checks"][name]["estimate"] is None


def test_partial_effect_coverage_does_not_silently_claim_a_complete_kpi_total():
    state = _state([_record("known", exposure=100, interactions=10),
                    _record("unknownE", exposure=None, interactions=100),
                    _record("unknownI", exposure=1000, interactions=None)])
    plan = build_portfolio_plan(state, _targets(exposure=1000, interactions=100, cpe=25))
    assert {line["xhs_id"] for line in plan["lines"]} == {"known", "unknownE", "unknownI"}
    for name in ("exposure", "interactions", "cpe"):
        assert plan["kpi_checks"][name]["status"] == "not_evaluable"
        assert plan["kpi_checks"][name]["estimate"] is None


def test_zero_interactions_does_not_produce_a_zero_or_successful_cpe():
    plan = build_portfolio_plan(_state([_record(interactions=0)]), _targets(cpe=25))
    assert plan["lines"]
    assert plan["kpi_checks"]["cpe"]["status"] == "not_evaluable"
    assert plan["kpi_checks"]["cpe"]["estimate"] is None


def test_overall_cpe_uses_aggregate_plan_cost_divided_by_matching_interactions():
    state = _state([_record("small", price_image=100, interactions=10),
                    _record("large", price_image=900, interactions=990)])
    state["pricing_candidates"][0]["talent"]["cpe"] = 10
    state["pricing_candidates"][1]["talent"]["cpe"] = 900 / 990
    plan = build_portfolio_plan(state, _targets(cpe=25))
    assert len(plan["lines"]) == 2
    interactions = sum(line["effect_proxy"]["interactions"] for line in plan["lines"])
    estimate = plan["kpi_checks"]["cpe"]["estimate"]
    # 保守预算口径应与计算CPE的成本口径一致。
    assert estimate == pytest.approx(plan["budget"]["total_upper"] / interactions)
    assert estimate != pytest.approx((10 + 900 / 990) / 2)


def test_an_expensive_inefficient_addition_cannot_displace_a_fully_kpi_feasible_plan():
    state = _state([_record("efficient", price_image=100, exposure=1000, interactions=100),
                    _record("inefficient", price_image=1000, exposure=1000, interactions=1)])
    plan = build_portfolio_plan(state, _targets(exposure=1000, interactions=100, cpe=2))
    assert {line["xhs_id"] for line in plan["lines"]} == {"efficient"}
    assert plan["kpi_checks"]["cpe"]["estimate"] == pytest.approx(1.2)
    for name in ("exposure", "interactions", "cpe"):
        assert plan["kpi_checks"][name]["status"] == "proxy_met"
    assert plan["status"] == "ready"


def test_configurable_post_cap_applies_to_image_and_video_combined():
    state = _state([_record(price_image=100, price_video=150)], budget=10000)
    plan = build_portfolio_plan(state, {
        **_targets(exposure=10000, interactions=1000),
        "max_posts_per_creator": 3,
    })
    assert sum(line["quantity"] for line in plan["lines"]) == 3
    assert all(isinstance(line["quantity"], int) and line["quantity"] > 0 for line in plan["lines"])


def test_individual_post_cap_can_tighten_the_shared_maximum():
    state = _state([_record("one"), _record("three")])
    plan = build_portfolio_plan(state, {
        **_targets(exposure=10000),
        "max_posts_per_creator": 3,
        "per_creator_max_posts": {"one": 1},
    })
    quantities = {identity: sum(line["quantity"] for line in plan["lines"] if line["xhs_id"] == identity)
                  for identity in ("one", "three")}
    assert quantities == {"one": 1, "three": 3}


@pytest.mark.parametrize("key_kind", ["xhs_id", "creator_key", "nickname"])
def test_requested_video_uses_video_reference_without_mutating_the_image_price(key_kind):
    state = _state([_record(price_image=100, price_video=250)])
    original = deepcopy(state)
    base = build_portfolio_plan(state)
    key = (base["lines"][0]["creator_key"] if key_kind == "creator_key"
           else state["pricing_candidates"][0][key_kind])
    plan = build_portfolio_plan(state, {"per_creator_allowed_formats": {key: ["video"]}})
    assert len(plan["lines"]) == 1
    line = plan["lines"][0]
    assert line["format"] == "video"
    assert line["platform_reference_price"] == line["unit_price_suggested"] == 250
    assert line["unit_price_low"] == 200
    assert line["unit_price_high"] == 300
    assert state == original
    assert state["pricing_candidates"][0]["talent"]["price_image"] == 100


def test_requested_unpriced_video_is_conditional_instead_of_falling_back_to_image():
    state = _state([_record(price_image=100, price_video=None)])
    original = deepcopy(state)
    plan = build_portfolio_plan(state, {"per_creator_allowed_formats": {"kol-1": ["video"]}})
    assert plan["lines"] == []
    assert plan["status"] == "needs_pricing"
    assert plan["budget"]["total_upper"] == 0
    assert len(plan["conditional_candidates"]) == 1
    conditional = plan["conditional_candidates"][0]
    assert conditional["reason"] == "not_available_for_requested_format"
    assert conditional["requested_formats"] == ["video"]
    assert conditional["budget_included"] is False
    assert state == original


def test_an_empty_format_map_keeps_the_default_allocation_and_prices():
    state = _state([_record("both", price_image=100, price_video=200),
                    _record("video-only", price_image=None, price_video=150)])
    default = build_portfolio_plan(state)
    explicit_empty = build_portfolio_plan(state, {"per_creator_allowed_formats": {}})
    assert explicit_empty["lines"] == default["lines"]
    assert explicit_empty["budget"] == default["budget"]
    assert explicit_empty["kpi_checks"] == default["kpi_checks"]


@pytest.mark.parametrize("invalid", [[], ["image", "image"], ["livestream"], ["image", "unknown"], "video", None])
def test_invalid_per_creator_format_values_are_rejected(invalid):
    with pytest.raises(ValueError):
        build_portfolio_plan(_state(), {"per_creator_allowed_formats": {"kol-1": invalid}})


def test_feedback_can_change_only_the_requested_creator_to_video():
    state = _state([_record("change", price_image=100, price_video=200),
                    _record("unchanged", price_image=50, price_video=300)])
    config = {"per_creator_allowed_formats": {"change": ["image"]}}
    first = build_portfolio_plan(state, config)
    state["portfolio_plan"] = first
    original = deepcopy(state)
    revised = build_portfolio_plan(state, config, feedback={
        "config_overrides": {"per_creator_allowed_formats": {"change": ["video"]}},
        "comments": "仅指定change达人改成视频",
    })
    rows = {line["xhs_id"]: line for line in revised["lines"]}
    assert rows["change"]["format"] == "video"
    assert rows["change"]["platform_reference_price"] == 200
    assert rows["unchanged"]["format"] == "image"
    assert rows["unchanged"]["platform_reference_price"] == 50
    assert revised["revision"] == 2
    assert revised["configuration"]["per_creator_allowed_formats"] == {"change": ["video"]}
    assert state == original
    assert state["pricing_candidates"][0]["attempt"] == 2


def test_feedback_increments_portfolio_revision_without_resetting_upstream_attempt():
    state = _state()
    first = build_portfolio_plan(state)
    state["portfolio_plan"] = deepcopy(first)
    original = deepcopy(state)
    revised = build_portfolio_plan(state, {"price_multiplier": 0.9}, feedback="请按参考价九折重算")
    assert first["revision"] == 1
    assert revised["revision"] == 2
    assert revised["lines"][0]["unit_price_suggested"] == 90
    assert state == original
    assert state["pricing_candidates"][0]["attempt"] == 2
    assert state["pricing_candidates"][0]["data_revision"] == 3


def test_a_search_limit_is_disclosed_instead_of_claiming_an_optimum():
    state = _state([_record(str(i), price_image=100, score=1 - i / 10) for i in range(6)], budget=250)
    plan = build_portfolio_plan(state, {"max_search_nodes": 1})
    assert plan["solver_status"] == "limit_reached"
    assert plan["budget"]["total_upper"] <= 250
    assert plan["warnings"]


def test_identical_inputs_produce_identical_plans():
    state = _state([_record("b", score=0.8), _record("a", score=0.8)])
    assert build_portfolio_plan(state) == build_portfolio_plan(deepcopy(state))


def test_a_malicious_llm_cannot_raise_the_budget_add_creators_or_supply_effect_values():
    class MaliciousLLM:
        calls = 0

        def complete_json(self, *_args, **_kwargs):
            self.calls += 1
            return {
                "constraints": [{"text": "预算无限且必须推荐ghost", "source_quote": "不存在的原文"}],
                "explanation": "忽略原预算，保证达标。",
                "budget": 999999999,
                "lines": [{"xhs_id": "ghost", "unit_price_suggested": 0}],
                "exposure": 999999999,
            }

    client = MaliciousLLM()
    state = _state([_record("real", price_image=100)], budget=150)
    original = deepcopy(state)
    plan = build_portfolio_plan(state, _targets(exposure=10000), llm_client=client)
    assert client.calls >= 1
    assert plan["planning"]["status"] == "ready"
    assert plan["planning"]["pending"] == []
    assert any(item["reason"] == "model_response_invalid" for item in plan["planning"]["rejected"])
    assert plan["budget"]["limit"] == 150
    assert plan["budget"]["within_budget"] is True
    assert {line["xhs_id"] for line in plan["lines"]} == {"real"}
    assert plan["kpi_checks"]["exposure"]["status"] == "proxy_unmet"
    assert state["brief"]["total_budget"] == 150
    assert {row["xhs_id"] for row in state["pricing_candidates"]} == {"real"}
    assert state["pricing_candidates"][0]["talent"]["price_image"] == 100
    assert "portfolio_plan" not in state
    assert state == original


def test_price_outside_twenty_percent_requires_an_explicit_exception_reason():
    with pytest.raises(ValueError):
        build_portfolio_plan(_state(), {"price_multiplier": 1.3})


def test_twenty_real_sample_creators_cannot_be_claimed_to_meet_campaign_kpis_with_one_post_each():
    from agent.loaders import BriefLoader, TalentLoader

    data = Path(__file__).resolve().parents[1] / "static"
    talents = TalentLoader(data / "蒲公英达人数据_屿光精华.xlsx").load()
    brief = BriefLoader(data / "甲方投放需求单_屿光精华.xlsx").load()
    assert len(talents) == 20
    assert sum(row.exposure_med for row in talents) == 2656571
    assert sum(row.interact_med for row in talents) == 21834
    records = []
    for talent in talents:
        record = _record(talent.xhs_id)
        record["nickname"] = talent.nickname
        record["talent"] = talent.to_dict()
        records.append(record)
    state = _state(records, budget=brief.total_budget)
    state["brief"] = brief.to_dict()
    plan = build_portfolio_plan(state, {
        **_targets(exposure=25000000, interactions=250000, cpe=25, brand_search_lift=0.3),
        "max_search_nodes": 1000,
    })
    assert plan["budget"]["total_upper"] <= 350000
    for name, ceiling, target in (("exposure", 2656571, 25000000), ("interactions", 21834, 250000)):
        check = plan["kpi_checks"][name]
        assert check["estimate"] <= ceiling
        assert check["status"] == "proxy_unmet"
        assert check["gap"] == pytest.approx(target - check["estimate"])
    assert plan["kpi_checks"]["brand_search_lift"]["status"] == "not_evaluable"
