"""定价工具的来源、同行可比性和区间冲突验收，使用构造数据。"""
from copy import deepcopy

import pytest

from agent.portfolio_pricing import price_for_creator


def _candidate():
    talent = {"xhs_id": "self", "nickname": "小桃", "content_type": "Beauty", "tier": "T3",
              "price_image": 100, "price_video": 200}
    return {"xhs_id": "self", "nickname": "小桃", "talent": talent}, talent


def _observation(identity, price, **changes):
    row = {"id": identity, "creator_id": identity, "content_type": "Beauty", "tier": "T3",
           "format": "image", "price": price, "source": "离线构造的行情记录", "as_of": "2026-11-01"}
    row.update(changes)
    return row


def _price(samples=None, *, state=None, config=None, form="image", talent_changes=None):
    record, talent = _candidate()
    talent.update(talent_changes or {})
    cfg = {"pricing_as_of": "2026-11-04"}
    if samples is not None:
        cfg["industry_benchmarks"] = samples
    cfg.update(config or {})
    return price_for_creator(record, talent, form, cfg, state or {"talents": []})


def test_three_comparable_observations_materially_inform_the_platform_price_corridor():
    rows = [_observation("a", 90), _observation("b", 100), _observation("c", 110)]
    result = _price(rows)
    assert result["status"] == "industry_reference"
    assert result["benchmark"]["kind"] == "industry_reference"
    assert result["benchmark"]["sample_count"] == 3
    assert result["benchmark"]["adequate_samples"] is True
    assert result["benchmark"]["median"] == 100
    assert 80 <= result["low"] <= result["suggested"] <= result["high"] <= 120
    assert result["low"] > 80 and result["high"] < 120
    assert result["rationale"]
    assert {sample["source"] for sample in result["benchmark"]["samples"]} == {"离线构造的行情记录"}
    assert all(sample["as_of"] == "2026-11-01" for sample in result["benchmark"]["samples"])


def test_matching_uses_content_type_tier_and_format_and_excludes_the_creator_itself():
    observations = [
        _observation("a", 90), _observation("b", 100), _observation("c", 110),
        _observation("fashion", 999, content_type="Fashion"),
        _observation("tier2", 999, tier="T2"),
        _observation("video", 999, format="video"),
        _observation("self-record", 999, creator_id="self"),
    ]
    result = _price(observations)
    assert result["benchmark"]["sample_count"] == 3
    assert result["benchmark"]["median"] == 100
    ids = {sample["id"] for sample in result["benchmark"]["samples"]}
    assert ids == {"a", "b", "c"}


def test_multiple_prices_for_one_peer_do_not_count_as_independent_observations():
    observations = [_observation("one", 90, creator_id="same-peer"),
                    _observation("two", 100, creator_id="same-peer"),
                    _observation("three", 110, creator_id="same-peer")]
    result = _price(observations)
    assert result["status"] == "platform_fallback"
    assert result["benchmark"]["adequate_samples"] is False


def test_insufficient_real_benchmarks_fall_back_to_the_platform_without_inventing_a_market():
    result = _price([_observation("one", 90), _observation("two", 110)])
    assert result["status"] == "platform_fallback"
    assert result["reference"] == result["suggested"] == 100
    assert result["low"] == 80 and result["high"] == 120
    assert result["benchmark"]["kind"] == "not_provided"
    assert result["benchmark"]["adequate_samples"] is False


def test_internal_dataset_peers_are_explicitly_simulated_and_not_real_market_evidence():
    peers = [{"xhs_id": key, "content_type": "Beauty", "tier": "T3", "price_image": price}
             for key, price in (("a", 90), ("b", 100), ("c", 110))]
    result = _price(state={"talents": peers})
    assert result["status"] == "sample_peer_reference"
    assert result["benchmark"]["kind"] == "sample_peer_reference"
    assert result["benchmark"]["real_market_data"] is False
    assert result["benchmark"]["adequate_samples"] is True
    assert all(sample["synthetic"] is True for sample in result["benchmark"]["samples"])


def test_explicit_synthetic_external_samples_cannot_be_reported_as_real_market_evidence():
    result = _price([_observation(key, price, synthetic=True)
                     for key, price in (("a", 90), ("b", 100), ("c", 110))])
    assert result["benchmark"]["kind"] == "industry_fixture"
    assert result["benchmark"]["real_market_data"] is False


def test_an_extreme_observation_is_flagged_and_does_not_dominate_the_reference():
    rows = [_observation(key, price) for key, price in
            (("a", 90), ("b", 95), ("c", 100), ("d", 105), ("extreme", 10000))]
    result = _price(rows)
    assert "extreme" in result["benchmark"]["outlier_ids"]
    assert result["benchmark"]["median"] < 110
    assert result["high"] <= 120


def test_future_benchmarks_are_excluded_only_against_the_supplied_date_anchor():
    rows = [_observation("a", 90), _observation("b", 100), _observation("c", 110),
            _observation("future", 1000, as_of="2026-12-01")]
    result = _price(rows)
    assert result["benchmark"]["excluded_future_ids"] == ["future"]
    assert result["benchmark"]["sample_count"] == 3
    record, talent = _candidate()
    no_anchor = price_for_creator(record, talent, "image", {"industry_benchmarks": rows}, {"talents": []})
    assert no_anchor["benchmark"]["as_of_status"] == "as_of_not_provided"
    assert no_anchor["benchmark"]["excluded_future_ids"] == []


def test_a_disjoint_peer_range_is_reported_as_a_conflict_and_preserves_platform_limits():
    result = _price([_observation("a", 300), _observation("b", 310), _observation("c", 320)])
    assert result["conflict"] is True
    assert result["low"] == 80 and result["high"] == 120
    assert result["suggested"] == 120
    assert result["rationale"]


def test_missing_platform_anchor_is_not_replaced_by_an_unverified_platform_price():
    result = _price([_observation("a", 90), _observation("b", 100), _observation("c", 110)],
                    talent_changes={"price_image": None})
    assert result["status"] == "unavailable"
    assert result["reference"] is None
    assert result["low"] is result["suggested"] is result["high"] is None


def test_an_explicit_multiplier_is_honored_and_keeps_all_proposed_prices_in_the_budget_range():
    result = _price([_observation("a", 100), _observation("b", 105), _observation("c", 110)],
                    config={"price_multiplier": 0.9})
    assert result["suggested"] == 90
    assert 80 <= result["low"] <= 90 <= result["high"] <= 120


def test_outside_corridor_pricing_requires_a_reason_and_expands_the_reserved_high_cost():
    with pytest.raises(ValueError):
        _price(config={"price_multiplier": 1.3})
    result = _price(config={"price_multiplier": 1.3, "exception_reason": "构造测试：另含授权费"})
    assert result["suggested"] == 130
    assert result["high"] >= 130
    assert result["exception"]["outside_platform_corridor"] is True
    assert result["exception"]["requires_approval"] is True
    assert result["exception"]["reason"] == "构造测试：另含授权费"


@pytest.mark.parametrize("field,value", [
    ("source", ""), ("as_of", "2026-99-01"), ("format", "livestream"),
    ("content_type", ""), ("tier", ""), ("price", True),
    ("price", -1), ("price", 0), ("price", float("nan")), ("price", float("inf")),
])
def test_invalid_external_observations_are_rejected_instead_of_silently_becoming_prices(field, value):
    sample = _observation("bad", 100)
    sample[field] = value
    with pytest.raises(ValueError):
        _price([sample])


def test_price_tool_never_mutates_the_platform_data_or_sources():
    record, talent = _candidate()
    cfg = {"industry_benchmarks": [_observation("a", 90), _observation("b", 100), _observation("c", 110)]}
    state = {"talents": []}
    original = deepcopy((record, talent, cfg, state))
    result = price_for_creator(record, talent, "image", cfg, state)
    assert (record, talent, cfg, state) == original
    result["benchmark"]["samples"][0]["source"] = "changed"
    assert (record, talent, cfg, state) == original
