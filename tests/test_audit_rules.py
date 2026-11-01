"""审核代码规则：重点验证前序失败、两侧标签及缺数边界，不调用模型。"""
from copy import deepcopy

import pytest

from agent.audit_rules import check_reason_rules, check_required_data


def _record(**updates):
    return {
        "nickname": "测试达人", "attempt": 1, "band": "mid",
        "stance": "recommend", "tag": "进店好",
        "sentence": "进店量较高，但单次互动成本偏高。", "error": None,
        "talent": {"visit_store_med": 800, "cpuv": 10, "price_video": None, "mcn": None},
        "missing_fields": ["price_video"],
        **updates,
    }


def _policy(*fields):
    return {"criteria": [{"field": field, "operator": "maximize"} for field in fields]}


def test_reason_error_is_not_misreported_as_an_invalid_tag():
    record = _record(error="生成失败", stance=None, tag=None, sentence=None)
    issues = check_reason_rules(record)
    assert [issue["code"] for issue in issues] == ["generation_error"]


@pytest.mark.parametrize("updates", [{"stance": None}, {"sentence": None}, {"sentence": "  "}])
def test_incomplete_reason_cannot_be_sent_to_semantic_audit(updates):
    assert check_reason_rules(_record(**updates))[0]["code"] == "generation_error"


@pytest.mark.parametrize("stance,tag", [("recommend", "价格贵"), ("reject", "数据好"), ("recommend", "超级推荐")])
def test_a_known_tag_on_the_wrong_side_also_needs_rewriting(stance, tag):
    issues = check_reason_rules(_record(stance=stance, tag=tag))
    assert issues[0]["code"] == "invalid_tag"


@pytest.mark.parametrize("stance", ["recommend", "reject"])
def test_shared_other_tag_is_legal_on_both_sides(stance):
    assert check_reason_rules(_record(stance=stance, tag="其它")) == []


def test_legal_format_does_not_prove_the_reason_is_true():
    record = _record(sentence="进店中位数达到99999，但单次互动成本偏高。")
    # 故意放入错误数字：当前函数只检查生成状态与标签，不冒充事实核对。
    assert check_reason_rules(record) == []


@pytest.mark.parametrize("band", ["clear_yes", "mid", "clear_no"])
def test_required_missing_metric_is_found_in_every_band_even_if_scoring_skipped_it(band):
    record = _record(
        band=band, talent={"visit_store_med": 800, "cpuv": None},
        missing_fields=[], evidence={"cpuv": 10},
        score_parts=[{"field": "visit_store_med", "raw": 800}],
    )
    issues = check_required_data(record, _policy("visit_store_med", "cpuv"))
    assert issues[0]["code"] == "data_missing"
    assert issues[0]["fields"] == ["cpuv"]


def test_irrelevant_video_price_and_empty_agency_are_not_automatically_missing():
    assert check_required_data(_record(), _policy("visit_store_med", "cpuv", "mcn")) == []


def test_zero_is_real_data_and_stale_missing_marker_does_not_trigger_refill():
    record = _record(talent={"visit_store_med": 0}, missing_fields=["visit_store_med"])
    assert check_required_data(record, _policy("visit_store_med")) == []


def test_an_explicit_is_missing_condition_does_not_demand_filling_its_target():
    policy = {"criteria": [{"field": "price_video", "operator": "is_missing"}]}
    assert check_required_data(_record(), policy) == []


def test_downstream_can_explicitly_request_video_price_and_inputs_stay_unchanged():
    record = _record()
    policy = _policy("visit_store_med", "cpuv")
    original = deepcopy((record, policy))
    issues = check_required_data(record, policy, additional_fields=("price_video", "price_video"))
    assert issues[0]["fields"] == ["price_video"]
    assert (record, policy) == original


def test_misspelling_an_explicit_required_field_is_a_programming_error():
    with pytest.raises(ValueError, match="额外必需字段"):
        check_required_data(_record(), _policy(), additional_fields=("invented_field",))


def test_a_broken_record_is_not_reported_as_a_missing_business_metric():
    with pytest.raises(ValueError, match="record.talent"):
        check_required_data({"nickname": "测试达人"}, _policy("cpuv"))
