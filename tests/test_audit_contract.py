"""审核接口的边界测试；不需要 API 或 LangGraph。"""
from copy import deepcopy

import pytest

from agent.audit_contract import AuditValidationError, parse_audit_result


def test_passing_a_reject_recommendation_preserves_identity_round_and_inputs():
    record = {
        "xhs_id": None, "nickname": "测试达人", "attempt": 2,
        "stance": "reject", "tag": "粉丝不匹配", "sentence": "粉丝与目标人群不符。",
    }
    response = {"verdict": "pass", "issues": []}
    original = deepcopy((record, response))
    result = parse_audit_result(response, record)
    assert result == {
        "xhs_id": None, "nickname": "测试达人", "attempt": 2,
        "verdict": "pass", "issues": [],
    }
    assert (record, response) == original
    assert record["stance"] == "reject"  # 通过审核不等于推荐达人


def test_missing_data_result_keeps_exact_fields_without_sharing_mutable_lists():
    response = {
        "verdict": "reject",
        "issues": [{"code": "data_missing", "message": "缺少进店成本，无法核对成本描述。", "fields": ["cpuv"]}],
    }
    result = parse_audit_result(response, {"xhs_id": "kol-1", "nickname": None, "attempt": 1})
    assert result["issues"][0]["fields"] == ["cpuv"]
    result["issues"][0]["fields"].append("visit_store_med")
    assert response["issues"][0]["fields"] == ["cpuv"]


@pytest.mark.parametrize("response", [
    [],
    {"verdict": "pass"},
    {"verdict": "unknown", "issues": []},
    {"verdict": "reject", "issues": []},
    {"verdict": "pass", "issues": "没有问题"},
    {"verdict": "pass", "issues": [], "stance": "recommend"},
    {"verdict": "pass", "issues": [], "tag": "数据好"},
    {"verdict": "pass", "issues": [], "sentence": "擅自修改的理由"},
    {"verdict": "pass", "issues": [], "talent": {"ctr": 0.9}},
    {"verdict": "pass", "issues": [], "nickname": "另一个达人"},
    {"verdict": "pass", "issues": [], "attempt": 99},
    {"verdict": "reject", "issues": ["有问题"]},
    {"verdict": "pass", "issues": [{"code": "data_conflict", "message": "数值矛盾", "fields": ["ctr"]}]},
    {"verdict": "reject", "issues": [{"code": "invented", "message": "说明", "fields": []}]},
    {"verdict": "reject", "issues": [{"code": "invalid_tag", "message": " ", "fields": []}]},
    {"verdict": "reject", "issues": [{"code": "data_missing", "message": "缺数", "fields": []}]},
    {"verdict": "reject", "issues": [{"code": "data_missing", "message": "缺数", "fields": ["invented_metric"]}]},
    {"verdict": "reject", "issues": [{"code": "data_conflict", "message": "数值矛盾", "fields": "ctr"}]},
])
def test_invalid_model_answer_is_an_execution_error_instead_of_a_business_verdict(response):
    with pytest.raises(AuditValidationError):
        parse_audit_result(response, {"nickname": "测试达人", "attempt": 1})


@pytest.mark.parametrize("record", [
    {"attempt": 1},
    {"nickname": "测试达人"},
    {"nickname": "测试达人", "attempt": True},
    {"nickname": "测试达人", "attempt": 0},
    {"nickname": "测试达人", "attempt": "1"},
    {"nickname": [], "attempt": 1},
])
def test_missing_identity_or_invalid_round_cannot_be_attached_to_an_audit(record):
    with pytest.raises(AuditValidationError):
        parse_audit_result({"verdict": "pass", "issues": []}, record)
