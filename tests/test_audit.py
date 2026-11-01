"""审核业务层用假 LLM 测试输入与控制流；不证明真实模型的审核准确率。"""
from copy import deepcopy

import pytest

from agent.audit import AuditAgent
from agent.audit_contract import AuditValidationError
from agent.requirements import LLMClient, LLMResponseError, LLMServiceError


def _record(**updates):
    return {
        "xhs_id": None, "nickname": "测试达人甲", "attempt": 2, "band": "mid",
        "score": 0.55, "stance": "recommend", "tag": "进店好",
        "sentence": "进店量较高，但进店成本还有优化空间。", "error": None,
        "talent": {"visit_store_med": 800, "cpuv": 8.7436, "ctr": 0.04, "price_video": None},
        "score_parts": [{"field": "visit_store_med", "raw": 800, "normalized": 0.8, "weight": 1}],
        "evidence": {"visit_store_med": 99999},  # 故意与原始指标不同，审核输入不能依赖它
        **updates,
    }


def _context():
    return {
        "brief": {"raw": {"达人要求": "排除纯vlog生活号", "核心KPI": "曝光"}},
        "requirement_profile": {"exclusions": ["纯vlog生活号"]},
        "approved_policy": {"criteria": [{"field": "visit_store_med", "operator": "maximize"}]},
        "client_answers": "本轮仍按原需求排除纯vlog生活号。",
    }


def _pass():
    return {"verdict": "pass", "issues": []}


def _issue(code="data_conflict", fields=None):
    return {
        "verdict": "reject",
        "issues": [{"code": code, "message": "理由中的数值与指标不一致。", "fields": fields or []}],
    }


def test_payload_keeps_raw_precision_original_brief_and_single_person_context():
    def transport(_system, payload):
        candidate = payload["candidate"]
        metrics = {item["field"]: item for item in candidate["metrics"]}
        assert metrics["visit_store_med"]["raw"] == 800
        assert metrics["cpuv"]["raw"] == 8.7436
        assert metrics["ctr"]["raw"] == 0.04
        assert metrics["ctr"]["display"] == "4.0%"
        assert metrics["price_video"]["raw"] is None
        assert "evidence" not in candidate
        assert "99999" not in str(payload)
        assert "talents" not in payload and "evaluated_talents" not in payload
        assert payload["context"]["original_brief"]["达人要求"] == "排除纯vlog生活号"
        assert payload["context"]["client_answers"]
        return _pass()

    result = AuditAgent(LLMClient(transport=transport)).audit(_record(), **_context())
    assert result["verdict"] == "pass"
    assert result["nickname"] == "测试达人甲"
    assert result["attempt"] == 2


@pytest.mark.parametrize("updates,code", [
    ({"tag": "价格贵"}, "invalid_tag"),
    ({"error": "生成失败", "stance": None, "sentence": None}, "generation_error"),
])
def test_known_code_problems_return_without_calling_the_model(updates, code):
    def transport(*_args):
        pytest.fail("代码已经发现问题时不应请求模型")

    result = AuditAgent(LLMClient(transport=transport)).audit(_record(**updates), **_context())
    assert result["verdict"] == "reject"
    assert result["issues"][0]["code"] == code


def test_a_valid_business_rejection_does_not_trigger_a_retry():
    calls = []

    def transport(*_args):
        calls.append(1)
        return _issue(fields=["visit_store_med"])

    result = AuditAgent(LLMClient(transport=transport)).audit(_record(), **_context())
    assert result["verdict"] == "reject"
    assert len(calls) == 1


def test_picture_requirement_invented_tag_is_rejected_directly_and_input_is_unchanged():
    record = _record(tag="超优质达人")
    original = deepcopy(record)

    def transport(*_args):
        pytest.fail("清单外标签应由代码直接打回")

    result = AuditAgent(LLMClient(transport=transport)).audit(record, **_context())
    assert result["verdict"] == "reject"
    assert result["issues"][0]["code"] == "invalid_tag"
    assert record == original


def test_picture_requirement_reason_is_checked_against_raw_ctr_and_score_basis():
    record = _record(
        tag="数据好", sentence="点击率表现好，但进店成本仍有优化空间。",
        score_parts=[{"field": "ctr", "raw": 0.04, "normalized": 0.1, "weight": 1}],
    )
    original = deepcopy(record)

    def transport(_system, payload):
        candidate = payload["candidate"]
        ctr = next(item for item in candidate["metrics"] if item["field"] == "ctr")
        assert ctr["raw"] == 0.04 and ctr["display"] == "4.0%"
        assert candidate["assessment"]["tag"] == "数据好"
        assert candidate["score_parts"][0]["normalized"] == 0.1
        return {"verdict": "reject", "issues": [{
            "code": "data_conflict", "fields": ["ctr"],
            "message": "理由称点击率表现好，但原值为4%、当前评分构成为低分项，该说法缺乏依据。",
        }]}

    result = AuditAgent(LLMClient(transport=transport)).audit(record, **_context())
    assert result["verdict"] == "reject" and result["issues"][0]["fields"] == ["ctr"]
    assert set(result) == {"xhs_id", "nickname", "attempt", "verdict", "issues"}
    assert record == original  # 审校只给意见，不能自行把原标签或理由改正确。


def test_picture_requirement_recommended_vlog_account_is_checked_against_client_exclusion():
    record = _record()
    record["talent"]["main_format"] = "vlog生活记录为主"
    original = deepcopy(record)

    def transport(_system, payload):
        assert payload["candidate"]["assessment"]["stance"] == "recommend"
        assert payload["context"]["original_brief"]["达人要求"] == "排除纯vlog生活号"
        assert payload["context"]["structured_requirements"]["exclusions"] == ["纯vlog生活号"]
        assert next(item for item in payload["candidate"]["metrics"] if item["field"] == "main_format")["raw"] == "vlog生活记录为主"
        return {"verdict": "reject", "issues": [{
            "code": "client_exclusion", "fields": ["main_format"],
            "message": "推荐者以vlog生活记录为主，与甲方明确排除项冲突。",
        }]}

    result = AuditAgent(LLMClient(transport=transport)).audit(record, **_context())
    assert result["verdict"] == "reject" and result["issues"][0]["code"] == "client_exclusion"
    assert record == original and record["stance"] == "recommend"


@pytest.mark.parametrize("code", ["data_conflict", "client_exclusion"])
def test_missing_data_still_checks_existing_facts_and_client_exclusions(code):
    record = _record(sentence="进店中位数99999，成本未知。", talent={"visit_store_med": 800, "cpuv": None})
    context = _context()
    context["approved_policy"]["criteria"].append({"field": "cpuv", "operator": "minimize"})
    original = deepcopy(record)
    calls = []

    def transport(_system, payload):
        calls.append(payload)
        metrics = {item["field"]: item["raw"] for item in payload["candidate"]["metrics"]}
        assert metrics["visit_store_med"] == 800 and metrics["cpuv"] is None
        return _issue(code, ["visit_store_med"] if code == "data_conflict" else ["main_format"])

    result = AuditAgent(LLMClient(transport=transport)).audit(record, **context)
    assert len(calls) == 1
    assert result["verdict"] == "reject"
    assert {issue["code"] for issue in result["issues"]} == {"data_missing", code}
    assert record == original


def test_model_pass_does_not_erase_a_required_field_missing_in_code_checks():
    calls = []

    def transport(*_args):
        calls.append(1)
        return _pass()

    result = AuditAgent(LLMClient(transport=transport)).audit(
        _record(talent={"visit_store_med": None}), **_context()
    )
    assert len(calls) == 1
    assert result["verdict"] == "reject"
    assert result["issues"][0]["fields"] == ["visit_store_med"]


@pytest.mark.parametrize("first_response", [
    {"verdict": "pass", "issues": [], "sentence": "模型不能改写理由"},
    _issue("data_missing", ["cpuv"]),  # 原始指标有值，不能接受模型假报缺失
    _issue("invalid_tag"),  # 闭集已由代码检查，合法标签缺乏依据应报告 data_conflict
])
def test_invalid_answers_are_corrected_once_without_changing_assessment_round(first_response):
    calls = []

    def transport(_system, payload):
        calls.append(deepcopy(payload))
        if len(calls) == 1:
            return first_response
        assert payload["correction"]["errors"]
        return _pass()

    result = AuditAgent(LLMClient(transport=transport)).audit(_record(), **_context())
    assert len(calls) == 2
    assert result["attempt"] == 2
    assert result["verdict"] == "pass"


def test_two_invalid_answers_raise_instead_of_pretending_to_pass_or_reject():
    calls = []

    def transport(*_args):
        calls.append(1)
        return {"verdict": "reject", "issues": []}

    with pytest.raises(AuditValidationError):
        AuditAgent(LLMClient(transport=transport)).audit(_record(), **_context())
    assert len(calls) == 2


def test_invalid_json_is_retried_using_the_existing_client_error_type():
    calls = []

    def transport(_system, payload):
        calls.append(1)
        if len(calls) == 1:
            raise LLMResponseError("JSON 无法解析")
        assert "correction" in payload
        return _pass()

    assert AuditAgent(LLMClient(transport=transport)).audit(_record(), **_context())["verdict"] == "pass"
    assert len(calls) == 2


def test_service_failure_propagates_without_a_business_verdict_or_local_retry():
    calls = []

    def transport(*_args):
        calls.append(1)
        raise LLMServiceError("模型服务暂时不可用")

    with pytest.raises(LLMServiceError):
        AuditAgent(LLMClient(transport=transport)).audit(_record(), **_context())
    assert len(calls) == 1


def test_passing_a_reject_assessment_does_not_change_the_original_stance():
    record = _record(stance="reject", tag="价格贵")
    result = AuditAgent(LLMClient(transport=lambda *_: _pass())).audit(record, **_context())
    assert result["verdict"] == "pass"
    assert record["stance"] == "reject"


def test_model_can_identify_a_missing_metric_needed_by_the_reason_outside_scoring():
    result = AuditAgent(LLMClient(transport=lambda *_: _issue("data_missing", ["price_video"]))).audit(
        _record(sentence="视频报价较低，但进店成本还有优化空间。"), **_context()
    )
    assert result["issues"][0]["fields"] == ["price_video"]


def test_request_uses_active_policy_and_does_not_share_mutable_upstream_data():
    record, context = _record(), _context()
    context["active_policy"] = {"criteria": [{"field": "cpuv", "operator": "minimize"}]}
    original = deepcopy((record, context))

    def transport(_system, payload):
        assert payload["context"]["active_policy"]["criteria"][0]["field"] == "cpuv"
        payload["candidate"]["score_parts"].clear()
        payload["context"]["structured_requirements"]["exclusions"].clear()
        payload["context"]["approved_policy"]["criteria"].clear()
        return _pass()

    AuditAgent(LLMClient(transport=transport)).audit(record, **context)
    assert (record, context) == original


@pytest.mark.parametrize("band", ["clear_yes", "clear_no"])
def test_clear_bands_cannot_accidentally_enter_model_reason_audit(band):
    def transport(*_args):
        pytest.fail("明确档不应进入理由审核模型")

    with pytest.raises(ValueError, match="只接收中间档"):
        AuditAgent(LLMClient(transport=transport)).audit(_record(band=band), **_context())
