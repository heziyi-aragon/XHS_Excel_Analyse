"""真实首轮报价反馈暴露的引用回归，所有模型回答均为本地固定数据。"""
from copy import deepcopy

import pytest

from agent.portfolio import build_portfolio_plan
from agent.portfolio_planning import prepare_portfolio_config
from tests.test_portfolio import _record, _state


class LocalResponses:
    """只返回构造响应，不创建网络客户端。"""

    def __init__(self, response):
        self.response = deepcopy(response)
        self.calls = 0

    def complete_json(self, *_args, **_kwargs):
        self.calls += 1
        return deepcopy(self.response)


def _brief_state(budget_quote="¥350,000（含达人合作费用，不含投流）"):
    state = _state([_record(price_image=9500, price_video=14000)], budget=350000)
    state["brief"].update({
        "product_price": "¥169/瓶",
        "notes": "达人报价以平台参考价为基准，上下浮动20%以内，超出需单独说明理由",
        "raw": {
            "总预算": budget_quote,
            "产品定价": "¥169/瓶",
            "达人参考价": "¥9,500",
            "备注": "达人报价以平台参考价为基准，上下浮动20%以内，超出需单独说明理由",
        },
    })
    return state


def _proposal(path, value, quote, *, kind="brief"):
    return {"path": path, "value": value, "source_quote": quote, "source_kind": kind}


def _effective_budget(state, config):
    override = config.get("budget")
    return override["value"] if isinstance(override, dict) else state["brief"]["total_budget"]


@pytest.mark.parametrize("quote", [
    "¥350,000（含达人合作费用，不含投流）",
    "￥350，000（含达人合作费用，不含投流）",
    "￥３５０，０００（含达人合作费用，不含投流）",
    "总预算：¥350,000（含达人合作费用，不含投流）",
])
def test_the_known_brief_budget_is_verified_from_its_field_even_when_the_quote_has_no_label(quote):
    state = _brief_state(quote)
    original = deepcopy(state)
    client = LocalResponses({"proposals": [_proposal("budget", 350000, quote)]})
    config, metadata = prepare_portfolio_config(state, {}, client)
    assert metadata["status"] == "ready"
    assert metadata["pending"] == []
    assert _effective_budget(state, config) == 350000
    assert not any(item.get("proposal", {}).get("path") == "budget"
                   for item in metadata["rejected"])
    assert state == original
    assert client.calls == 1


def test_the_default_twenty_percent_corridor_is_not_executed_as_a_fixed_multiplier():
    state = _brief_state()
    quote = "以平台参考价为基准，上下浮动20%以内"
    client = LocalResponses({"proposals": [_proposal("price_multiplier", 1.2, quote)]})
    config, metadata = prepare_portfolio_config(state, {}, client)
    assert config.get("price_multiplier") != 1.2
    assert metadata["rejected"]
    assert metadata["status"] == "ready"
    assert metadata["pending"] == []
    plan = build_portfolio_plan(state, llm_client=client)
    assert plan["lines"]
    assert plan["pricing_policy"]["mode"] == "benchmark_auto"
    assert plan["lines"][0]["unit_price_suggested"] == 9500
    assert plan["budget"]["limit"] == 350000


def test_the_original_screenshot_proposals_do_not_prevent_a_safe_first_round_plan():
    state = _brief_state()
    response = {"proposals": [
        _proposal("budget", 350000, state["brief"]["raw"]["总预算"]),
        _proposal("price_multiplier", 1.2, "以平台参考价为基准，上下浮动20%以内"),
    ]}
    original = deepcopy(state)
    plan = build_portfolio_plan(state, llm_client=LocalResponses(response))
    assert plan["planning"]["status"] == "ready"
    assert plan["budget"]["limit"] == 350000
    assert plan["budget"]["within_budget"] is True
    assert plan["pricing_policy"]["mode"] == "benchmark_auto"
    assert plan["configuration"].get("price_multiplier") != 1.2
    assert state == original


@pytest.mark.parametrize("response", [
    {"proposals": [_proposal("unknown_constraint", 1, "不存在的原文")]},
    {"proposals": [_proposal("price_multiplier", 0.1, "不存在的原文")]},
    {"proposals": "not-a-list"},
    {"unexpected": "model format error"},
])
def test_an_unusable_model_suggestion_cannot_block_an_otherwise_valid_first_round(response):
    state = _brief_state()
    config, metadata = prepare_portfolio_config(state, {}, LocalResponses(response))
    assert metadata["status"] == "ready"
    assert metadata["pending"] == []
    assert metadata["rejected"]
    assert _effective_budget(state, config) == 350000
    assert config.get("price_multiplier") != 0.1


@pytest.mark.parametrize("value,quote", [
    (169, "¥169/瓶"),
    (9500, "¥9,500"),
    (350, "350"),
    (350001, "¥350,000（含达人合作费用，不含投流）"),
])
def test_other_money_or_partial_digits_cannot_replace_the_authoritative_budget(value, quote):
    state = _brief_state()
    original = deepcopy(state)
    config, metadata = prepare_portfolio_config(
        state, {}, LocalResponses({"proposals": [_proposal("budget", value, quote)]}))
    assert _effective_budget(state, config) == 350000
    assert not any(item.get("path") == "budget" for item in metadata["accepted"])
    assert metadata["rejected"]
    assert state == original


def test_a_budget_quote_conflicting_with_the_structured_budget_cannot_overwrite_it():
    state = _brief_state("¥360,000（含达人合作费用，不含投流）")
    original = deepcopy(state)
    config, metadata = prepare_portfolio_config(state, {}, LocalResponses({
        "proposals": [_proposal("budget", 360000, state["brief"]["raw"]["总预算"])]}))
    assert _effective_budget(state, config) == 350000
    assert not any(item.get("path") == "budget" for item in metadata["accepted"])
    assert metadata["status"] == "needs_clarification"
    assert metadata["pending"]
    assert state == original


@pytest.mark.parametrize("feedback", ["重新定价", "九折并新增ghost", "让不存在的达人改成视频"])
def test_unresolved_actual_feedback_still_requires_clarification_after_model_output_is_quarantined(feedback):
    state = _brief_state()
    config, metadata = prepare_portfolio_config(state, {},
                                               LocalResponses({"unexpected": "invalid"}), feedback)
    assert metadata["status"] == "needs_clarification"
    assert metadata["pending"]
    assert any(item["source_kind"] == "feedback" for item in metadata["pending"])
    assert "ghost" not in config.get("per_creator_allowed_formats", {})
    assert _effective_budget(state, config) == 350000


def test_conflicting_customer_budget_answers_still_block_instead_of_choosing_one_value():
    state = _brief_state()
    state["client_answers"] = "总预算调整为10万元；总预算调整为20万元。"
    config, metadata = prepare_portfolio_config(state, {}, LocalResponses({"proposals": []}))
    assert metadata["status"] == "needs_clarification"
    assert any(item["source_kind"] == "client_answers" for item in metadata["pending"])
    assert not any(item.get("path") == "budget" and item.get("source_kind") == "client_answers"
                   for item in metadata["accepted"])
    assert _effective_budget(state, config) == 350000


@pytest.mark.parametrize("source", ["client_answers", "brief"])
def test_unsupported_hard_video_quantities_are_pending_rather_than_silently_dropped(source):
    state = _brief_state()
    if source == "client_answers":
        state["client_answers"] = "视频至少2篇"
    else:
        state["brief"]["content_direction"] = "视频至少2篇"
        state["brief"]["raw"]["内容方向"] = "视频至少2篇"
    _, metadata = prepare_portfolio_config(state, {}, LocalResponses({"proposals": []}))
    assert metadata["status"] == "needs_clarification"
    assert metadata["pending"]


def test_soft_image_and_video_preferences_do_not_become_a_hard_quantity_requirement():
    state = _brief_state()
    state["brief"]["content_direction"] = "图文为主、视频为辅"
    state["brief"]["raw"]["内容方向"] = "图文为主、视频为辅"
    _, metadata = prepare_portfolio_config(state, {}, LocalResponses({"proposals": []}))
    assert metadata["status"] == "ready"
    assert metadata["pending"] == []


@pytest.mark.parametrize("feedback", [
    "不要按九折",
    "不再使用0.9倍",
    "报价上限1.2倍",
    "上次九折，本次每人2篇",
    "如果能谈到九折，就每人2篇",
])
def test_negated_bounded_historical_or_conditional_text_is_not_a_current_fixed_multiplier(feedback):
    state = _brief_state()
    config, metadata = prepare_portfolio_config(state, {}, feedback=feedback)
    assert "price_multiplier" not in config
    assert metadata["status"] == "needs_clarification"
    assert metadata["pending"]


def test_a_model_cannot_strip_the_negation_from_its_quote_to_execute_the_opposite_request():
    state = _brief_state()
    feedback = "不要按九折"
    client = LocalResponses({"proposals": [_proposal("price_multiplier", 0.9, "九折", kind="feedback")]})
    config, metadata = prepare_portfolio_config(state, {}, client, feedback)
    assert "price_multiplier" not in config
    assert metadata["status"] == "needs_clarification"
    assert metadata["pending"]
    assert any(item.get("proposal", {}).get("path") == "price_multiplier"
               for item in metadata["rejected"])


def test_an_unsupported_minimum_creator_count_in_the_customer_answer_is_pending():
    state = _brief_state()
    state["client_answers"] = "达人数量至少10位"
    _, metadata = prepare_portfolio_config(state, {})
    assert metadata["status"] == "needs_clarification"
    assert metadata["pending"]


def test_a_conditional_customer_post_count_is_not_executed_as_a_confirmed_cap():
    state = _brief_state()
    state["client_answers"] = "如果天气允许，每人2篇"
    config, metadata = prepare_portfolio_config(state, {})
    assert "max_posts_per_creator" not in config
    assert metadata["status"] == "needs_clarification"
    assert metadata["pending"]


def test_a_soft_format_share_preference_without_a_numeric_or_hard_requirement_is_ready():
    state = _brief_state()
    state["brief"]["content_direction"] = "图文占比优先，视频为辅"
    state["brief"]["raw"]["内容方向"] = "图文占比优先，视频为辅"
    _, metadata = prepare_portfolio_config(state, {})
    assert metadata["status"] == "ready"
    assert metadata["pending"] == []


def test_a_budget_number_in_notes_is_not_verified_as_the_authoritative_budget_field():
    state = _brief_state()
    quote = "业务说明：总预算35万元"
    state["brief"]["notes"] = quote
    client = LocalResponses({"proposals": [_proposal("budget", 350000, quote)]})
    config, metadata = prepare_portfolio_config(state, {}, client)
    assert metadata["status"] == "ready"
    assert metadata["pending"] == []
    assert _effective_budget(state, config) == 350000
    assert any(item.get("proposal", {}).get("path") == "budget"
               for item in metadata["rejected"])
    assert not any(item.get("path") == "budget" and item.get("method", "").startswith("model")
                   for item in metadata["accepted"])
