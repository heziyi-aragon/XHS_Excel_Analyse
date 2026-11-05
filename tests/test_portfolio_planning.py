"""求解前约束翻译：自然语言、来源核对和越权边界，完全离线。"""
from copy import deepcopy

import pytest

from agent.portfolio_planning import prepare_portfolio_config
from tests.test_portfolio import _record, _state


def _input():
    state = _state([_record("kol-1", price_image=100, price_video=200)], budget=350000)
    state["pricing_candidates"][0]["nickname"] = "小桃"
    state["pricing_candidates"][0]["talent"]["nickname"] = "小桃"
    state["brief"]["kpi"] = "曝光≥1000；互动≥100；整体CPE≤2；品牌词搜索指数提升≥30%"
    return state


@pytest.mark.parametrize("text,expected", [("九折", 0.9), ("按9折重新计算", 0.9),
                                           ("报价按0.9倍计算", 0.9), ("85折", 0.85)])
def test_clear_discount_feedback_is_executable_without_a_model(text, expected):
    cfg, metadata = prepare_portfolio_config(_input(), {}, feedback=text)
    assert cfg["price_multiplier"] == expected
    assert metadata["status"] == "ready"
    assert metadata["model_used"] is False
    assert any(item["path"] == "price_multiplier" and item["source_quote"] in text
               for item in metadata["accepted"])


@pytest.mark.parametrize("text", ["每人2篇", "每人最多2篇"])
def test_clear_quantity_feedback_updates_the_creator_cap(text):
    cfg, metadata = prepare_portfolio_config(_input(), {}, feedback={"comments": text})
    assert cfg["max_posts_per_creator"] == 2
    assert metadata["status"] == "ready"


@pytest.mark.parametrize("text,allowed", [("小桃改成视频", ["video"]), ("小桃改为图文", ["image"])])
def test_known_creator_format_feedback_becomes_an_identity_bound_constraint(text, allowed):
    state = _input()
    original = deepcopy(state)
    cfg, metadata = prepare_portfolio_config(state, {}, feedback=text)
    assert cfg["per_creator_allowed_formats"]["kol-1"] == allowed
    assert metadata["status"] == "ready"
    assert state == original


def test_latest_customer_budget_answer_has_a_traceable_numeric_source():
    state = _input()
    state["client_answers"] = "总预算调整为10万元，其他要求不变。"
    cfg, metadata = prepare_portfolio_config(state, {})
    assert cfg["budget"]["value"] == 100000
    assert cfg["budget"]["source_quote"] in state["client_answers"]
    assert "client_answers" in cfg["budget"]["source"]
    assert any(item["path"] == "budget" and item["source_kind"] == "client_answers"
               for item in metadata["accepted"])


def test_explicit_config_paths_take_priority_and_internal_context_is_removed():
    config = {"price_multiplier": 0.85,
              "__planning_context__": {"explicit_paths": ["price_multiplier"]}}
    original = deepcopy(config)
    cfg, metadata = prepare_portfolio_config(_input(), config, feedback="九折")
    assert cfg["price_multiplier"] == 0.85
    assert "__planning_context__" not in cfg
    assert config == original
    assert not any(item["path"] == "price_multiplier" and item["value"] == 0.9
                   for item in metadata["accepted"])


@pytest.mark.parametrize("text", ["重新定价", "更有性价比一些", "让没有提供过的陌生达人改成视频"])
def test_unresolved_language_is_pending_instead_of_claiming_execution(text):
    cfg, metadata = prepare_portfolio_config(_input(), {}, feedback=text)
    assert metadata["status"] == "needs_clarification"
    assert metadata["pending"]
    assert "per_creator_allowed_formats" not in cfg


@pytest.mark.parametrize("text", ["将预算提高为100万元", "把曝光目标降低到500", "整体CPE放宽为25元"])
def test_consultant_feedback_cannot_expand_customer_budget_or_relax_customer_kpis(text):
    cfg, metadata = prepare_portfolio_config(_input(), {}, feedback=text)
    assert cfg.get("budget", {}).get("value", 350000) <= 350000
    assert cfg.get("targets", {}).get("exposure", {}).get("value", 1000) >= 1000
    assert cfg.get("targets", {}).get("cpe", {}).get("value", 2) <= 2
    assert metadata["pending"] or metadata["rejected"]


def test_model_proposals_need_verifiable_quotes_values_and_existing_identity():
    class MaliciousModel:
        calls = 0

        def complete_json(self, *_args, **_kwargs):
            self.calls += 1
            return {"proposals": [
                {"path": "budget.value", "value": 999999999,
                 "source_quote": "总预算35万元", "source_kind": "brief"},
                {"path": "price_multiplier", "value": 0.1,
                 "source_quote": "九折", "source_kind": "feedback"},
                {"path": "per_creator_allowed_formats.ghost", "value": ["video"],
                 "source_quote": "ghost改成视频", "source_kind": "feedback"},
                {"path": "unit_price_suggested", "value": 1,
                 "source_quote": "九折", "source_kind": "feedback"},
                {"path": "max_posts_per_creator", "value": 10,
                 "source_quote": "根本不存在的引文", "source_kind": "feedback"},
            ]}

    state = _input()
    state["brief"]["raw"] = {"总预算": "总预算35万元"}
    original = deepcopy(state)
    client = MaliciousModel()
    cfg, metadata = prepare_portfolio_config(state, {}, client, "九折，ghost改成视频")
    assert client.calls == 1
    assert metadata["model_used"] is True
    assert len(metadata["rejected"]) >= 5
    assert cfg.get("budget", {}).get("value", 350000) <= 350000
    assert cfg["price_multiplier"] == 0.9
    assert "ghost" not in cfg.get("per_creator_allowed_formats", {})
    assert "unit_price_suggested" not in cfg
    assert cfg.get("max_posts_per_creator", 1) != 10
    assert state == original


def test_a_valid_model_proposal_is_applied_before_solving_and_records_provenance():
    class Model:
        def complete_json(self, *_args, **_kwargs):
            return {"proposals": [{"path": "max_posts_per_creator", "value": 2,
                                   "source_quote": "每人2篇", "source_kind": "feedback"}]}

    cfg, metadata = prepare_portfolio_config(_input(), {}, Model(), {"message": "每人2篇"})
    assert cfg["max_posts_per_creator"] == 2
    assert metadata["model_used"] is True
    assert metadata["status"] == "ready"
    assert any(item["path"] == "max_posts_per_creator" for item in metadata["accepted"])


def test_optional_model_failure_preserves_deterministic_feedback_execution():
    class FailedModel:
        def complete_json(self, *_args, **_kwargs):
            raise RuntimeError("offline model unavailable")

    cfg, metadata = prepare_portfolio_config(_input(), {}, FailedModel(), "九折")
    assert cfg["price_multiplier"] == 0.9
    assert metadata["status"] == "ready"
    assert metadata["rejected"]


def test_budget_number_does_not_consume_the_following_chinese_discount_clause():
    cfg, metadata = prepare_portfolio_config(_input(), {}, feedback="预算10万，九折")
    assert cfg["budget"]["value"] == 100000
    assert cfg["price_multiplier"] == 0.9
    assert metadata["status"] == "ready"


def test_executing_one_clause_does_not_hide_an_unresolved_second_instruction():
    cfg, metadata = prepare_portfolio_config(_input(), {}, feedback="九折并新增ghost")
    assert cfg["price_multiplier"] == 0.9
    assert metadata["status"] == "needs_clarification"
    assert metadata["pending"]


@pytest.mark.parametrize("text,path,value", [
    ("按九折重新报价", "price_multiplier", 0.9),
    ("每人最多2篇，重新定价", "max_posts_per_creator", 2),
])
def test_repricing_words_do_not_block_an_already_specific_executable_change(text, path, value):
    cfg, metadata = prepare_portfolio_config(_input(), {}, feedback=text)
    assert cfg[path] == value
    assert metadata["status"] == "ready"
    assert metadata["pending"] == []
