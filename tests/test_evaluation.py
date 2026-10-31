"""中间档理由：闭集校验，以及 Send 扇出后的上下文隔离。"""
from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from agent.evaluation import EvaluationAgent
from agent.nodes.evaluation import EvaluationNodes, collect_reasons, route_reasons
from agent.requirements import LLMClient
from agent.state import GraphState


def _mid(nickname: str, visit: float = 800) -> dict[str, Any]:
    talent = {
        "nickname": nickname,
        "xhs_id": None,
        "visit_store_med": visit,
        "cpuv": 10,
        "main_format": "图文为主",
    }
    return {
        "xhs_id": None,
        "nickname": nickname,
        "band": "mid",
        "score": 0.55,
        "missing_fields": [],
        "attempt": 1,
        "talent": talent,
        "score_parts": [
            {
                "field": "visit_store_med",
                "operator": "maximize",
                "label": "进店中位数",
                "raw": visit,
                "normalized": 0.8,
                "weight": 0.6,
                "weight_effective": 0.6,
                "contribution": 0.48,
            },
            {
                "field": "cpuv",
                "operator": "minimize",
                "label": "进店成本",
                "raw": 10,
                "normalized": 0.2,
                "weight": 0.4,
                "weight_effective": 0.4,
                "contribution": 0.08,
            },
        ],
    }


def _valid() -> dict[str, str]:
    return {
        "stance": "recommend",
        "tag": "进店好",
        "sentence": "进店中位数较高，但进店成本只是中等，先留在中间档。",
    }


def test_invalid_tag_retries_once_then_keeps_the_row() -> None:
    calls = {"count": 0}

    def transport(_system: str, payload: dict[str, Any]) -> dict[str, Any]:
        calls["count"] += 1
        assert payload["candidate"]["nickname"] == "早八"
        assert "ctr" not in str(payload)
        if calls["count"] == 1:
            return {"stance": "recommend", "tag": "非常好", "sentence": "随便写的。"}
        return _valid()

    draft = EvaluationAgent(LLMClient(transport=transport)).write_reason(_mid("早八"))
    assert calls["count"] == 2
    assert draft["tag"] == "进店好"
    assert draft["error"] is None
    assert draft["evidence"]["visit_store_med"] == 800
    assert "非常好" not in str(draft["evidence"])


def test_second_invalid_response_does_not_raise() -> None:
    def transport(_system: str, _payload: dict[str, Any]) -> dict[str, Any]:
        return {"stance": "recommend", "tag": "数据差", "sentence": "标签站错边。"}

    draft = EvaluationAgent(LLMClient(transport=transport)).write_reason(_mid("早八"))
    assert draft["tag"] is None
    assert draft["error"]
    assert draft["nickname"] == "早八"


def test_fanout_hides_other_talents_and_isolates_failures() -> None:
    def transport(_system: str, payload: dict[str, Any]) -> dict[str, Any]:
        nickname = payload["candidate"]["nickname"]
        text = str(payload)
        assert nickname in {"甲", "乙"}
        assert ("甲" in text) != ("乙" in text)
        if nickname == "甲":
            return {"stance": "reject", "tag": "发明标签", "sentence": "不行。"}
        return _valid()

    nodes = EvaluationNodes(LLMClient(transport=transport))
    graph = StateGraph(GraphState)
    graph.add_node("write_one_reason", nodes.write_one_reason)
    graph.add_node("collect_reasons", collect_reasons)
    graph.add_conditional_edges(START, route_reasons)
    graph.add_edge("write_one_reason", "collect_reasons")
    graph.add_edge("collect_reasons", END)

    first, second = _mid("甲", 700), _mid("乙", 900)
    result = graph.compile().invoke(
        {
            "reason_queue": [first, second],
            "scored_talents": [first, second],
            "requirement_profile": {"exclusions": ["机构营销号"], "kpi": "进店"},
            "brief": {"compliance": "不承诺具体效果"},
            "approved_policy": {"criteria": []},
        }
    )
    by_name = {item["nickname"]: item for item in result["evaluated_talents"]}
    assert by_name["甲"]["error"]
    assert by_name["甲"]["tag"] is None
    assert by_name["乙"]["tag"] == "进店好"
    assert by_name["乙"]["error"] is None
    assert result["workflow_status"] == "reasons_ready"
    assert result["next_step"]["name"] == "audit_reasons"


def test_empty_mid_queue_skips_the_model() -> None:
    called = {"count": 0}

    def transport(_system: str, _payload: dict[str, Any]) -> dict[str, Any]:
        called["count"] += 1
        return _valid()

    nodes = EvaluationNodes(LLMClient(transport=transport))
    graph = StateGraph(GraphState)
    graph.add_node("write_one_reason", nodes.write_one_reason)
    graph.add_node("collect_reasons", collect_reasons)
    graph.add_conditional_edges(START, route_reasons)
    graph.add_edge("write_one_reason", "collect_reasons")
    graph.add_edge("collect_reasons", END)
    clear = _mid("明确")
    clear["band"] = "clear_yes"
    clear["tag"] = "数据好"
    clear["source"] = "rules"
    clear["sentence"] = "点击率高。"
    result = graph.compile().invoke(
        {
            "reason_queue": [],
            "scored_talents": [clear],
            "requirement_profile": {},
            "brief": {},
            "approved_policy": {"criteria": []},
        }
    )
    assert called["count"] == 0
    assert result["evaluated_talents"][0]["tag"] == "数据好"
    assert result["evaluated_talents"][0]["source"] == "rules"


def test_overlong_sentence_stays_on_the_row() -> None:
    def transport(_system: str, _payload: dict[str, Any]) -> dict[str, Any]:
        return {"stance": "recommend", "tag": "进店好", "sentence": "很" * 51}

    draft = EvaluationAgent(LLMClient(transport=transport)).write_reason(_mid("早八"))
    assert draft["sentence"] is None
    assert draft["error"]
    assert "50" in draft["error"]
