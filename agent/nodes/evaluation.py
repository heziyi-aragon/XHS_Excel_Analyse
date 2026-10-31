"""第七步的图节点：按中间档达人 Send，再把并行结果汇总成一份名单。"""
from __future__ import annotations

from typing import Any

from langgraph.types import Send

from agent.evaluation import EvaluationAgent
from agent.reasons import BAND_MID, MAX_REASON_ATTEMPTS, person_key
from agent.requirements import LLMClient, LLMServiceError


class EvaluationNodes:
    """一个评估实例对应一条达人。节点本身不保存其他人的上下文。"""

    def __init__(self, llm: LLMClient) -> None:
        self.agent = EvaluationAgent(llm)

    def write_one_reason(self, state: dict[str, Any]) -> dict[str, Any]:
        try:
            draft = self.agent.write_reason(state)
        except LLMServiceError:
            raise
        except Exception as exc:  # 单条意外失败不能取消同批其他人
            draft = {
                "xhs_id": state.get("xhs_id"),
                "nickname": state.get("nickname"),
                "stance": None,
                "tag": None,
                "sentence": None,
                "evidence": {},
                "attempt": int(state.get("attempt") or 1),
                "error": str(exc),
                "source": "model",
            }
        return {"reason_results": [draft]}


def route_reasons(state: dict[str, Any]) -> str | list[Send]:
    """每位中间档达人一条 Send。队列为空时直接去汇总。"""
    profile = state.get("requirement_profile") or {}
    brief = state.get("brief") or {}
    policy = state.get("approved_policy") or {}
    shared = {
        "exclusions": profile.get("exclusions") or [],
        "kpi": profile.get("kpi") or "",
        "hard_requirements": profile.get("hard_requirements") or [],
        "compliance": brief.get("compliance") or "",
        "policy_criteria": policy.get("criteria") or [],
    }
    sends: list[Send] = []
    for item in state.get("reason_queue") or []:
        attempt = int(item.get("attempt") or 1)
        if attempt > MAX_REASON_ATTEMPTS:
            continue
        sends.append(
            Send(
                "write_one_reason",
                {
                    "xhs_id": item.get("xhs_id"),
                    "nickname": item.get("nickname"),
                    "band": item.get("band"),
                    "score": item.get("score"),
                    "score_parts": item.get("score_parts") or [],
                    "missing_fields": item.get("missing_fields") or [],
                    "talent": item.get("talent") or {},
                    "attempt": attempt,
                    **shared,
                },
            )
        )
    if not sends:
        return "collect_reasons"
    return sends


def collect_reasons(state: dict[str, Any]) -> dict[str, Any]:
    """明确档沿用代码标签；中间档合并本轮最新的一条模型结果。"""
    latest: dict[str, dict[str, Any]] = {}
    for item in state.get("reason_results") or []:
        key = person_key(item)
        previous = latest.get(key)
        if previous is None or int(item.get("attempt") or 1) >= int(
            previous.get("attempt") or 1
        ):
            latest[key] = item

    evaluated: list[dict[str, Any]] = []
    for record in state.get("scored_talents") or []:
        merged = dict(record)
        if merged.get("band") == BAND_MID:
            extra = latest.get(person_key(merged))
            if extra:
                for key in (
                    "stance",
                    "tag",
                    "sentence",
                    "evidence",
                    "attempt",
                    "error",
                    "source",
                ):
                    if key in extra:
                        merged[key] = extra[key]
            else:
                merged["error"] = merged.get("error") or "没有生成推荐理由"
        evaluated.append(merged)

    band_order = {"clear_yes": 0, "mid": 1, "clear_no": 2}

    def sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
        score = record.get("score")
        score_sort = -(score if isinstance(score, (int, float)) else -1)
        return (
            band_order.get(str(record.get("band")), 9),
            score_sort,
            str(record.get("nickname") or ""),
        )

    evaluated.sort(key=sort_key)
    return {
        "evaluated_talents": evaluated,
        "workflow_status": "reasons_ready",
        "next_step": {
            "name": "audit_reasons",
            "display_name": "审核核对",
            "status": "interface_ready",
            "implemented": False,
            "required_state": [
                "evaluated_talents",
                "approved_policy",
                "requirement_profile",
            ],
        },
    }
