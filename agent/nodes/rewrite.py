"""将可修正的审核问题转为单人重写任务，复用已有 Send 扇出。"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from agent.reasons import BAND_MID, MAX_REASON_ATTEMPTS, person_key


def route_after_audit(state: dict[str, Any]) -> str:
    """只自动处理可重写任务；其他未解决事项保留给后续工具或人工。"""
    for task in state.get("audit_pending") or []:
        if task["action"] == "rewrite_reason" and task["attempt"] < MAX_REASON_ATTEMPTS:
            return "rewrite"
    records = {person_key(row): row for row in state.get("audited_talents") or []}
    for task in state.get("audit_pending") or []:
        if task["action"] != "supplement_data":
            continue
        record = records.get(person_key(task), {})
        if not any(
            person_key(item) == person_key(task) and item["attempt"] == task["attempt"]
            and item.get("data_revision", 0) == int(record.get("data_revision") or 0)
            and item["fields"] == task["fields"]
            for item in state.get("supplement_history") or []
        ):
            return "supplement"
    return "finish"


def prepare_reason_rewrites(state: dict[str, Any]) -> dict[str, Any]:
    """只重建被打回者的队列；每条任务携带原评估和具体问题，评估轮次增加 1。"""
    records = {person_key(record): record for record in state["evaluated_talents"]}
    queue: list[dict[str, Any]] = []
    seen: set[str] = set()
    for task in state.get("audit_pending") or []:
        if task["action"] != "rewrite_reason":
            continue
        key = person_key(task)
        record = records.get(key)
        if record is None or key in seen:
            raise ValueError("重写任务的达人无法匹配，或任务重复")
        if record.get("band") != BAND_MID or task["attempt"] != record["attempt"]:
            raise ValueError("重写任务的档位或被审轮次与当前评估不一致")
        if record["attempt"] >= MAX_REASON_ATTEMPTS:
            raise ValueError("已达到评估轮次上限，不能继续自动重写")
        if not task.get("issues"):
            raise ValueError("重写任务必须携带具体审核问题")
        seen.add(key)
        payload = deepcopy({
            name: record.get(name)
            for name in ("xhs_id", "nickname", "band", "score", "score_parts", "missing_fields", "talent", "data_revision")
        })
        payload["attempt"] = record["attempt"] + 1
        payload["review_feedback"] = {
            "previous_assessment": {
                name: record.get(name) for name in ("stance", "tag", "sentence")
            },
            "issues": deepcopy(task["issues"]),
        }
        queue.append(payload)
    if not queue:
        raise ValueError("没有可执行的理由重写任务")
    return {"reason_queue": queue, "workflow_status": "rewriting_reasons"}
