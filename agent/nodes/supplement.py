"""补数回路：只查前序同批输入，填缺失字段，复用评分与评估，不修改原表。"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from agent.nodes.audit_handoff import lookup_loaded_input
from agent.nodes.evaluation import route_reasons
from agent.reasons import BAND_MID, MAX_REASON_ATTEMPTS, person_key
from agent.screening import apply_hard_filters, clean_talent, score_candidates


def supplement_data(state: dict[str, Any]) -> dict[str, Any]:
    """有来源的值才填入工作副本；没查到的值仍留空，交末端标记放行。"""
    records = {person_key(row): row for row in state["audited_talents"]}
    history = deepcopy(state.get("supplement_history") or [])
    patches: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    for task in state.get("audit_pending") or []:
        if task["action"] != "supplement_data":
            continue
        key = person_key(task)
        record = records.get(key)
        if record is None or key in seen or record["attempt"] != task["attempt"]:
            raise ValueError("补数任务无法唯一对应当前达人及评估版本")
        if any(issue["code"] != "data_missing" for issue in task["issues"]):
            raise ValueError("混合审核问题必须先回评估修正")
        seen.add(key)
        lookup = lookup_loaded_input(record, task["fields"], state)
        values = deepcopy(lookup["available_values"])
        if any(record["talent"].get(field) is not None and
               not (isinstance(record["talent"][field], str) and not record["talent"][field].strip())
               for field in values):
            raise ValueError("补数只能填空字段，不能覆盖已有指标")
        history.append({
            "xhs_id": record.get("xhs_id"), "nickname": record.get("nickname"),
            "attempt": record["attempt"], "data_revision": int(record.get("data_revision") or 0),
            "fields": deepcopy(task["fields"]), "values": values, "input_check": lookup,
            "status": "failed" if not values else ("partial" if lookup["unavailable_fields"] else "success"),
            "assessment": deepcopy(record),
        })
        if values:
            patches[key] = values
    result: dict[str, Any] = {"supplement_history": history, "supplement_reassessment_required": bool(patches)}
    if not patches:
        return result

    baseline = state.get("scored_talents")
    if not isinstance(baseline, list) or len(baseline) != len(records):
        raise ValueError("补数成功后需要完整评分基线，不能仅改审核副本")
    if {person_key(row) for row in baseline} != set(records):
        raise ValueError("评分基线与审核名单不一致，不能重新评估")
    policy = state.get("active_policy") or state["approved_policy"]
    # 同批 min-max 标尺可能因补值变化，必须复用上游的完整评分函数。
    pool = []
    for row in baseline:
        talent = deepcopy(row["talent"])
        talent.update(patches.get(person_key(row), {}))
        pool.append(clean_talent(talent))
    rescored = score_candidates(pool, policy)
    evaluated = {person_key(row): row for row in state["evaluated_talents"]}
    scored, queue, changed = [], [], []
    for fresh in rescored:
        key = person_key(fresh)
        old = evaluated[key]
        affected = any(fresh[name] != old.get(name) for name in ("talent", "score", "score_parts", "band"))
        if not affected:
            scored.append(deepcopy(old))
            continue
        changed.append(key)
        fresh["data_revision"] = int(old.get("data_revision") or 0) + 1
        fresh["attempt"] = min(old["attempt"] + 1, MAX_REASON_ATTEMPTS)
        _, excluded = apply_hard_filters([fresh["talent"]], policy)
        if excluded:
            fresh["supplement_review_reason"] = "补回数据后不满足当前硬条件，需复核：" + excluded[0]["reason"]
        elif old["attempt"] >= MAX_REASON_ATTEMPTS:
            fresh["error"] = "补数或评分已更新，但两轮评估次数已用完，新数据尚未完成重新评估。"
        elif fresh["band"] == BAND_MID:
            queue.append(deepcopy(fresh))
        scored.append(fresh)
    # 保留源 talents 不动。工作副本和汇总基线同步更新，旧理由用 data_revision 隔离。
    result.update({
        "scored_talents": scored, "screened_talents": pool, "reason_queue": queue,
        "workflow_status": "reevaluating_after_supplement",
    })
    for entry in history:
        if entry["status"] in ("success", "partial") and entry.get("applied") is None:
            entry.update(applied=True, affected_people=changed)
    if "cleaned_talents" in state:
        result["cleaned_talents"] = [clean_talent({**deepcopy(row), **patches.get(person_key(row), {})})
                                      for row in state["cleaned_talents"]]
    return result


def route_after_supplement(state: dict[str, Any]):
    return route_reasons(state) if state.get("supplement_reassessment_required") else "finalize_audit"
