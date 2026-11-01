"""审核末端交接：补数失败标记放行，未解决的单条记录暂存，其余候选继续。"""
from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

from agent.audit_reporting import audit_report_fields
from agent.reasons import STANCE_RECOMMEND, person_key


def finalize_audit(state: dict[str, Any]) -> dict[str, Any]:
    """按 workflow 标记补数失败，不将缺数等同于审核通过或提前人工审批。"""
    records = {person_key(row): deepcopy(row) for row in state["audited_talents"]}
    supplements = []
    unresolved = []
    held_keys: set[str] = set()
    seen: set[str] = set()
    for task in state.get("audit_pending") or []:
        key = person_key(task)
        record = records.get(key)
        if record is None or key in seen or task["attempt"] != record["attempt"]:
            raise ValueError("审核交接任务无法唯一对应当前达人及评估轮次")
        seen.add(key)
        action = task["action"]
        entry = deepcopy(task)
        entry.update({
            "original_action": action,
            "data_incomplete": any(issue["code"] == "data_missing" for issue in task["issues"]),
            "assessment": {name: deepcopy(record.get(name)) for name in ("band", "score", "stance", "tag", "sentence")},
        })
        if action == "supplement_data":
            saved = next((item for item in reversed(state.get("supplement_history") or [])
                          if person_key(item) == key and item["attempt"] == task["attempt"]
                          and item.get("data_revision", 0) == int(record.get("data_revision") or 0)
                          and item["fields"] == task["fields"]), None)
            lookup = deepcopy(saved["input_check"]) if saved else lookup_loaded_input(record, task["fields"], state)
            entry["input_check"] = lookup
            entry["reason_code"] = "required_data_unavailable"
            if lookup["available_values"]:
                entry["action"] = "restore_input_data"
                entry["reason_code"] = "input_and_assessment_disagree"
                entry["message"] = "原始读表记录中有部分所需值，与当前评估记录不一致；该达人暂不进入可投放候选，保留原值与问题供后续复核，其余候选继续。"
            else:
                entry["message"] = "当前输入中没有可唯一确认的所需字段值，也未接入同一达人的补充来源；保留空值，标记数据不全后放行至投放组合，留待后续人工审核裁决。"
                # workflow 的放行例外只针对补数失败，不能覆盖同一记录上的其他审核问题。
                if all(issue["code"] == "data_missing" for issue in task["issues"]):
                    entry.update(action="flagged_release", status="failed")
                    record.update(
                        data_incomplete=True,
                        missing_data_fields=deepcopy(task["fields"]),
                        release_status="flagged_release",
                    )
                    supplements.append(entry)
                    continue
                raise ValueError("混合审核问题应先返回评估，不能当成单纯补数任务交接")
        elif action == "rewrite_limit_reached":
            entry["reason_code"] = "revision_limit_reached"
            entry["message"] = "已到自动重写上限，该达人审核仍未通过；保留记录及问题，暂不进入可投放候选，其余候选继续，问题随方案交给最终人工审核。"
        elif action == "retry_audit":
            entry["reason_code"] = "audit_execution_error"
            entry["message"] = "该达人的审核回答未成功生成有效结论；保留错误，暂不进入可投放候选，其余候选继续，问题随方案交给最终人工审核。"
        elif action == "review_supplement_data":
            entry["reason_code"] = "post_supplement_policy_conflict"
            entry["message"] = "补回数据后发现不满足已批准的实际执行条件；该人暂存待复核，其余候选继续，不自动放宽条件。"
        else:
            raise ValueError("仍有可重写或未知任务，不能跳过自动处理完成审核交接")
        unresolved.append(entry)
        held_keys.add(key)
        record.update(release_status="held_for_review", hold_reason_code=entry["reason_code"])

    # 单条暂存不阻断整批；缺数放行例外也不能覆盖暂存状态。原推荐倾向不变。
    candidates = [deepcopy(row) for key, row in records.items()
                  if key not in held_keys and (row.get("release_status") == "flagged_release" or
                      (key not in seen and row["audit_status"] in ("passed", "skipped_clear_band")))
                  and row.get("stance") == STANCE_RECOMMEND]
    if not candidates:
        message = "没有可用的推荐候选，投放组合无法生成；完整名单和未解决问题已保留，不将暂存记录视为审核通过。"
    elif unresolved:
        message = "投放组合尚未实现；可用候选已就绪，暂存达人不参与组合。须将完整审核名单、暂存问题及缺数标记一起传至最终人工审核。"
    elif supplements:
        message = "投放组合尚未实现；补数失败者已标记数据不全并放行，后续须携带标记至人工审核。"
    else:
        message = "投放组合尚未实现。"
    next_step = {
        "name": "build_portfolio", "display_name": "投放组合，拟定价格",
        "status": "interface_ready" if candidates else "no_candidates", "implemented": False,
        "required_state": ["pricing_candidates", "audited_talents", "audit_pending", "supplement_results",
                           "audit_unverified_checks", "audit_summary", "brief", "requirement_profile", "active_policy"],
        "message": message,
    }
    result = {
        "audited_talents": list(records.values()),
        "audit_pending": unresolved,
        "supplement_results": [deepcopy(item) for item in state.get("supplement_history") or []
                               if item["status"] in ("success", "partial")] + supplements,
        "manual_review_queue": [], "pricing_candidates": candidates,
        "workflow_status": "audit_complete_with_pending" if unresolved else (
            "audit_complete_with_incomplete_data" if supplements else "audit_complete"
        ),
        "next_step": next_step,
    }
    result.update(audit_report_fields({**state, **result}))
    return result


def lookup_loaded_input(record: dict[str, Any], fields: list[str], state: dict[str, Any]) -> dict[str, Any]:
    """只核对组员的 load_inputs 已读取的同一批记录，不扫描别的项目或猜测旧表行号。"""
    identity = record.get("xhs_id")
    match_field = "xhs_id" if identity else "nickname"
    match_value = identity or record.get("nickname")
    matches = [row for row in state.get("talents") or [] if row.get(match_field) == match_value]
    available = {}
    if len(matches) == 1:
        for field in fields:
            value = matches[0].get(field)
            if _available(value):
                available[field] = deepcopy(value)
    status = "checked" if len(matches) == 1 else ("ambiguous_record" if matches else "record_not_found")
    return {
        "source_path": state.get("talent_path"), "match_field": match_field, "status": status,
        "available_values": available,
        "unavailable_fields": [field for field in fields if field not in available],
        "supplementary_source_status": "not_connected",
    }


def _available(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    return isinstance(value, str) and value.strip().casefold() not in {"", "-", "—", "/", "nan", "n/a", "无"}
