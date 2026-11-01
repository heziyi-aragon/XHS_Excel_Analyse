"""审核结果与下游交接：未变化的评估可复用，改写后独立复审，保留审核历史。"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from agent.audit import AuditAgent
from agent.audit_contract import AuditValidationError, parse_audit_result
from agent.audit_rules import check_reason_rules, check_required_data
from agent.reasons import (
    BAND_CLEAR_NO,
    BAND_CLEAR_YES,
    BAND_MID,
    MAX_REASON_ATTEMPTS,
    STANCE_RECOMMEND,
    person_key,
)
from agent.requirements import LLMClient, LLMResponseError


class AuditNodes:
    def __init__(self, llm: LLMClient) -> None:
        self.agent = AuditAgent(llm)

    def audit_reasons(self, state: dict[str, Any]) -> dict[str, Any]:
        """逐人核对并整体写回最新结果；改写回路不重复审核未变化的记录。"""
        if not isinstance(state.get("evaluated_talents"), list):
            raise ValueError("缺少 evaluated_talents，不能执行首轮审核")
        approved = state.get("approved_policy")
        policy = state.get("active_policy")
        if policy is None:
            policy = approved
        if not isinstance(approved, dict) or not isinstance(policy, dict):
            raise ValueError("缺少已批准或实际执行的策略，不能审核")

        context = deepcopy({
            "brief": state.get("brief") or {},
            "requirement_profile": state.get("requirement_profile") or {},
            "approved_policy": approved,
            "active_policy": policy,
            "client_answers": state.get("client_answers") or "",
        })
        same_context = state.get("audit_context") == context
        previous_reviews = {person_key(row): row for row in state.get("audited_talents") or []}
        previous_results = {person_key(row): row for row in state.get("audit_results") or []}
        history = deepcopy(state.get("audit_history") or [])
        results, audited, pending, candidates = [], [], [], []
        seen: set[str] = set()
        for record in state["evaluated_talents"]:
            key = person_key(record)
            if not key.strip() or key in seen:
                raise ValueError("审核名单存在空身份或重复达人，无法可靠匹配结果")
            seen.add(key)
            if type(record.get("attempt")) is not int or record["attempt"] < 1:
                raise ValueError("审核记录的 attempt 必须为正整数")

            merged = deepcopy(record)
            merged.pop("audit_error", None)
            band = record.get("band")
            if record.get("supplement_review_reason"):
                issues = [{"code": "data_conflict", "message": record["supplement_review_reason"], "fields": []}]
                status = "needs_resolution"
                if band == BAND_MID:
                    previous = previous_reviews.get(key)
                    saved = previous_results.get(key)
                    if same_context and previous is not None and saved is not None and _assessment_unchanged(record, previous):
                        result = deepcopy(saved)
                    else:
                        result = parse_audit_result({"verdict": "reject", "issues": issues}, record)
                        history.append({"assessment": deepcopy(record), "result": deepcopy(result), "context": deepcopy(context)})
                    results.append(result)
            elif band in (BAND_CLEAR_YES, BAND_CLEAR_NO):
                issues = check_required_data(record, policy)
                if record.get("error"):
                    # 补数后若已无重新评估次数，明确档也不能沿用未经确认的新结论。
                    issues += check_reason_rules(record)
                status = "needs_resolution" if issues else "skipped_clear_band"
            elif band == BAND_MID:
                previous = previous_reviews.get(key)
                if same_context and previous is not None and _assessment_unchanged(record, previous):
                    # 只复用数据、理由、轮次与审核上下文都没变的记录，防止旧结论套到新版本。
                    merged = deepcopy(previous)
                    status, issues = merged["audit_status"], merged["audit_issues"]
                    if status != "execution_error":
                        saved = previous_results.get(key)
                        if saved is None or saved["attempt"] != record["attempt"]:
                            raise ValueError("已保存的审核结论与评估轮次不一致，不能复用")
                        results.append(deepcopy(saved))
                else:
                    try:
                        result = self.agent.audit(record, **context)
                    except (AuditValidationError, LLMResponseError) as exc:
                        # 单条非法回答保留为执行错误；服务错误继续交给现有 CLI 处理。
                        status, issues = "execution_error", []
                        merged["audit_error"] = {
                            "type": type(exc).__name__,
                            "message": "审核回答连续两次未通过校验，需要重新执行审核。",
                        }
                    else:
                        results.append(deepcopy(result))
                        history.append({
                            "assessment": deepcopy(record),
                            "result": deepcopy(result),
                            "context": deepcopy(context),
                        })
                        issues = result["issues"]
                        status = "passed" if result["verdict"] == "pass" else "needs_resolution"
            else:
                raise ValueError(f"审核记录使用了未知档位「{band}」")

            merged["audit_status"] = status
            merged["audit_issues"] = deepcopy(issues)
            audited.append(merged)
            if status in ("needs_resolution", "execution_error"):
                pending.append(_pending_task(merged))
            elif merged.get("stance") == STANCE_RECOMMEND:
                candidates.append(deepcopy(merged))

        has_rewrites = any(task["action"] == "rewrite_reason" for task in pending)
        if has_rewrites:
            next_step = {
                "name": "prepare_reason_rewrites",
                "display_name": "按审核意见重写理由",
                "status": "ready",
                "implemented": True,
                "required_state": ["evaluated_talents", "audit_pending"],
                "message": "仅重写有可修正问题且未到轮次上限的中间档达人。",
            }
        elif pending:
            needs_lookup = any(task["action"] == "supplement_data" for task in pending)
            next_step = {
                "name": "supplement_data" if needs_lookup else "finalize_audit",
                "display_name": "查回缺失字段并返回评估" if needs_lookup else "处理缺数标记与审核交接",
                "status": "ready",
                "implemented": True,
                "required_state": ["audited_talents", "audit_pending", "active_policy"],
                "message": "自动重写结束，接下来核对已加载输入；单纯补数失败时标记放行，未解决的单条记录暂存，其余候选继续。",
            }
        else:
            next_step = {
                "name": "build_portfolio",
                "display_name": "投放组合，拟定价格",
                "status": "interface_ready" if candidates else "no_candidates",
                "implemented": False,
                "required_state": ["pricing_candidates", "audited_talents", "audit_pending", "supplement_results",
                                   "brief", "requirement_profile", "active_policy"],
                "message": "投放组合尚未实现。" if candidates else "没有已就绪的推荐候选，需要先由顾问确认后续处理。",
            }
        return {
            "audit_results": results,
            "audited_talents": audited,
            "audit_pending": pending,
            "pricing_candidates": candidates,
            "audit_context": context,
            "audit_history": history,
            "workflow_status": "audit_rewrite_required" if has_rewrites else ("audit_needs_resolution" if pending else "audit_complete"),
            "next_step": next_step,
        }


def _pending_task(record: dict[str, Any]) -> dict[str, Any]:
    """其他审核错误先回评估纠正；单纯缺数才补数，轮次耗尽不新增人工节点。"""
    issues = record["audit_issues"]
    fields = list(dict.fromkeys(
        field for issue in issues if issue["code"] == "data_missing" for field in issue["fields"]
    ))
    if record["audit_status"] == "execution_error":
        action = "retry_audit"
    elif record.get("supplement_review_reason"):
        action = "review_supplement_data"
    elif any(issue["code"] != "data_missing" for issue in issues):
        action = "rewrite_limit_reached" if record["attempt"] >= MAX_REASON_ATTEMPTS else "rewrite_reason"
    elif fields:
        action = "supplement_data"
    else:
        raise ValueError("审核未通过但没有可处理的具体问题")
    task = {
        "xhs_id": record.get("xhs_id"),
        "nickname": record.get("nickname"),
        "attempt": record["attempt"],
        "action": action,
        "fields": fields,
        "issues": deepcopy(issues),
    }
    if record.get("audit_error"):
        task["error"] = deepcopy(record["audit_error"])
    return task


def _assessment_unchanged(record: dict[str, Any], previous: dict[str, Any]) -> bool:
    assessment = {
        name: value for name, value in previous.items()
        if name not in {"audit_status", "audit_issues", "audit_error"}
    }
    return record == assessment
