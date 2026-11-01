from __future__ import annotations

from agent.audit_rules import check_reason_rules
from agent.reasons import person_key


def audit_report_fields(state: dict) -> dict:
    rows = state.get("audited_talents") or []
    policy = state.get("active_policy")
    if policy is None:
        policy = state.get("approved_policy") or {}
    # 当前没有人工核实结果的输入接口，不能仅凭模型没有报错就标记这些事项已完成。
    checks = [
        {"description": description, "status": "unverified", "source": "policy.manual_checks"}
        for description in dict.fromkeys(policy.get("manual_checks") or [])
    ]
    reviewed = {
        person_key(item["assessment"]) for item in state.get("audit_history") or []
        if not check_reason_rules(item["assessment"])
        and not item["assessment"].get("supplement_review_reason")
    } & {person_key(row) for row in rows}
    held = {person_key(row) for row in rows if row.get("release_status") == "held_for_review"}
    return {
        "audit_unverified_checks": checks,
        "audit_summary": {
            "scope": "单人理由审校及策略必要数据检查；不代表账号调查或组合KPI验收完成",
            "entered_count": len(rows),
            "model_reviewed_count": len(reviewed),
            "skipped_reason_review_count": sum(row.get("band") in ("clear_yes", "clear_no") for row in rows),
            "usable_recommendations": len(state.get("pricing_candidates") or []),
            "rejected_count": sum(row.get("stance") == "reject" and person_key(row) not in held for row in rows),
            "held_count": len(held),
            "flagged_release_count": sum(row.get("release_status") == "flagged_release" for row in rows),
            "unverified_check_count": len(checks),
            "excluded_before_audit_count": len(state.get("excluded_talents") or []),
            "policy_changed_after_approval": bool(state.get("active_policy") is not None and
                state.get("approved_policy") != state["active_policy"]),
        },
    }
