"""第四步到第六步的图节点：只搬运状态，规则在 agent.screening。"""
from __future__ import annotations

import copy
from typing import Any

from agent.screening import (
    MAX_RELAX_ROUNDS,
    MIN_CANDIDATES,
    can_relax,
    clean_talents,
    dedupe_by_agency_fans,
    apply_hard_filters,
    score_candidates,
    widen_policy,
)


def preprocess_table(state: dict[str, Any]) -> dict[str, Any]:
    """清洗文本、标出缺失字段。原表 talents 保持不动。"""
    cleaned = clean_talents(state.get("talents") or [])
    missing_people = [item for item in cleaned if item.get("missing_fields")]
    both_prices = [
        str(item.get("nickname"))
        for item in cleaned
        if item.get("price_image") is None and item.get("price_video") is None
    ]
    issues = list(state.get("issues") or [])
    issues.append(
        f"清洗完成 {len(cleaned)} 人，其中 {len(missing_people)} 人有缺失字段。"
    )
    if both_prices:
        issues.append("图文和视频参考价都缺失：" + "、".join(both_prices))
    return {
        "cleaned_talents": cleaned,
        "issues": issues,
        "workflow_status": "screening",
    }


def screen_talents(state: dict[str, Any]) -> dict[str, Any]:
    """按当前硬条件筛选，再去掉同机构且粉丝构成几乎一样的账号。"""
    source = state.get("active_policy") or state.get("approved_policy")
    if not isinstance(source, dict):
        raise ValueError("缺少已批准的筛选策略，不能筛选达人")
    policy = copy.deepcopy(source)
    kept, excluded = apply_hard_filters(state.get("cleaned_talents") or [], policy)
    deduped, kept = dedupe_by_agency_fans(kept)

    log = list(state.get("screen_log") or [])
    log.append(
        f"本轮保留 {len(kept)} 人，排除 {len(excluded)} 人，去重 {len(deduped)} 人。"
    )
    for item in excluded:
        log.append(f"排除「{item.get('nickname')}」：{item.get('reason')}")
    for item in deduped:
        log.append(
            f"去重去掉「{item.get('nickname')}」，保留同机构粉丝构成接近的「{item.get('kept')}」。"
        )

    issues = list(state.get("issues") or [])
    issues.append(
        f"筛选后保留 {len(kept)} 人，排除 {len(excluded)} 人，去重 {len(deduped)} 人。"
    )
    return {
        "active_policy": policy,
        "screened_talents": kept,
        "excluded_talents": excluded,
        "deduped_talents": deduped,
        "screen_log": log,
        "issues": issues,
        "workflow_status": "screening",
    }


def relax_screening(state: dict[str, Any]) -> dict[str, Any]:
    """放宽一条数值硬条件。客户写明的文字排除项保持原样。"""
    policy = state.get("active_policy") or state.get("approved_policy") or {}
    widened, message = widen_policy(policy, state.get("cleaned_talents") or [])
    log = list(state.get("screen_log") or [])
    log.append(message or "没有可以放宽的数值硬条件，文字排除项保持不变。")
    return {
        "active_policy": widened,
        "screen_relax_count": int(state.get("screen_relax_count") or 0) + 1,
        "screen_log": log,
        "workflow_status": "screening",
    }


def route_after_screen(state: dict[str, Any]) -> str:
    """人数够了，或已经放宽 3 次，或只剩不能放宽的排除项时，进入打分。"""
    if len(state.get("screened_talents") or []) >= MIN_CANDIDATES:
        return "score"
    if int(state.get("screen_relax_count") or 0) >= MAX_RELAX_ROUNDS:
        return "score"
    policy = state.get("active_policy") or state.get("approved_policy") or {}
    if not can_relax(policy, state.get("cleaned_talents") or []):
        return "score"
    return "relax"


def score_talents(state: dict[str, Any]) -> dict[str, Any]:
    """打分分档。明确档直接写好标签；中间档放进 reason_queue。"""
    policy = state.get("active_policy") or state.get("approved_policy") or {}
    scored = score_candidates(state.get("screened_talents") or [], policy)
    queue = [
        {
            "xhs_id": item.get("xhs_id"),
            "nickname": item.get("nickname"),
            "band": item.get("band"),
            "score": item.get("score"),
            "score_parts": item.get("score_parts"),
            "missing_fields": item.get("missing_fields"),
            "talent": item.get("talent"),
            "attempt": int(item.get("attempt") or 1),
        }
        for item in scored
        if item.get("band") == "mid"
    ]
    return {
        "scored_talents": scored,
        "reason_queue": queue,
        "workflow_status": "scored",
    }
