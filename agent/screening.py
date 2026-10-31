"""第四步到第六步：清洗、硬性筛选、去重、打分分档。

这些步骤规则可写、结果可核对，所以全部是普通函数，不调用模型。
明确档的标签也在这里从分数构成里挑出；中间档只标记档位，理由留给扇出节点。
"""
from __future__ import annotations

import copy
import re
from typing import Any

from agent.domain import FIELD_LABELS, format_metric
from agent.reasons import (
    BAND_CLEAR_NO,
    BAND_CLEAR_YES,
    BAND_MID,
    STANCE_RECOMMEND,
    STANCE_REJECT,
    tag_for_score_part,
)

# 清洗后少于这个人数，就放宽数值硬条件再筛。和加载阶段「少于 5 人」的提示一致。
MIN_CANDIDATES = 5
MAX_RELAX_ROUNDS = 3
RELAX_STEP = 0.15

# 屿光样例里，进店权重 0.6、进店成本权重 0.4 时，
# 强账号大约在 0.75 以上，弱账号在 0.35 以下，中间才需要模型写理由。
CLEAR_YES_MIN = 0.72
CLEAR_NO_MAX = 0.35

# 声量文化的两位差在 0.02 以内；启明星的两位年龄占比差 0.11。0.05 能分开这两种情况。
FAN_OVERLAP_MAX_DELTA = 0.05
_FAN_FIELDS = ("ratio_female", "ratio_age_18_24", "ratio_age_over_44")

_TEXT_FIELDS = (
    "nickname",
    "xhs_id",
    "mcn",
    "is_new_kol",
    "content_type",
    "tier",
    "main_format",
)
# 机构可以合法为空（未签约达人），不把 mcn 算成缺失。
_TRACKED_FIELDS = (
    "main_format",
    "content_type",
    "tier",
    "price_image",
    "price_video",
    "fans_wan",
    "ratio_age_18_24",
    "ratio_age_over_44",
    "ratio_female",
    "ctr",
    "cpv",
    "cpe",
    "cpuv",
    "exposure_med",
    "read_med",
    "interact_med",
    "visit_store_med",
    "ratio_discover",
    "ratio_search",
    "ads_count_90d",
)
_NUMERIC_OPERATORS = frozenset({"gt", "gte", "lt", "lte"})


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def clean_text(value: Any) -> str | None:
    """去掉首尾空白，并把中文逗号收成英文逗号。空串视为缺失。"""
    if value is None:
        return None
    text = str(value).replace("\u3000", " ").replace("，", ",").strip()
    text = re.sub(r"[ \t]+", " ", text)
    return text or None


def clean_talent(raw: dict[str, Any]) -> dict[str, Any]:
    """复制一份达人记录，规范化文本并标出缺失字段。不改原表。"""
    talent = dict(raw)
    for key in _TEXT_FIELDS:
        if key in talent:
            talent[key] = clean_text(talent.get(key))
    talent["missing_fields"] = [
        key for key in _TRACKED_FIELDS if talent.get(key) in (None, "")
    ]
    return talent


def clean_talents(talents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [clean_talent(talent) for talent in talents]


_VALUE_ALIASES: dict[str, tuple[str, ...]] = {
    "beauty": ("beauty", "美妆", "护肤", "美妆护肤"),
    "美妆": ("beauty", "美妆", "护肤", "美妆护肤"),
    "护肤": ("beauty", "美妆", "护肤", "美妆护肤"),
    "美妆护肤": ("beauty", "美妆", "护肤", "美妆护肤"),
    "lifestyle": ("lifestyle", "生活", "生活方式"),
    "生活": ("lifestyle", "生活", "生活方式"),
    "生活方式": ("lifestyle", "生活", "生活方式"),
    "fashion": ("fashion", "时尚", "穿搭"),
    "时尚": ("fashion", "时尚", "穿搭"),
    "穿搭": ("fashion", "时尚", "穿搭"),
    "mother&baby": ("mother&baby", "母婴", "亲子"),
    "母婴": ("mother&baby", "母婴", "亲子"),
    "food": ("food", "美食"),
    "美食": ("food", "美食"),
    "y": ("y", "yes", "true", "是"),
    "是": ("y", "yes", "true", "是"),
    "n": ("n", "no", "false", "否"),
    "否": ("n", "no", "false", "否"),
}


def _candidate_forms(text: Any) -> tuple[str, ...]:
    folded = str(text or "").strip().casefold()
    if not folded:
        return ("",)
    return _VALUE_ALIASES.get(folded, (folded,))


def _same(raw: Any, value: Any) -> bool:
    if _is_number(raw) and _is_number(value):
        return float(raw) == float(value)
    raw_forms = set(_candidate_forms(raw))
    val_forms = set(_candidate_forms(value))
    return bool(raw_forms & val_forms)


def _contains(raw: Any, value: Any) -> bool:
    if raw is None or value is None:
        return False
    raw_forms = _candidate_forms(raw)
    val_forms = _candidate_forms(value)
    return any(vf and vf in rf for rf in raw_forms for vf in val_forms)


def _is_in(raw: Any, value: Any) -> bool:
    options = value if isinstance(value, (list, tuple, set)) else [value]
    return any(_same(raw, option) for option in options)


def criterion_passes(talent: dict[str, Any], criterion: dict[str, Any]) -> bool:
    """硬条件的 operator 是「通过」条件。maximize / minimize 不是门槛。"""
    operator = str(criterion.get("operator") or "")
    raw = talent.get(str(criterion.get("field") or ""))
    value = criterion.get("value")
    if operator == "is_missing":
        return raw in (None, "")
    if operator == "not_missing":
        return raw not in (None, "")
    if operator in _NUMERIC_OPERATORS:
        if not _is_number(raw) or not _is_number(value):
            return False
        left, right = float(raw), float(value)
        if operator == "gt":
            return left > right
        if operator == "gte":
            return left >= right
        if operator == "lt":
            return left < right
        return left <= right
    if operator == "eq":
        return _same(raw, value)
    if operator == "ne":
        return not _same(raw, value)
    if operator == "contains":
        return _contains(raw, value)
    if operator == "not_contains":
        return not _contains(raw, value)
    if operator == "in":
        return _is_in(raw, value)
    if operator == "not_in":
        return not _is_in(raw, value)
    if operator in {"maximize", "minimize"}:
        return True
    return False


def _failure_text(criterion: dict[str, Any]) -> str:
    field = FIELD_LABELS.get(str(criterion.get("field") or ""), str(criterion.get("field") or "指标"))
    operator = str(criterion.get("operator") or "")
    value = criterion.get("value")
    shown = "、".join(str(item) for item in value) if isinstance(value, (list, tuple)) else value
    texts = {
        "not_contains": f"{field}包含「{shown}」",
        "contains": f"{field}不包含「{shown}」",
        "eq": f"{field}不是「{shown}」",
        "ne": f"{field}等于「{shown}」",
        "gt": f"{field}未高于 {shown}",
        "gte": f"{field}低于 {shown}",
        "lt": f"{field}未低于 {shown}",
        "lte": f"{field}高于 {shown}",
        "in": f"{field}不属于 {shown}",
        "not_in": f"{field}属于 {shown}",
        "is_missing": f"{field}有数据",
        "not_missing": f"{field}缺失",
    }
    return texts.get(operator, f"{field}未满足 {operator} {shown}")


def apply_hard_filters(
    talents: list[dict[str, Any]], policy: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """只执行 hard 条件。软性权重留到打分，不在这里淘汰人。"""
    hard = [item for item in policy.get("criteria") or [] if item.get("hard")]
    kept: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for talent in talents:
        reasons = [
            _failure_text(criterion)
            for criterion in hard
            if not criterion_passes(talent, criterion)
        ]
        if reasons:
            excluded.append(
                {
                    "nickname": talent.get("nickname"),
                    "xhs_id": talent.get("xhs_id"),
                    "reason": "；".join(reasons),
                }
            )
        else:
            kept.append(talent)
    return kept, excluded


def _fans_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if any(left.get(key) is None or right.get(key) is None for key in _FAN_FIELDS):
        return False
    return all(
        abs(float(left[key]) - float(right[key])) <= FAN_OVERLAP_MAX_DELTA
        for key in _FAN_FIELDS
    )


def dedupe_by_agency_fans(
    talents: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """同一机构里粉丝构成几乎一样的账号只留一个。

    保留粉丝更多的；粉丝相同则保留缺失更少的，再按昵称稳定排序。
    没有机构的达人不参与这组比较。
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    independents: list[dict[str, Any]] = []
    for talent in talents:
        agency = talent.get("mcn")
        if not agency:
            independents.append(talent)
            continue
        grouped.setdefault(str(agency), []).append(talent)

    kept = list(independents)
    dropped: list[dict[str, Any]] = []
    for agency, rows in grouped.items():
        rows = sorted(
            rows,
            key=lambda item: (
                -(item.get("fans_wan") if _is_number(item.get("fans_wan")) else -1),
                len(item.get("missing_fields") or []),
                str(item.get("nickname") or ""),
            ),
        )
        chosen: list[dict[str, Any]] = []
        for talent in rows:
            overlap = next((item for item in chosen if _fans_overlap(talent, item)), None)
            if overlap is None:
                chosen.append(talent)
                continue
            dropped.append(
                {
                    "nickname": talent.get("nickname"),
                    "xhs_id": talent.get("xhs_id"),
                    "mcn": agency,
                    "kept": overlap.get("nickname"),
                    "kept_xhs_id": overlap.get("xhs_id"),
                }
            )
        kept.extend(chosen)
    return dropped, kept


def _soft_criteria(policy: dict[str, Any]) -> list[dict[str, Any]]:
    criteria = []
    for item in policy.get("criteria") or []:
        if item.get("hard"):
            continue
        if not _is_number(item.get("weight")) or float(item["weight"]) <= 0:
            continue
        criteria.append(item)
    return criteria


def _normalize(
    talent: dict[str, Any],
    criterion: dict[str, Any],
    ranges: dict[tuple[str, str], tuple[float, float]],
) -> float | None:
    operator = str(criterion.get("operator") or "")
    field = str(criterion.get("field") or "")
    raw = talent.get(field)
    if operator in {"maximize", "minimize"}:
        if not _is_number(raw):
            return None
        low, high = ranges.get((field, operator), (None, None))
        if low is None or high is None:
            return None
        if high == low:
            scaled = 1.0
        else:
            scaled = (float(raw) - low) / (high - low)
        return 1.0 - scaled if operator == "minimize" else scaled
    if raw in (None, ""):
        return None
    return 1.0 if criterion_passes(talent, criterion) else 0.0


def _band_for(score: float | None) -> str:
    if score is None:
        return BAND_MID
    if score >= CLEAR_YES_MIN:
        return BAND_CLEAR_YES
    if score <= CLEAR_NO_MAX:
        return BAND_CLEAR_NO
    return BAND_MID


def _dominant_part(parts: list[dict[str, Any]], stance: str) -> dict[str, Any]:
    if stance == STANCE_RECOMMEND:
        return max(parts, key=lambda part: (part["contribution"], part["weight"]))
    return max(
        parts,
        key=lambda part: (
            part["weight_effective"] * (1.0 - part["normalized"]),
            part["weight"],
        ),
    )


def _code_sentence(part: dict[str, Any], stance: str, tag: str) -> str:
    value = format_metric(str(part["field"]), part.get("raw"))
    label = str(part.get("label") or part["field"])
    if stance == STANCE_RECOMMEND:
        return f"{label}为{value}，是拉高总分的主要项，记为「{tag}」。"
    return f"{label}为{value}，是拖低总分的主要项，记为「{tag}」。"


def score_candidates(
    talents: list[dict[str, Any]], policy: dict[str, Any]
) -> list[dict[str, Any]]:
    """在筛选后的候选人内部做 min-max，再按权重加总。

    某位达人缺了被加权的字段时，用他剩下的指标重新归一权重，
    不把缺失当成 0 分。全部加权字段都缺失时分数为 None，档位落在中间档，
    交给模型说明数据不全，而不是假装分数已经明确。
    """
    soft = _soft_criteria(policy)
    ranges: dict[tuple[str, str], tuple[float, float]] = {}
    for criterion in soft:
        operator = str(criterion.get("operator") or "")
        field = str(criterion.get("field") or "")
        if operator not in {"maximize", "minimize"}:
            continue
        values = [
            float(talent[field])
            for talent in talents
            if _is_number(talent.get(field))
        ]
        if values:
            ranges[(field, operator)] = (min(values), max(values))

    scored: list[dict[str, Any]] = []
    for talent in talents:
        parts: list[dict[str, Any]] = []
        for criterion in soft:
            normalized = _normalize(talent, criterion, ranges)
            if normalized is None:
                continue
            field = str(criterion.get("field") or "")
            parts.append(
                {
                    "field": field,
                    "operator": str(criterion.get("operator") or ""),
                    "label": FIELD_LABELS.get(field, field),
                    "raw": talent.get(field),
                    "normalized": normalized,
                    "weight": float(criterion["weight"]),
                }
            )
        weight_sum = sum(part["weight"] for part in parts)
        if weight_sum <= 0:
            score = None
            for part in parts:
                part["weight_effective"] = 0.0
                part["contribution"] = 0.0
        else:
            for part in parts:
                part["weight_effective"] = part["weight"] / weight_sum
                part["contribution"] = part["normalized"] * part["weight_effective"]
            score = sum(part["contribution"] for part in parts)

        band = _band_for(score)
        record: dict[str, Any] = {
            "xhs_id": talent.get("xhs_id"),
            "nickname": talent.get("nickname"),
            "band": band,
            "score": score,
            "score_parts": parts,
            "missing_fields": list(talent.get("missing_fields") or []),
            "talent": talent,
            "attempt": 1,
            "stance": None,
            "tag": None,
            "sentence": None,
            "evidence": {part["field"]: part.get("raw") for part in parts},
            "error": None,
            "source": None,
        }
        if band in {BAND_CLEAR_YES, BAND_CLEAR_NO} and parts:
            stance = STANCE_RECOMMEND if band == BAND_CLEAR_YES else STANCE_REJECT
            part = _dominant_part(parts, stance)
            tag = tag_for_score_part(part, stance)
            record["stance"] = stance
            record["tag"] = tag
            record["sentence"] = _code_sentence(part, stance, tag)
            record["source"] = "rules"
        scored.append(record)
    return scored


def _threshold_text(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _pass_phrase(field: str, operator: str, value: float) -> str:
    label = FIELD_LABELS.get(field, field)
    shown = _threshold_text(value)
    if operator == "gte":
        return f"{label}不低于 {shown}"
    if operator == "gt":
        return f"{label}高于 {shown}"
    if operator == "lte":
        return f"{label}不高于 {shown}"
    return f"{label}低于 {shown}"


def binding_numeric_index(policy: dict[str, Any], talents: list[dict[str, Any]]) -> int | None:
    """找出当前淘汰人数最多的那条数值硬条件。文字排除项不参与放宽。"""
    criteria = policy.get("criteria") or []
    best_index: int | None = None
    best_count = 0
    for index, criterion in enumerate(criteria):
        if not criterion.get("hard"):
            continue
        if str(criterion.get("operator") or "") not in _NUMERIC_OPERATORS:
            continue
        if not _is_number(criterion.get("value")):
            continue
        others = [
            item
            for other_index, item in enumerate(criteria)
            if item.get("hard") and other_index != index
        ]
        count = 0
        for talent in talents:
            if criterion_passes(talent, criterion):
                continue
            if all(criterion_passes(talent, other) for other in others):
                count += 1
        if count > best_count:
            best_count = count
            best_index = index
    return best_index


def can_relax(policy: dict[str, Any], talents: list[dict[str, Any]]) -> bool:
    return binding_numeric_index(policy, talents) is not None


def widen_threshold(operator: str, value: float) -> float:
    delta = abs(value) * RELAX_STEP
    if delta == 0:
        delta = 1.0
    if operator in {"gt", "gte"}:
        return round(value - delta, 4)
    return round(value + delta, 4)


def widen_policy(
    policy: dict[str, Any], talents: list[dict[str, Any]]
) -> tuple[dict[str, Any], str | None]:
    """把最紧的一条数值硬条件放宽 15%。返回新策略副本和说明；没有可放宽项时说明为 None。"""
    copied = copy.deepcopy(policy)
    index = binding_numeric_index(copied, talents)
    if index is None:
        return copied, None
    criterion = copied["criteria"][index]
    operator = str(criterion["operator"])
    before = float(criterion["value"])
    after = widen_threshold(operator, before)
    criterion["value"] = after
    field = str(criterion.get("field") or "")
    message = (
        f"候选人数不足 {MIN_CANDIDATES} 人，将「{_pass_phrase(field, operator, before)}」"
        f"放宽为「{_pass_phrase(field, operator, after)}」。"
    )
    return copied, message
