"""推荐 / 不推荐原因的闭集标签。

评估模型和打分节点共用这里的词表。模型只能原样返回其中的词。
「其它」在两边都有，所以每条结果必须另带 stance，审校才知道它站在哪一侧。
"""
from __future__ import annotations

from typing import Any


RECOMMEND_TAGS: tuple[str, ...] = (
    "数据好",
    "进店好",
    "内容契合",
    "人设契合",
    "粉丝匹配",
    "展示形式合适",
    "剧情植入适合",
    "阅读互动强",
    "题材匹配",
    "风格匹配",
    "其它",
)

REJECT_TAGS: tuple[str, ...] = (
    "阅读低",
    "数据差",
    "数据不稳",
    "内容不合适",
    "账号类型不合适",
    "机构号",
    "vlog为主",
    "粉丝不匹配",
    "价格贵",
    "更新少",
    "内容水",
    "商单表现差",
    "破圈账号不符",
    "美妆类不符",
    "彩妆号",
    "粉丝年龄大",
    "粉丝年龄小",
    "消费力弱",
    "人设不符",
    "内容杂乱",
    "其它",
)

BAND_CLEAR_YES = "clear_yes"
BAND_CLEAR_NO = "clear_no"
BAND_MID = "mid"

BAND_LABELS = {
    BAND_CLEAR_YES: "明确推荐",
    BAND_CLEAR_NO: "明确不推荐",
    BAND_MID: "中间档",
}

STANCE_RECOMMEND = "recommend"
STANCE_REJECT = "reject"

# 审校打回后最多再写一次。加上第一次，同一位达人最多进入扇出两轮。
MAX_REASON_ATTEMPTS = 2

# 旧原型把「总结说明」限制在 50 字。这里继续用这个上限，方便审校逐句核对。
MAX_SENTENCE_LENGTH = 50

_RECOMMEND_STANCES = {
    "recommend",
    "recommended",
    "yes",
    "true",
    "推荐",
    "值得推荐",
}
_REJECT_STANCES = {
    "reject",
    "rejected",
    "no",
    "false",
    "不推荐",
    "不值得推荐",
}

# 明确档由分数构成挑标签：哪一项主导总分，就用哪一条。
_RECOMMEND_BY_FIELD = {
    "ctr": "数据好",
    "cpv": "数据好",
    "cpe": "数据好",
    "exposure_med": "数据好",
    "read_med": "阅读互动强",
    "interact_med": "阅读互动强",
    "ratio_discover": "数据好",
    "ratio_search": "阅读互动强",
    "visit_store_med": "进店好",
    "cpuv": "进店好",
    "ratio_female": "粉丝匹配",
    "ratio_age_18_24": "粉丝匹配",
    "ratio_age_over_44": "粉丝匹配",
    "fans_wan": "粉丝匹配",
    "content_type": "内容契合",
    "main_format": "展示形式合适",
    "tier": "粉丝匹配",
    "price_image": "其它",
    "price_video": "其它",
    "ads_count_90d": "其它",
    "is_new_kol": "其它",
    "mcn": "其它",
}

_REJECT_BY_FIELD = {
    "read_med": "阅读低",
    "interact_med": "阅读低",
    "exposure_med": "阅读低",
    "ratio_discover": "阅读低",
    "ratio_search": "阅读低",
    "ctr": "数据差",
    "visit_store_med": "数据差",
    "cpv": "价格贵",
    "cpe": "价格贵",
    "cpuv": "价格贵",
    "price_image": "价格贵",
    "price_video": "价格贵",
    "ratio_female": "粉丝不匹配",
    "fans_wan": "粉丝不匹配",
    "content_type": "内容不合适",
    "main_format": "内容不合适",
    "tier": "账号类型不合适",
    "mcn": "机构号",
    "ads_count_90d": "更新少",
    "is_new_kol": "其它",
}


class ReasonValidationError(ValueError):
    """模型写的理由不在闭集里，或句子无法交给审校。"""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("；".join(errors))


def person_key(record: dict[str, Any]) -> str:
    """同一位达人的稳定键。表里常常没有小红书号，就退回昵称。"""
    return str(record.get("xhs_id") or record.get("nickname") or "")


def tags_for_stance(stance: str) -> tuple[str, ...]:
    if stance == STANCE_RECOMMEND:
        return RECOMMEND_TAGS
    if stance == STANCE_REJECT:
        return REJECT_TAGS
    return ()


def normalize_stance(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if text in _RECOMMEND_STANCES:
        return STANCE_RECOMMEND
    if text in _REJECT_STANCES:
        return STANCE_REJECT
    return None


def tag_for_score_part(part: dict[str, Any], stance: str) -> str:
    """按主导指标从闭集里挑标签。年龄方向相反时换成「粉丝年龄小」。"""
    field = str(part.get("field") or "")
    operator = str(part.get("operator") or "")
    if stance == STANCE_RECOMMEND:
        return _RECOMMEND_BY_FIELD.get(field, "其它")
    if field == "ratio_age_18_24" and operator == "minimize":
        return "粉丝年龄小"
    if field == "ratio_age_over_44" and operator == "maximize":
        return "粉丝年龄小"
    if field in {"ratio_age_18_24", "ratio_age_over_44"}:
        return "粉丝年龄大"
    return _REJECT_BY_FIELD.get(field, "其它")


def evidence_from_parts(payload: dict[str, Any]) -> dict[str, Any]:
    """证据从原始分数构成抄出，不采用模型复述的数字。"""
    evidence: dict[str, Any] = {}
    for part in payload.get("score_parts") or []:
        if isinstance(part, dict) and part.get("field"):
            evidence[str(part["field"])] = part.get("raw")
    for field in payload.get("missing_fields") or []:
        evidence.setdefault(str(field), None)
    return evidence
