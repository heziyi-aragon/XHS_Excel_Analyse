"""第七步：给一位中间档达人写推荐理由。

每次调用只看见这一位达人。校验失败在这一条里重试一次；
仍然失败就把错误写在这条结果上，不把同批其他人一起取消。
"""
from __future__ import annotations

from typing import Any

from agent.domain import FIELD_LABELS, format_metric
from agent.reasons import (
    MAX_SENTENCE_LENGTH,
    RECOMMEND_TAGS,
    REJECT_TAGS,
    STANCE_RECOMMEND,
    ReasonValidationError,
    evidence_from_parts,
    normalize_stance,
    tags_for_stance,
)
from agent.requirements import LLMClient, LLMResponseError


class EvaluationAgent:
    """用闭集标签解释中间档：高分项和低分项同时存在，所以需要单独写一句。"""

    SYSTEM_PROMPT = f"""你是小红书达人投放的评估顾问。现在只评价一位已经落在中间档的达人。
这个人的高分项和低分项同时存在，所以代码没有直接下结论。请决定更应该推荐还是不推荐，并给出原因。

规则：
- stance 只能是 "recommend" 或 "reject"。
- stance 为 recommend 时，tag 必须从推荐标签中原样选择一个：{"、".join(RECOMMEND_TAGS)}。
- stance 为 reject 时，tag 必须从不推荐标签中原样选择一个：{"、".join(REJECT_TAGS)}。
- 不得发明清单外的标签，不得改写标签用字。
- sentence 用一句中文说明，不超过 {MAX_SENTENCE_LENGTH} 个字，必须同时点出一项优势和一项短板（数据缺失时改为说明缺了哪一项）。
- 句子里的数字只能来自本次给出的指标，不得编造，不得写成 ctr、cpuv 这类字段名。
- 看不到其他达人，不得和其他人比较，也不得声称已经完成客户 KPI。
- 关键评分指标缺失时，stance 用 reject，tag 用「数据不稳」。
- 只返回一个 JSON 对象，字段必须是 stance、tag、sentence。

参考结构，不要照抄内容：
{{"stance": "recommend", "tag": "进店好", "sentence": "进店中位数较高，但进店成本只是中等，先留在中间档。"}}
"""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def write_reason(self, payload: dict[str, Any]) -> dict[str, Any]:
        request: dict[str, Any] = {"candidate": _candidate_view(payload)}
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                result = self.llm.complete_json(self.SYSTEM_PROMPT, request)
                return _draft_from_model(result, payload)
            except (LLMResponseError, ReasonValidationError) as exc:
                last_error = exc
                if attempt == 0:
                    request["correction"] = {
                        "message": "上次输出未通过校验，请按闭集标签完整重写 JSON。",
                        "errors": getattr(exc, "errors", [str(exc)]),
                    }
        return _failed_draft(payload, str(last_error or "推荐理由没有通过校验"))


def _candidate_view(payload: dict[str, Any]) -> dict[str, Any]:
    """送给模型的视图只有中文指标名。字段名留在代码侧的 evidence 里。"""
    talent = payload.get("talent") if isinstance(payload.get("talent"), dict) else {}
    metrics = []
    for field, label in FIELD_LABELS.items():
        if field not in talent or talent.get(field) in (None, ""):
            continue
        metrics.append({"label": label, "value": format_metric(field, talent.get(field))})
    parts = []
    for part in payload.get("score_parts") or []:
        if not isinstance(part, dict):
            continue
        field = str(part.get("field") or "")
        parts.append(
            {
                "label": part.get("label") or FIELD_LABELS.get(field, field),
                "value": format_metric(field, part.get("raw")),
                "normalized": round(float(part.get("normalized") or 0), 3),
                "weight": part.get("weight"),
                "contribution": round(float(part.get("contribution") or 0), 3),
            }
        )
    policy = []
    for criterion in payload.get("policy_criteria") or []:
        if not isinstance(criterion, dict) or criterion.get("hard"):
            continue
        field = str(criterion.get("field") or "")
        operator = str(criterion.get("operator") or "")
        if operator == "maximize":
            direction = "越高越优先"
        elif operator == "minimize":
            direction = "越低越优先"
        else:
            direction = operator
        policy.append(
            {
                "label": FIELD_LABELS.get(field, field),
                "direction": direction,
                "weight": criterion.get("weight"),
                "reason": criterion.get("reason") or "",
            }
        )
    score = payload.get("score")
    return {
        "nickname": payload.get("nickname") or talent.get("nickname"),
        "score": None if score is None else round(float(score), 3),
        "metrics": metrics,
        "score_parts": parts,
        "missing": [
            FIELD_LABELS.get(str(field), str(field))
            for field in payload.get("missing_fields") or []
        ],
        "exclusions": list(payload.get("exclusions") or []),
        "kpi": payload.get("kpi") or "",
        "hard_requirements": list(payload.get("hard_requirements") or []),
        "compliance": payload.get("compliance") or "",
        "policy": policy,
    }


def _identity(payload: dict[str, Any]) -> tuple[Any, Any, int]:
    talent = payload.get("talent") if isinstance(payload.get("talent"), dict) else {}
    attempt = payload.get("attempt") or 1
    try:
        attempt_number = int(attempt)
    except (TypeError, ValueError):
        attempt_number = 1
    return (
        payload.get("xhs_id") or talent.get("xhs_id"),
        payload.get("nickname") or talent.get("nickname"),
        attempt_number,
    )


def _failed_draft(payload: dict[str, Any], message: str) -> dict[str, Any]:
    xhs_id, nickname, attempt = _identity(payload)
    return {
        "xhs_id": xhs_id,
        "nickname": nickname,
        "stance": None,
        "tag": None,
        "sentence": None,
        "evidence": evidence_from_parts(payload),
        "attempt": attempt,
        "error": message,
        "source": "model",
    }


def _draft_from_model(result: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    stance = normalize_stance(result.get("stance", result.get("倾向")))
    tag = str(result.get("tag") or result.get("原因标签") or "").strip()
    sentence = str(result.get("sentence") or result.get("总结说明") or "").strip()
    errors: list[str] = []
    if stance is None:
        errors.append("stance 只能是 recommend 或 reject")
    elif tag not in tags_for_stance(stance):
        errors.append(f"标签「{tag}」不在{('推荐' if stance == STANCE_RECOMMEND else '不推荐')}闭集中")
    if not sentence:
        errors.append("sentence 不能为空")
    elif len(sentence) > MAX_SENTENCE_LENGTH:
        errors.append(f"sentence 超过 {MAX_SENTENCE_LENGTH} 个字")
    leaked = _leaked_field_names(sentence)
    if leaked:
        errors.append("sentence 使用了字段名：" + "、".join(leaked))
    if errors:
        raise ReasonValidationError(errors)

    xhs_id, nickname, attempt = _identity(payload)
    return {
        "xhs_id": xhs_id,
        "nickname": nickname,
        "stance": stance,
        "tag": tag,
        "sentence": sentence,
        "evidence": evidence_from_parts(payload),
        "attempt": attempt,
        "error": None,
        "source": "model",
    }


def _leaked_field_names(sentence: str) -> list[str]:
    folded = sentence.casefold()
    leaked = []
    for field in FIELD_LABELS:
        if len(field) >= 3 and field.casefold() in folded:
            leaked.append(field)
    return leaked
