"""第八步的输出约定。这里只校验结构，不调用模型，也不判断理由真假。

模型只返回 verdict 和 issues；达人身份、被审评估轮次由代码带入。
审核结果单独保存，不覆盖第七步的推荐倾向、标签、说明或指标。
"""
from __future__ import annotations

from typing import Any, Literal, TypedDict, cast

from agent.domain import FIELD_LABELS


AuditVerdict = Literal["pass", "reject"]
AuditIssueCode = Literal[
    "invalid_tag", "data_conflict", "client_exclusion", "data_missing", "generation_error"
]
AUDIT_ISSUE_CODES = frozenset(
    {"invalid_tag", "data_conflict", "client_exclusion", "data_missing", "generation_error"}
)


class AuditIssue(TypedDict):
    code: AuditIssueCode
    message: str
    fields: list[str]


class AuditResult(TypedDict):
    xhs_id: str | None
    nickname: str | None
    attempt: int  # 被审核的评估轮次，不是审核 API 的重试次数
    verdict: AuditVerdict
    issues: list[AuditIssue]


class AuditValidationError(ValueError):
    """审核回答结构不合法；这不是对达人或其理由的业务打回。"""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("；".join(errors))


def parse_audit_result(
    response: dict[str, Any], record: dict[str, Any]
) -> AuditResult:
    """严格校验审核回答，再附加来自上游的身份与轮次。两个输入均不修改。

    结构合法不代表审核判断正确。事实核对、必要缺失项检查和路由由后续实现。
    """
    errors: list[str] = []
    if not isinstance(response, dict):
        raise AuditValidationError(["审核回答必须是 JSON 对象"])
    if set(response) != {"verdict", "issues"}:
        errors.append("审核回答只能包含 verdict 和 issues，且两个字段都必须提供")

    verdict = response.get("verdict")
    if verdict not in ("pass", "reject"):
        errors.append("verdict 只能是 pass 或 reject")

    issues: list[AuditIssue] = []
    raw_issues = response.get("issues")
    if not isinstance(raw_issues, list):
        errors.append("issues 必须是数组")
    else:
        for index, item in enumerate(raw_issues, start=1):
            prefix = f"第 {index} 条审核问题"
            if not isinstance(item, dict):
                errors.append(f"{prefix}必须是对象")
                continue
            issue_error_count = len(errors)
            if set(item) != {"code", "message", "fields"}:
                errors.append(f"{prefix}必须且只能包含 code、message、fields")
            code = item.get("code")
            if not isinstance(code, str) or code not in AUDIT_ISSUE_CODES:
                errors.append(f"{prefix}使用了不支持的问题类型")
            message = item.get("message")
            if not isinstance(message, str) or not message.strip():
                errors.append(f"{prefix}必须提供非空的具体说明")
            fields = item.get("fields")
            if not isinstance(fields, list) or any(
                not isinstance(name, str) or name not in FIELD_LABELS for name in fields
            ):
                errors.append(f"{prefix}的 fields 必须是现有达人指标字段名组成的数组")
            elif code == "data_missing" and not fields:
                errors.append(f"{prefix}属于数据缺失，必须指出需要补充的字段")
            # 只有全部检查通过才会返回结果；这里复制列表，避免与模型回答共用可变对象。
            if len(errors) == issue_error_count:
                issues.append(
                    {
                        "code": cast(AuditIssueCode, code),
                        "message": message.strip(),
                        "fields": list(fields),
                    }
                )

        if verdict == "pass" and raw_issues:
            errors.append("审核通过时 issues 必须为空")
        if verdict == "reject" and not raw_issues:
            errors.append("审核打回时至少需要一条具体问题")

    xhs_id = record.get("xhs_id")
    nickname = record.get("nickname")
    for name, value in (("xhs_id", xhs_id), ("nickname", nickname)):
        if value is not None and not isinstance(value, str):
            errors.append(f"上游 {name} 必须是字符串或 None")
    if not (
        (isinstance(xhs_id, str) and xhs_id.strip())
        or (isinstance(nickname, str) and nickname.strip())
    ):
        errors.append("上游记录缺少可用于匹配达人的小红书号和昵称")
    attempt = record.get("attempt")
    if type(attempt) is not int or attempt < 1:
        errors.append("上游 attempt 必须是正整数，表示被审核的评估轮次")

    if errors:
        raise AuditValidationError(errors)
    return {
        "xhs_id": xhs_id,
        "nickname": nickname,
        "attempt": attempt,
        "verdict": cast(AuditVerdict, verdict),
        "issues": issues,
    }
