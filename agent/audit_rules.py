"""第八步的确定性检查：收集问题，不调用 API，不作完整审核通过结论。

理由规则供中间档使用；必要数据检查独立调用，覆盖明确档与中间档。
后续审核节点负责结合这些问题、模型核对结果与补数状态决定路由。
"""
from __future__ import annotations

from typing import Any

from agent.audit_contract import AuditIssue
from agent.domain import FIELD_LABELS
from agent.reasons import STANCE_RECOMMEND, STANCE_REJECT, tags_for_stance


def check_reason_rules(record: dict[str, Any]) -> list[AuditIssue]:
    """检查中间档是否有可审核的理由，以及标签是否属于对应清单。

    空列表只表示这些代码规则没有发现问题；不能据此给出 verdict=pass。
    是否违反客户排除项、理由是否与指标相符，留给后续独立核对。
    """
    if record.get("error"):
        return [{
            "code": "generation_error",
            "message": "上游记录了理由生成错误，需要先重新生成理由。",
            "fields": [],
        }]

    stance = record.get("stance")
    sentence = record.get("sentence")
    if stance not in (STANCE_RECOMMEND, STANCE_REJECT):
        return [{
            "code": "generation_error",
            "message": "上游没有提供有效的推荐倾向，无法审核这条理由。",
            "fields": [],
        }]
    if not isinstance(sentence, str) or not sentence.strip():
        return [{
            "code": "generation_error",
            "message": "上游没有提供非空的理由说明，无法进行事实核对。",
            "fields": [],
        }]

    tag = record.get("tag")
    # 直接使用评估模块共用的词表，不能另抄一份，更不能只检查两侧词表的并集。
    if tag not in tags_for_stance(stance):
        side = "推荐" if stance == STANCE_RECOMMEND else "不推荐"
        return [{
            "code": "invalid_tag",
            "message": f"标签「{tag}」不在{side}原因清单内，需要评估模块重新选择。",
            "fields": [],
        }]
    return []


def check_required_data(
    record: dict[str, Any],
    policy: dict[str, Any],
    *,
    additional_fields: tuple[str, ...] = (),
) -> list[AuditIssue]:
    """检查实际策略使用的指标及调用方明确追加的字段，适用于所有档位。

    policy 应传 active_policy，没有放宽时可用 approved_policy。
    不从 score_parts 推断必需字段：打分代码会跳过缺失项，那里可能已无该字段。
    不照抄 missing_fields：其中可能包含当前任务不需要的数据，也可能尚未更新。
    additional_fields 供后续核对理由、投放组合时明确补充所需字段，不默认要求双报价。
    """
    talent = record.get("talent")
    if not isinstance(talent, dict):
        # 这是输入结构损坏，不能伪装成某位达人的全部业务指标都缺失。
        raise ValueError("必要数据检查需要上游 record.talent 为字典")

    required: list[str] = []
    for criterion in policy.get("criteria") or []:
        field = criterion["field"]
        # 昵称/账号不是可补指标；MCN 为空可合法表示未签约，不能自动当成缺数。
        if field not in FIELD_LABELS or field == "mcn":
            continue
        # 策略明确寻找空值时，空值本身就是条件，不应反过来要求补齐。
        if criterion.get("operator") == "is_missing":
            continue
        required.append(field)

    for field in additional_fields:
        if field not in FIELD_LABELS:
            raise ValueError(f"额外必需字段「{field}」不在已有达人指标中")
        required.append(field)

    missing: list[str] = []
    for field in dict.fromkeys(required):  # 保持策略顺序并去重，便于补数工具逐字段处理
        value = talent.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            missing.append(field)
    if not missing:
        return []

    labels = "、".join(FIELD_LABELS[field] for field in missing)
    return [{
        "code": "data_missing",
        "message": f"缺少当前判断所需的{labels}，需要补充对应字段后再核对。",
        "fields": missing,
    }]
