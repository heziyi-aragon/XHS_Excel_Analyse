"""第十步：文案建议 Agent 的内容建议生成、输出校验和结果关联。

复用现有 LLMClient，不单独管理密钥、模型、网络或 JSON 解析。
仅创建实例不会调用 API；调用 call_llm 才会发起模型请求。
"""
from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from copy import deepcopy
from typing import Any

from agent.requirements import LLMClient, LLMResponseError, LLMServiceError


# 当前项目暂定分类，并非对历史 OLAY 表格「5大部分」的还原。
# 后续组织提示词及校验模型输出时，共用此配置。
ANGLES: dict[str, list[str]] = {
    "单品种草": ["配方卖点", "质地展示"],
    "合集种草": ["护肤搭配", "护肤流程"],
    "场景种草": ["换季护肤", "通勤护肤"],
    "知识科普": ["成分知识", "护肤误区"],
    "大促种草": ["双十一清单", "双十二选购"],
}


def quote_remarks(talent: dict[str, Any]) -> list[str]:
    """按当前达人原始报价生成备注，不使用模型判断。"""
    remarks = []
    for field, label in (("price_image", "图文"), ("price_video", "视频")):
        raw = talent.get(field)
        try:
            value = Decimal(str(raw))
            available = not isinstance(raw, bool) and value.is_finite() and value > 0
        except (InvalidOperation, ValueError):
            available = False
        if not available:
            remarks.append(f"未提供有效{label}报价；如需{label}合作，请先补充报价。")
    return remarks


def without_quote_missing_warnings(warnings: list[str]) -> list[str]:
    """缺报价归备注；只移除报价缺失提示，保留其他风险句。"""
    cleaned = []
    for warning in warnings:
        # 兼容第9步现有英文消息，以及模型可能给出的中文复述。
        warning = re.sub(
            r"(?:id:)?[\w-]+: price_(?:image|video) is unavailable; this format cannot be budgeted\.",
            "", warning)
        remaining = []
        for clause in re.split(r"(?<=[。；])|\n", warning):
            quote = re.search(r"price_(?:image|video)|(?:图文|视频)(?:合作)?报价", clause)
            missing = re.search(r"缺少|缺失|未提供|未填写|没有|为空|不可用|unavailable|missing", clause, re.I)
            if quote and missing:
                continue
            if clause.strip():
                remaining.append(clause.strip())
        text = "".join(remaining).strip()
        if text:
            cleaned.append(text)
    return list(dict.fromkeys(cleaned))


def without_clear_band_warnings(warnings: list[str], audit_status: Any) -> list[str]:
    """明确档跳过理由审校是流程状态，由Python移除其说明。"""
    if audit_status != "skipped_clear_band":
        return list(warnings)
    cleaned = []
    for warning in warnings:
        remaining = []
        skipped_explanation = False
        for clause in re.split(r"(?<=[。；;，,])|\n", warning):
            if re.search(r"skipped[ _-]*clear[ _-]*band|跳过.*(?:理由|模型).*审[核校]|无需.*理由.*审[核校]", clause, re.I):
                skipped_explanation = True
                continue
            # 单独保留的后半句仍是上述流程状态的解释。
            if skipped_explanation and re.search(r"(?:不代表|不等于|并非).*(?:完整业务审核|全部业务审核|完整审核)", clause):
                continue
            if clause.strip():
                remaining.append(clause.strip())
        text = "".join(remaining).strip("，,；; ")
        if text:
            cleaned.append(text)
    return list(dict.fromkeys(cleaned))


def without_pricing_date_warnings(warnings: list[str]) -> list[str]:
    """由Python排除定价基准日期缺失说明，保留相邻的价格风险。"""
    cleaned = []
    for warning in warnings:
        remaining = []
        for clause in re.split(r"(?<=[。；;，,])|\n", warning):
            date_field = re.search(r"pricing[_ ]as[_ ]of|定价基准日期|定价基准日", clause, re.I)
            missing = re.search(r"未提供|未设置|未指定|未填写|缺少|缺失|没有|为空|not (?:provided|supplied|set)|missing", clause, re.I)
            if date_field and missing:
                continue
            if clause.strip():
                remaining.append(clause.strip())
        text = "".join(remaining).strip("，,；; ")
        if text:
            cleaned.append(text)
    return list(dict.fromkeys(cleaned))


def normalize_portfolio_format(form: str) -> str:
    """将第九步的合作形式转换为写建议模块使用的值。"""
    mapping = {
        "image": "image_text",
        "video": "video",
    }
    if form not in mapping:
        raise ValueError(f"不支持的第九步合作形式：{form!r}")
    return mapping[form]


def portfolio_price_fields(line: dict[str, Any]) -> dict[str, Any]:
    """将第九步建议价格映射为写建议模块的字段。"""
    return {
        "unit_price": line["unit_price_suggested"],
        "total_price": line["line_cost_suggested"],
    }


def find_audited_record(
    xhs_id: str,
    audited_talents: list[dict[str, Any]],
) -> dict[str, Any]:
    """按达人 ID 唯一匹配审核记录，返回副本。"""
    if not isinstance(xhs_id, str) or not xhs_id.strip():
        raise ValueError("投放项缺少有效的 xhs_id")

    matches = [
        record
        for record in audited_talents
        if record.get("xhs_id") == xhs_id
    ]
    if len(matches) != 1:
        raise ValueError(f"达人 {xhs_id!r} 无法唯一匹配审核记录")

    return deepcopy(matches[0])


def build_advice_placement(
    line: dict[str, Any],
    audited_talents: list[dict[str, Any]],
) -> dict[str, Any]:
    """将第九步投放行与审核记录组装为写建议输入。"""
    record = find_audited_record(line.get("xhs_id"), audited_talents)

    if record.get("release_status") == "held_for_review":
        raise ValueError("暂存待复核的达人不能生成投放建议")

    placement = deepcopy(line)
    placement["format"] = normalize_portfolio_format(line["format"])
    placement.update(portfolio_price_fields(line))
    placement["talent"] = deepcopy(record["talent"])
    placement["evaluation"] = {
        field: deepcopy(record[field])
        for field in (
            "stance", "tag", "sentence", "band",
            "score", "attempt", "data_revision",
        )
        if field in record
    }
    for field in (
        "audit_status", "audit_issues", "data_incomplete",
        "missing_data_fields", "release_status",
    ):
        if field in record:
            placement[field] = deepcopy(record[field])

    placement["audit_field_note"] = (
        "release_status、data_incomplete、missing_data_fields 是可选异常标记；"
        "未设置仅表示上游未标记对应异常，不表示交接失败或资料缺失，"
        "也不表示全部数据已核实。数据缺口提示应依据上游明确的数据不全标记、"
        "具体缺项或已有问题说明，不得仅因这些字段未出现而新增待核实事项。"
    )
    return placement


def build_advice_placements(
    portfolio_plan: dict[str, Any],
    audited_talents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """组装已选投放项，并保留方案级提示。"""
    lines = portfolio_plan.get("lines")
    if not isinstance(lines, list):
        raise ValueError("portfolio_plan.lines 必须为列表")

    plan_warnings = collect_portfolio_warnings(portfolio_plan)
    placements = []

    for line in lines:
        placement = build_advice_placement(line, audited_talents)
        placement["warnings"] = list(dict.fromkeys(
            _upstream_warnings(placement) + plan_warnings
        ))
        placements.append(placement)

    return placements


def collect_portfolio_warnings(
    portfolio_plan: dict[str, Any],
) -> list[str]:
    """提取第九步方案提示，去除空白和重复项。"""
    warnings = portfolio_plan.get("warnings", [])
    if not isinstance(warnings, list) or any(
        not isinstance(item, str) for item in warnings
    ):
        raise ValueError("portfolio_plan.warnings 必须为字符串列表")

    return without_quote_missing_warnings([
        item.strip() for item in warnings if item.strip()
    ])


class AdviceAgent:
    """结合客户需求和上游结果，补全内容建议。

    - 接收客户需求和达人资料。
    - 接收第八步的评估理由、审核结果及待核实事项。
    - 接收第九步确定的投放名单、组合、形式、条数和价格。
    - 根据以上信息生成内容角度、子角度及内容建议。
    - 输出前检查内容建议是否符合客户内容方向和合规要求。
    - 保留上游决策和待核实事项，交后续人工审核。
    """

    # 职责分工：
    # 1. LLM 负责补写：内容角度、子角度、内容建议，并结合客户需求检查这些建议。
    # 2. Python 负责校验建议并关联回对应投放项，保留上游决策和待核实事项。
    #    最终推荐表整理与 Excel 导出归第十二步，当前类不负责。

    @property
    def system_prompt(self) -> str:
        """简短业务提示词，由 generate() 逐项调用。"""
        angles = json.dumps(ANGLES, ensure_ascii=False, indent=2)
        return f"""你是小红书达人投放的文案建议顾问，每次为一个已选投放项补全内容建议。

输入说明：
1. brief：客户需求。
   【上游已实现】来自第一步读取需求单，包含产品、卖点、目标人群、内容方向和合规要求。
2. placement：当前需要补写建议的一项投放安排。
   来自第九步 portfolio_plan.lines，经适配后包含：
   talent（达人资料）、evaluation（评估理由）、
   format（合作形式）、quantity（条数）及建议价格。
   image_text 表示图文，video 表示视频。
3. 第八步审核信息与上游提示：
   audit_status 表示理由审核状态；
   data_incomplete、missing_data_fields 表示数据缺口；
   release_status 表示上游交接状态；
   warnings 汇总当前已传入的投放项提示与方案提示。
   skipped_clear_band 表示按流程跳过模型理由审校；
   flagged_release 表示带缺数标记放行，均不代表完整业务审核通过。
   audit_field_note 说明可选异常字段的含义；上述异常字段未设置时，
   只表示上游未标记对应异常，不能据此推断交接失败、资料缺失或全部数据完整。
仅使用本次实际传入的资料；未提供的信息不代表已核实或已通过审核。

根据输入的客户需求、达人资料、第八步审核信息和第九步投放安排：
1. 从给定列表选择一个内容角度及对应子角度。
2. 写出适合本次合作形式的内容建议。
3. 输出前检查建议是否符合客户内容方向和合规要求，修正不符合的内容。

规则：
- 只使用已提供的资料，不编造产品信息、达人经历或使用效果。
- 不改变上游名单、形式、条数、价格和推荐理由。
- 保留上游待核实事项；无法解决的信息缺口写入 warnings。
- 对审核交接和数据缺口，只依据上游明确的数据不全标记、具体缺项或已有问题说明保留提示；
  不得仅因 release_status、data_incomplete、missing_data_fields 未出现，
  新增“交接状态待核实”“无法确认是否存在数据缺口”等提示。
- 只返回 JSON 对象，字段为 angle、sub_angle、details、warnings。
- warnings 为字符串数组，无待核实事项时返回 []。

允许的内容角度与子角度：
{angles}
"""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def call_llm(
        self, system_prompt: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """预留的 LLM 调用入口，返回 JSON 对象解析后的字典。

        system_prompt 必须明确要求返回 JSON 对象。
        沿用共享客户端的异常，由后续业务逻辑决定重试或交人工处理。
        此处只负责调用，不代表结果已经通过内容建议的业务校验。
        """
        return self.llm.complete_json(system_prompt, payload)

    def generate_from_portfolio(
        self,
        brief: dict[str, Any],
        portfolio_plan: dict[str, Any],
        audited_talents: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """将第九步方案适配后，调用已有的建议生成逻辑。"""
        placements = build_advice_placements(
            portfolio_plan, audited_talents
        )
        results = self.generate(brief, placements)
        for result in results:
            result["portfolio_revision"] = portfolio_plan.get("revision")
        return results

    def generate(
        self,
        brief: dict[str, Any],
        selected_placements: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """逐项生成并关联建议；格式失败重试一次，单条失败保留记录。

        服务故障向调用方报错；结构校验不代表已证明内容合规。
        每次按当前输入完整生成建议，不复用上一版内容。
        """
        if not isinstance(brief, dict) or not brief:
            raise ValueError("brief 必须为非空字典")
        if not isinstance(selected_placements, list):
            raise ValueError("selected_placements 必须为列表")
        seen = set()
        # 发起 API 调用前校验整批身份，避免重复或无法关联。
        for placement in selected_placements:
            if not isinstance(placement, dict):
                raise ValueError("投放项必须为字典")
            identity = placement.get("xhs_id")
            form = placement.get("format")
            if not isinstance(identity, str) or not identity.strip():
                raise ValueError("投放项缺少有效的 xhs_id")
            if form not in ("image_text", "video"):
                raise ValueError("投放项 format 必须为 image_text 或 video")
            if not isinstance(placement.get("talent"), dict):
                raise ValueError("投放项缺少 talent 字典")
            if placement["talent"].get("xhs_id") not in (None, identity):
                raise ValueError("投放项与达人资料的 ID 不一致")
            key = (identity, form)
            if key in seen:
                raise ValueError("同一达人同一形式的投放项重复")
            seen.add(key)
            _upstream_warnings(placement)

        results = []
        for placement in selected_placements:
            payload = {"brief": deepcopy(brief), "placement": deepcopy(placement)}
            original_warnings = without_pricing_date_warnings(without_clear_band_warnings(
                without_quote_missing_warnings(_upstream_warnings(placement)),
                placement.get("audit_status"),
            ))
            payload["placement"]["warnings"] = original_warnings
            payload["placement"].pop("remarks", None)
            record = deepcopy(placement)
            record["remarks"] = quote_remarks(placement["talent"])
            last_error = ""
            for attempt in range(2):
                try:
                    advice = validate_advice(self.call_llm(self.system_prompt, payload))
                except LLMServiceError:
                    raise
                except (LLMResponseError, AdviceValidationError) as exc:
                    last_error = str(exc)
                    payload["correction"] = "上次输出校验失败，请完整重写 JSON：" + last_error
                else:
                    record.update(advice)
                    record["warnings"] = without_pricing_date_warnings(without_clear_band_warnings(
                        without_quote_missing_warnings(original_warnings + advice["warnings"]),
                        placement.get("audit_status"),
                    ))
                    record["advice_status"] = "generated"
                    record["advice_error"] = None
                    break
            else:
                record.update(angle=None, sub_angle=None, details=None,
                              warnings=list(dict.fromkeys(original_warnings + ["内容建议生成失败：" + last_error])),
                              advice_status="failed", advice_error=last_error)
            results.append(record)
        return results


class AdviceValidationError(ValueError):
    """模型建议结构或枚举不符合约定。"""


def validate_advice(value: Any) -> dict[str, Any]:
    """仅检查结构和角度组合，不替代模型需求核对与人工审核。"""
    fields = {"angle", "sub_angle", "details", "warnings"}
    if not isinstance(value, dict) or set(value) != fields:
        raise AdviceValidationError("必须且仅返回 angle、sub_angle、details、warnings")
    angle, sub = value["angle"], value["sub_angle"]
    if not isinstance(angle, str) or angle not in ANGLES:
        raise AdviceValidationError("angle 不在允许列表中")
    if not isinstance(sub, str) or sub not in ANGLES[angle]:
        raise AdviceValidationError("sub_angle 不属于对应内容角度")
    if not isinstance(value["details"], str) or not value["details"].strip():
        raise AdviceValidationError("details 必须为非空文字")
    warnings = value["warnings"]
    if not isinstance(warnings, list) or any(not isinstance(w, str) or not w.strip() for w in warnings):
        raise AdviceValidationError("warnings 必须为非空文字组成的列表，或空列表")
    return {**value, "details": value["details"].strip(),
            "warnings": list(dict.fromkeys(w.strip() for w in warnings))}


def _upstream_warnings(placement: dict[str, Any]) -> list[str]:
    warnings = []
    for field in ("review_flags", "warnings"):
        values = placement.get(field, [])
        if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
            raise ValueError(f"{field} 必须为字符串列表")
        warnings.extend(v.strip() for v in values if v.strip())
    if placement.get("data_incomplete"):
        fields = placement.get("missing_data_fields", [])
        if not isinstance(fields, list) or any(not isinstance(f, str) for f in fields):
            raise ValueError("missing_data_fields 必须为字符串列表")
        warnings.append("上游数据不全：" + ("、".join(fields) or "具体缺项未提供"))
    if placement.get("release_status"):
        warnings.append("上游交接状态：" + str(placement["release_status"]))
    return list(dict.fromkeys(warnings))
