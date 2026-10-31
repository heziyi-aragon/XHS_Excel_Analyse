"""领域对象层：Brief（需求单）与 TalentProfile（达人档案）。

这一层完全不认识 LangGraph，也不认识 Excel——
loader 负责把 Excel 变成这里的对象，node 负责把对象放进状态。
任何一层想单独测试，都不需要另外两层在场。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


def parse_number(value: Any) -> float | None:
    """把表格里的脏值解析成数字。解析不了返回 None，绝不抛异常。

    能处理：9500 / '¥350,000（含达人合作费用）' / 0.113 / '-'
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text in {"", "-", "—", "无", "N/A"}:
        return None
    # 去掉货币符号、千分位和括号备注，保留数字与小数点
    cleaned = "".join(ch for ch in text if ch.isdigit() or ch == ".")
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_ratio(value: Any) -> float | None:
    """解析比例：0.942 或 '94.2%' 都归一成 0.942。"""
    if isinstance(value, str) and value.strip().endswith("%"):
        num = parse_number(value[:-1])
        return None if num is None else num / 100
    return parse_number(value)


@dataclass
class Brief:
    """甲方需求单。raw 保留标签->原文的映射，供后续 LLM 节点直接阅读。"""

    brand: str = ""
    product: str = ""
    product_price: str = ""
    objective: str = ""
    selling_points: str = ""
    target_audience: str = ""
    total_budget: float | None = None
    campaign_period: str = ""
    kpi: str = ""
    content_direction: str = ""
    compliance: str = ""
    creator_requirements: str = ""
    exclusions: str = ""
    notes: str = ""
    raw: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# 达人表中「数值型」列：表头关键字 -> (字段名, 解析方式)
_TALENT_NUMERIC: dict[str, tuple[str, Any]] = {
    "平台图文参考价": ("price_image", parse_number),
    "平台视频参考价": ("price_video", parse_number),
    "粉丝数": ("fans_wan", parse_number),
    "18-24岁粉丝占比": ("ratio_age_18_24", parse_ratio),
    "＞44岁粉丝占比": ("ratio_age_over_44", parse_ratio),
    "女粉占比": ("ratio_female", parse_ratio),
    "CTR": ("ctr", parse_ratio),
    "CPV": ("cpv", parse_number),
    "合作笔记平均CPE": ("cpe", parse_number),
    "蒲公英前选CPUV": ("cpuv", parse_number),
    "曝光数中位值": ("exposure_med", parse_number),
    "阅读数中位值": ("read_med", parse_number),
    "互动量中位值": ("interact_med", parse_number),
    "外溢进店中位数": ("visit_store_med", parse_number),
    "阅读来源-发现页占比": ("ratio_discover", parse_ratio),
    "阅读来源-搜索页占比": ("ratio_search", parse_ratio),
    "近90天合作笔记数": ("ads_count_90d", parse_number),
}

# 达人表中「文本型」列：表头关键字 -> 字段名
_TALENT_TEXT: dict[str, str] = {
    "达人昵称": "nickname",
    "小红书号": "xhs_id",
    "所属机构": "mcn",
    "是否New KOL": "is_new_kol",
    "Type": "content_type",
    "Tier": "tier",
    "主营内容形式": "main_format",
}

# 顾问界面和代码生成的理由共用同一套中文名，避免各写一套。
FIELD_LABELS: dict[str, str] = {
    "mcn": "所属机构",
    "is_new_kol": "New KOL 标记",
    "content_type": "内容类型",
    "tier": "达人层级",
    "main_format": "主营内容形式",
    "price_image": "图文参考价",
    "price_video": "视频参考价",
    "fans_wan": "粉丝量",
    "ratio_age_18_24": "18–24 岁粉丝占比",
    "ratio_age_over_44": "44 岁以上粉丝占比",
    "ratio_female": "女性粉丝占比",
    "ctr": "点击率",
    "cpv": "单次播放成本",
    "cpe": "单次互动成本",
    "cpuv": "进店成本",
    "exposure_med": "曝光中位数",
    "read_med": "阅读中位数",
    "interact_med": "互动中位数",
    "visit_store_med": "进店中位数",
    "ratio_discover": "发现页阅读占比",
    "ratio_search": "搜索页阅读占比",
    "ads_count_90d": "近 90 天合作笔记数",
}

RATIO_FIELDS = frozenset(
    {
        "ratio_age_18_24",
        "ratio_age_over_44",
        "ratio_female",
        "ctr",
        "ratio_discover",
        "ratio_search",
    }
)
MONEY_FIELDS = frozenset({"price_image", "price_video"})


def format_metric(field: str, value: Any) -> str:
    """把达人字段格式化成顾问可读的一个值。"""
    if value is None or value == "":
        return "缺失"
    if field in RATIO_FIELDS and isinstance(value, (int, float)):
        return f"{float(value) * 100:.1f}%"
    if field in MONEY_FIELDS and isinstance(value, (int, float)):
        return f"¥{float(value):,.0f}"
    if field == "fans_wan" and isinstance(value, (int, float)):
        return f"{float(value):.1f}万"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (int, float)):
        number = float(value)
        if number.is_integer():
            return f"{int(number):,}"
        text = f"{number:.2f}".rstrip("0").rstrip(".")
        return text
    return str(value)


@dataclass
class TalentProfile:
    """单个达人的档案。没有对应数据的字段就是 None，不猜、不补。"""

    nickname: str = ""
    xhs_id: str = ""
    mcn: str | None = None
    is_new_kol: str = ""
    content_type: str = ""
    tier: str = ""
    main_format: str = ""
    price_image: float | None = None
    price_video: float | None = None
    fans_wan: float | None = None
    ratio_age_18_24: float | None = None
    ratio_age_over_44: float | None = None
    ratio_female: float | None = None
    ctr: float | None = None
    cpv: float | None = None
    cpe: float | None = None
    cpuv: float | None = None
    exposure_med: float | None = None
    read_med: float | None = None
    interact_med: float | None = None
    visit_store_med: float | None = None
    ratio_discover: float | None = None
    ratio_search: float | None = None
    ads_count_90d: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _string_list(value: Any) -> list[str]:
    """把 LLM 返回的不稳定列表安全收敛成字符串列表。"""
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "是"}
    return bool(value)


@dataclass
class RequirementProfile:
    """需求分析节点的结构化输出。"""

    summary: str = ""
    objective: str = ""
    target_audience: str = ""
    kpi: str = ""
    campaign_constraints: list[str] = field(default_factory=list)
    hard_requirements: list[str] = field(default_factory=list)
    soft_preferences: list[str] = field(default_factory=list)
    exclusions: list[str] = field(default_factory=list)
    consultant_assumptions: list[str] = field(default_factory=list)
    downstream_dependencies: list[str] = field(default_factory=list)
    # 兼容旧的中间产物；模糊点本身不再自动阻断流程。
    ambiguities: list[str] = field(default_factory=list)
    clarification_questions: list[str] = field(default_factory=list)
    is_complete: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RequirementProfile":
        questions = _string_list(data.get("clarification_questions"))
        ambiguities = _string_list(data.get("ambiguities"))
        return cls(
            summary=str(data.get("summary") or "").strip(),
            objective=str(data.get("objective") or "").strip(),
            target_audience=str(data.get("target_audience") or "").strip(),
            kpi=str(data.get("kpi") or "").strip(),
            campaign_constraints=_string_list(data.get("campaign_constraints")),
            hard_requirements=_string_list(data.get("hard_requirements")),
            soft_preferences=_string_list(data.get("soft_preferences")),
            exclusions=_string_list(data.get("exclusions")),
            consultant_assumptions=_string_list(data.get("consultant_assumptions")),
            downstream_dependencies=_string_list(data.get("downstream_dependencies")),
            ambiguities=ambiguities,
            clarification_questions=questions,
            # 只有明确列出的客户阻断问题才暂停；普通模糊点由顾问作假设。
            is_complete=not questions,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ALLOWED_CRITERION_OPERATORS = frozenset(
    {
        "eq",
        "ne",
        "gt",
        "gte",
        "lt",
        "lte",
        "contains",
        "not_contains",
        "in",
        "not_in",
        "is_missing",
        "not_missing",
        "maximize",
        "minimize",
    }
)


@dataclass
class FilterCriterion:
    """一条机器可执行的筛选或评分条件。"""

    field: str
    operator: str
    value: Any = None
    hard: bool = False
    weight: float = 0.0
    reason: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FilterCriterion":
        weight = parse_ratio(data.get("weight"))
        return cls(
            field=str(data.get("field") or "").strip(),
            operator=str(data.get("operator") or "").strip(),
            value=data.get("value"),
            hard=_parse_bool(data.get("hard")),
            weight=0.0 if weight is None else float(weight),
            reason=str(data.get("reason") or "").strip(),
        )


class PolicyValidationError(ValueError):
    """筛选策略不可执行；errors 可直接反馈给 LLM 修正。"""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("；".join(errors))


@dataclass
class ScreeningPolicy:
    """筛选条件与权重节点的结构化输出。"""

    name: str = ""
    criteria: list[FilterCriterion] = field(default_factory=list)
    manual_checks: list[str] = field(default_factory=list)
    rationale: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScreeningPolicy":
        raw_criteria = data.get("criteria")
        criteria = []
        if isinstance(raw_criteria, list):
            criteria = [
                FilterCriterion.from_dict(item)
                for item in raw_criteria
                if isinstance(item, dict)
            ]
        return cls(
            name=str(data.get("name") or "").strip(),
            criteria=criteria,
            manual_checks=_string_list(data.get("manual_checks")),
            rationale=str(data.get("rationale") or "").strip(),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self, available_fields: set[str]) -> None:
        """用确定性规则约束 LLM 输出，保证后续代码可以直接执行。"""
        errors: list[str] = []
        if not self.criteria:
            errors.append("筛选策略至少需要一条条件")
        elif len(self.criteria) > 8:
            errors.append(f"筛选策略最多保留 8 条核心条件，当前为 {len(self.criteria)} 条")

        if len(self.manual_checks) > 5:
            errors.append(
                f"人工核对项最多保留 5 条，当前为 {len(self.manual_checks)} 条"
            )

        for index, criterion in enumerate(self.criteria, start=1):
            prefix = f"第 {index} 条条件"
            if criterion.field not in available_fields:
                errors.append(f"{prefix}使用了不存在的字段「{criterion.field}」")
            if criterion.operator not in ALLOWED_CRITERION_OPERATORS:
                errors.append(f"{prefix}使用了不支持的运算符「{criterion.operator}」")
            if not criterion.reason:
                errors.append(f"{prefix}没有说明采用该条件的事实依据")
            if (
                criterion.operator
                not in {"is_missing", "not_missing", "maximize", "minimize"}
                and criterion.value is None
            ):
                errors.append(f"{prefix}缺少比较值")
            if criterion.hard and criterion.operator in {"maximize", "minimize"}:
                errors.append(f"{prefix}是硬条件，不能使用趋势评分运算符")
            if criterion.hard and criterion.weight != 0:
                errors.append(f"{prefix}是硬条件，权重必须为 0")
            if not criterion.hard and criterion.weight <= 0:
                errors.append(f"{prefix}是软条件，权重必须大于 0")

        soft_weight = sum(c.weight for c in self.criteria if not c.hard)
        if not any(not c.hard for c in self.criteria):
            errors.append("筛选策略至少需要一条软性评分条件")
        elif abs(soft_weight - 1.0) > 0.001:
            errors.append(f"软性条件权重总和必须为 1，当前为 {soft_weight:.4f}")

        errors.extend(self._find_hard_conflicts())
        if errors:
            raise PolicyValidationError(errors)

    def _find_hard_conflicts(self) -> list[str]:
        """检查常见的等值和数值区间冲突，避免生成永远无法通过的策略。"""
        errors: list[str] = []
        by_field: dict[str, list[FilterCriterion]] = {}
        for criterion in self.criteria:
            if criterion.hard:
                by_field.setdefault(criterion.field, []).append(criterion)

        for field_name, criteria in by_field.items():
            equals = {str(c.value) for c in criteria if c.operator == "eq"}
            if len(equals) > 1:
                errors.append(f"字段「{field_name}」存在互相冲突的等值硬条件")

            lowers = [
                float(c.value)
                for c in criteria
                if c.operator in {"gt", "gte"} and isinstance(c.value, (int, float))
            ]
            uppers = [
                float(c.value)
                for c in criteria
                if c.operator in {"lt", "lte"} and isinstance(c.value, (int, float))
            ]
            if lowers and uppers and max(lowers) > min(uppers):
                errors.append(f"字段「{field_name}」的数值上下限互相冲突")
        return errors


@dataclass
class LoadReport:
    """加载过程的体检报告。有问题记在这里，不中断流程。"""

    issues: list[str] = field(default_factory=list)

    def add(self, message: str) -> None:
        self.issues.append(message)
