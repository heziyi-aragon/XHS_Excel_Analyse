"""需求分析与筛选策略：集中管理本阶段的 LLM 调用和业务规则。

LangGraph 节点只负责搬运状态；提示词、模型调用与输出校验都收口在这里。
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Callable

from agent.domain import (
    ALLOWED_CRITERION_OPERATORS,
    PolicyValidationError,
    RequirementProfile,
    ScreeningPolicy,
    TalentProfile,
)


JsonTransport = Callable[[str, dict[str, Any]], dict[str, Any]]


class LLMResponseError(ValueError):
    """模型返回了无法解析的结构化结果。"""


class LLMServiceError(RuntimeError):
    """模型服务因网络或代理问题暂时不可用。"""


class LLMClient:
    """OpenAI 兼容接口的最小封装，支持注入假客户端进行离线测试。"""

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        model: str = "gpt-4o-mini",
        transport: JsonTransport | None = None,
        trust_env: bool = False,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.transport = transport
        self.trust_env = trust_env
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_env(cls) -> "LLMClient":
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            # .env 支持是便利功能；直接使用环境变量时不应因此中断。
            pass
        return cls(
            api_key=os.getenv("API_KEY", ""),
            base_url=os.getenv("BASE_URL", ""),
            model=os.getenv("LLM_MODEL", os.getenv("MODEL", "gpt-4o-mini")),
            # Clash TUN 已经接管系统流量时，再读取 HTTP(S)_PROXY 会形成第二条、
            # 更容易配置错误的代理链。默认忽略它，仍可用环境变量显式开启。
            trust_env=os.getenv("LLM_TRUST_ENV", "false").strip().lower()
            in {"1", "true", "yes", "on"},
            timeout_seconds=cls._read_timeout(),
        )

    @staticmethod
    def _read_timeout() -> float:
        raw_value = os.getenv("LLM_TIMEOUT_SECONDS", "120").strip()
        try:
            timeout = float(raw_value)
        except ValueError as exc:
            raise RuntimeError("LLM_TIMEOUT_SECONDS 必须是秒数，例如 120") from exc
        if timeout <= 0:
            raise RuntimeError("LLM_TIMEOUT_SECONDS 必须大于 0")
        return timeout

    def complete_json(
        self, system_prompt: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        if self.transport is not None:
            result = self.transport(system_prompt, payload)
            if not isinstance(result, dict):
                raise LLMResponseError("测试或自定义 LLM 客户端必须返回字典")
            return result

        if not self.api_key:
            raise RuntimeError("缺少 API_KEY，无法执行需求分析智能体")

        try:
            from openai import (
                APIConnectionError,
                APITimeoutError,
                DefaultHttpxClient,
                OpenAI,
            )
        except ImportError as exc:
            raise RuntimeError("缺少 openai 依赖，请先安装 requirements.txt") from exc

        client_kwargs: dict[str, Any] = {
            "api_key": self.api_key,
            "http_client": DefaultHttpxClient(trust_env=self.trust_env),
            "timeout": self.timeout_seconds,
        }
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        client = OpenAI(**client_kwargs)
        try:
            response = client.chat.completions.create(
                model=self.model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": json.dumps(payload, ensure_ascii=False, indent=2),
                    },
                ],
            )
        except APITimeoutError as exc:
            route = (
                "当前正在读取 HTTP_PROXY/HTTPS_PROXY，请检查代理地址和端口。"
                if self.trust_env
                else "程序已忽略 HTTP_PROXY/HTTPS_PROXY；请确认 Clash Verge 的 TUN 模式和当前节点可用。"
            )
            raise LLMServiceError(f"连接模型服务超时。{route}") from exc
        except APIConnectionError as exc:
            route = (
                "请检查 HTTP_PROXY/HTTPS_PROXY 的协议、地址和端口。"
                if self.trust_env
                else "请确认 Clash Verge 的 TUN 模式和当前节点可用。"
            )
            raise LLMServiceError(f"无法连接模型服务。{route}") from exc
        content = response.choices[0].message.content or ""
        try:
            result = json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMResponseError(f"LLM 未返回合法 JSON：{exc.msg}") from exc
        if not isinstance(result, dict):
            raise LLMResponseError("LLM JSON 的顶层必须是对象")
        return result


class RequirementAnalysisAgent:
    """用 LLM 区分客户决策、顾问判断和后续依赖。"""

    SYSTEM_PROMPT = """你是广告代理公司的资深小红书投放顾问助理。
你的工作不是把所有不确定性都退回给客户，而是先区分已经明确的事实、可以继续处理的事项、必须由客户决定的问题，以及投放执行前再处理的事项。

请把需求分成以下几类：
1. campaign_constraints：预算、周期、产品、内容合规等项目整体约束；
2. hard_requirements：能够直接作用于达人、决定通过或不通过的明确条件；
3. soft_preferences：用于达人排序和权重分配的偏好；
4. exclusions：客户明确排除的达人或账号类型；
5. consultant_assumptions：原文未完全定义，但顾问可以按数据字段、行业常规或保守原则作出的合理假设；
6. downstream_dependencies：需要在报价、内容制作或效果验收前补齐，但不妨碍当前达人筛选的信息；
7. clarification_questions：必须由客户拥有决策权、不同答案会显著改变硬筛/合规/验收、且无法作合理假设的问题。

遵守以下边界：
- 达人层级、达人数量、预算分配、图文视频比例、筛选权重和执行阈值属于顾问方案，不要反问客户；
- 上述方案也不在本节点提前拍板，不得自行给出预算百分比、图文视频比例、达人数量、粉丝分层区间、税费或服务费规则；
- 可使用需求单和达人表已经提供的字段口径，不要再次追问其定义；
- KPI 报告口径、证明材料等若不影响当前筛选，放入 downstream_dependencies；
- clarification_questions 最多 3 个，必须是本轮无法继续的真正阻断项；没有阻断项就返回空数组；
- 不得虚构客户已经明确给出的数字，但允许把顾问采用的默认判断写进 consultant_assumptions；
- hard_requirements 只写客户明确规定“必须/仅限/不得低于”的达人硬门槛，不要放产品名称、总预算、投放周期或一般内容规范；需求单中写着“优先”“为主、为辅”的条件（如某垂类优先、新达人优先、图文为主视频为辅）属于排序偏好，必须放入 soft_preferences，不得改写成“不进入候选”放进 hard_requirements。
- exclusions 已单独列出，hard_requirements 中不要重复同一条排除要求。
- client_answers 非空时，把它视为客户对上一轮问题的补充；已经回答的问题不得再次追问。
- 你在本节点看不到达人数据的实际分布，不得声称现有达人与目标人群匹配、数据表现良好或能够完成 KPI。
- 未做预算测算时，不得声称预算偏紧、充足、合理或足以完成目标。
- 不得把常见行业做法冒充客户事实。需要采用默认处理方式时，必须说明需求单缺少什么、现有信息支持怎样处理。

写作要求：
- 写给需要快速复核方案的广告顾问，使用自然、简洁、具体的中文；
- summary 用 2 至 3 句话概括需求单已经明确的达人方向、考核目标和排除要求，不超过 120 个汉字；没有事实依据时不要添加风险判断；
- 每条只表达一个意思，直接说结论，不使用“赋能、抓手、闭环、综合考量”等空话；
- 不写字段名、程序术语或英文枚举，不解释你的推理过程；
- 每个判断都要带上事实依据。优先使用“需求单明确……，因此……”或“需求单没有说明……，现有信息只能支持……”这样的自然句式；
- consultant_assumptions 中每条都要说明缺少的信息、采用当前处理方式的依据，以及该处理方式适用于哪个环节；
- downstream_dependencies 中每条都要说明需要在什么时间补充什么信息，以及不补充会影响什么；
- clarification_questions 中每条都要说明具体缺口和它会改变的筛选、合规或验收结果，不能只写“请确认某某口径”；
- campaign_constraints 最多 5 条，hard_requirements 最多 4 条，soft_preferences 最多 5 条，exclusions 最多 3 条，consultant_assumptions 最多 3 条，downstream_dependencies 最多 4 条；
- 各栏目之间不得重复同一件事。

只返回一个 JSON 对象，字段必须为：
summary, objective, target_audience, kpi, campaign_constraints, hard_requirements,
soft_preferences, exclusions, consultant_assumptions, downstream_dependencies,
clarification_questions, is_complete。
列表字段必须是字符串数组。只有 clarification_questions 非空时，is_complete 才为 false。

参考下面的简洁程度和表达方式，只学习结构，不要照抄内容：
{
  "summary": "需求单将年轻女性列为主要受众，并要求优先选择护肤内容达人。纯生活记录账号和已有多次合作记录的达人不进入候选。",
  "objective": "扩大产品认知并带动站内搜索",
  "target_audience": "年轻女性护肤人群",
  "kpi": "曝光、互动成本和品牌词搜索增长",
  "campaign_constraints": ["需求单明确总费用不得超过预算上限，后续组合需要在该金额内安排"],
  "hard_requirements": ["需求单要求达人能够承接护肤内容，不具备相关内容经验的达人不进入候选"],
  "soft_preferences": ["需求单将年轻女性列为主要受众，因此比较达人时优先参考女性粉丝占比"],
  "exclusions": ["需求单明确排除纯生活记录账号"],
  "consultant_assumptions": ["需求单没有规定近三个月的截止日；筛选工作有明确执行日期，因此按执行日向前计算三个月"],
  "downstream_dependencies": ["发布内容前需要确定审核人与修改时限，否则会影响达人排期"],
  "clarification_questions": [],
  "is_complete": true
}
"""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def analyze(
        self,
        brief: dict[str, Any],
        brief_raw: dict[str, Any],
        client_answers: str = "",
    ) -> RequirementProfile:
        payload = {
            "structured_brief": brief,
            "original_brief": brief_raw,
            "client_answers": client_answers,
        }
        try:
            result = self.llm.complete_json(self.SYSTEM_PROMPT, payload)
        except LLMResponseError as first_error:
            payload["correction"] = f"上次输出无法解析：{first_error}。请严格返回 JSON。"
            result = self.llm.complete_json(self.SYSTEM_PROMPT, payload)

        profile = RequirementProfile.from_dict(result)
        self._apply_required_checks(profile)
        return profile

    @staticmethod
    def _apply_required_checks(profile: RequirementProfile) -> None:
        required = {
            "objective": "投放目标",
            "target_audience": "目标人群",
            "kpi": "核心 KPI",
        }
        for field_name, label in required.items():
            if not getattr(profile, field_name):
                question = f"请确认本次投放的{label}。"
                if question not in profile.clarification_questions:
                    profile.clarification_questions.append(question)

        if not profile.hard_requirements and not profile.soft_preferences:
            profile.consultant_assumptions.append(
                "需求未给出明确达人指标时，由顾问依据核心 KPI 和现有数据设置筛选权重。"
            )

        limits = {
            "campaign_constraints": 5,
            "hard_requirements": 4,
            "soft_preferences": 5,
            "exclusions": 3,
            "consultant_assumptions": 3,
            "downstream_dependencies": 4,
        }
        for field_name, limit in limits.items():
            items = getattr(profile, field_name)
            setattr(
                profile,
                field_name,
                RequirementAnalysisAgent._deduplicate_questions(items)[:limit],
            )

        profile.clarification_questions = RequirementAnalysisAgent._deduplicate_questions(
            profile.clarification_questions
        )[:3]
        profile.is_complete = not profile.clarification_questions

    @staticmethod
    def _deduplicate_questions(questions: list[str]) -> list[str]:
        """去掉同义前缀和标点造成的重复，保留更完整的问法。"""
        unique: list[tuple[str, str]] = []
        for question in questions:
            text = question.strip()
            if not text:
                continue
            key = re.sub(r"[\W_]+", "", text.lower())
            key = re.sub(r"^(请确认|需要确认|请问)", "", key)
            duplicate_index = next(
                (
                    index
                    for index, (known_key, _known_text) in enumerate(unique)
                    if key == known_key
                    or (len(key) >= 8 and key in known_key)
                    or (len(known_key) >= 8 and known_key in key)
                ),
                None,
            )
            if duplicate_index is None:
                unique.append((key, text))
            elif len(text) > len(unique[duplicate_index][1]):
                unique[duplicate_index] = (key, text)
        return [text for _key, text in unique]


_FIELD_DESCRIPTIONS: dict[str, str] = {
    "nickname": "达人昵称",
    "xhs_id": "小红书号",
    "mcn": "所属机构",
    "is_new_kol": "是否 New KOL（表中取值为 Y 或 N）",
    "content_type": "内容类型（表中取值为 Beauty、Lifestyle、Fashion、Mother&Baby、Food）",
    "tier": "达人层级（表中取值为 T1、T2、T3、T4）",
    "main_format": "主营内容形式（表中取值为 图文为主、视频为主、图文视频均衡、vlog为主）",
    "price_image": "图文参考价（元）",
    "price_video": "视频参考价（元）",
    "fans_wan": "粉丝数（万）",
    "ratio_age_18_24": "18-24 岁粉丝占比",
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


def talent_filter_fields() -> set[str]:
    """返回允许 LLM 在筛选策略中引用的达人字段。"""
    model_fields = set(TalentProfile.__dataclass_fields__)
    return model_fields.intersection(_FIELD_DESCRIPTIONS)


class ScreeningPolicyAgent:
    """用 LLM 将结构化需求翻译为可执行条件，并用代码强制校验。"""

    SYSTEM_PROMPT = """你是小红书达人投放的筛选策略智能体。
请将结构化需求转换成机器可执行的筛选条件：
- 只有客户明确要求“必须满足”或“明确排除”（如排除纯 vlog 生活号）的条件才设为 hard=true、weight=0；需求单中表述为“优先”“为主、为辅”的垂类偏好（content_type）、新达人偏好（is_new_kol）或图文/视频形式偏好必须设为 hard=false 并分配权重参与打分，不得设为 hard=true 直接淘汰；
- 影响候选人排序的指标设为 hard=false，并分配正数权重；没有明确阈值时使用 maximize 或 minimize，不得编造阈值；
- 所有软性条件的权重总和必须等于 1；
- 不得使用给定字段清单以外的字段；不得臆造需求中没有的阈值；
- campaign_constraints、consultant_assumptions 和 downstream_dependencies 不是达人字段，不要硬塞进 criteria；
- 无法用现有字段自动判断、但仍需要落实的达人要求写入 manual_checks，不要因此停止生成策略；
- operator 只能使用给定的运算符。
- criteria 只保留最重要的 5 至 8 条；同类指标合并取舍，不要为了覆盖所有字段而堆砌；
- manual_checks 最多 5 条，每条只保留一个可执行核对动作，不放合同、物流、内容审核或投后复盘事项；
- rationale 用自然中文写 2 句话，不超过 120 个汉字；不得出现 criteria、manual_checks、maximize、minimize 等程序字段名；
- reason 和 manual_checks 要像顾问工作说明，使用短句，不复述整份需求。
- 每条 reason 必须指出它对应需求分析中的哪项事实；不能只写“表现更好”“更匹配”“兼顾效率”等没有来源的结论。
- 你没有收到达人的实际数值，不得声称候选达人表现良好、人群已经匹配、预算足够或 KPI 可以完成。
- rationale 只能解释为什么选择这些指标、为什么这样分配权重，不得描述尚未计算的筛选结果。
- 如果输入中包含 consultant_review，必须针对顾问意见修改上一版方案，不得忽略或原样返回。

只返回一个 JSON 对象。参考下面的结构和简洁程度：
{
  "name": "护肤达人筛选策略",
  "criteria": [
    {"field": "main_format", "operator": "not_contains", "value": "vlog", "hard": true, "weight": 0, "reason": "需求单明确排除纯生活记录账号"},
    {"field": "ratio_female", "operator": "maximize", "value": null, "hard": false, "weight": 0.4, "reason": "需求单将女性列为主要受众"},
    {"field": "cpe", "operator": "minimize", "value": null, "hard": false, "weight": 0.6, "reason": "需求单把互动成本列为考核指标"}
  ],
  "manual_checks": ["查看近期内容，确认账号长期聚焦护肤主题"],
  "rationale": "需求单明确排除纯生活记录账号，因此先按内容形式剔除。目标受众和互动成本都有明确要求，剩余达人据此比较女性粉丝占比和单次互动成本。"
}
"""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        self.available_fields = talent_filter_fields()

    def build(
        self,
        profile: RequirementProfile,
        review_feedback: str = "",
        previous_policy: dict[str, Any] | None = None,
    ) -> ScreeningPolicy:
        if not profile.is_complete:
            raise ValueError("需求尚未确认完整，不能生成筛选条件与权重")

        payload: dict[str, Any] = {
            "requirement_profile": profile.to_dict(),
            "available_fields": {
                key: _FIELD_DESCRIPTIONS[key] for key in sorted(self.available_fields)
            },
            "allowed_operators": sorted(ALLOWED_CRITERION_OPERATORS),
        }
        if review_feedback:
            payload["consultant_review"] = {
                "feedback": review_feedback,
                "previous_policy": previous_policy or {},
                "instruction": (
                    "根据顾问意见修改筛选方案。保留未被要求修改且仍合理的部分，"
                    "完整返回修订后的 JSON。"
                ),
            }
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                result = self.llm.complete_json(self.SYSTEM_PROMPT, payload)
                policy = ScreeningPolicy.from_dict(result)
                policy.validate(self.available_fields)
                return policy
            except (LLMResponseError, PolicyValidationError) as exc:
                last_error = exc
                if attempt == 0:
                    payload["correction"] = {
                        "message": "上次输出未通过校验，请修正后完整重写 JSON。",
                        "errors": getattr(exc, "errors", [str(exc)]),
                    }

        assert last_error is not None
        raise last_error
