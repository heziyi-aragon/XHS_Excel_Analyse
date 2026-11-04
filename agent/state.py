"""GraphState：整张图唯一的数据总线。

约定：
- 只放可序列化的 dict / list / str / number，领域对象在节点内部用完就转 dict。
  这样以后接 LangGraph 的 checkpointer（暂停等人）不用改任何代码。
- total=False：允许字段还没被填，后续节点自己判断有没有。
"""
from __future__ import annotations

from operator import add
from typing import Annotated, TypedDict


class GraphState(TypedDict, total=False):
    # -- 输入 --
    brief_path: str          # 需求单文件路径
    talent_path: str         # 达人数据文件路径

    # -- 第一步产出：读取需求清单 & 达人列表 --
    brief_raw: dict          # 需求单 标签->原文（给 LLM 节点看的原文）
    brief: dict              # 结构化后的 Brief
    talents: list[dict]      # 结构化后的达人列表
    issues: list[str]        # 加载与解析过程中的所有问题记录

    # -- 第二步产出：需求分析（LLM） --
    requirement_profile: dict
    clarification_questions: list[str]
    client_answers: str

    # -- 第三步产出：筛选条件与权重（LLM + 代码校验） --
    screening_policy: dict
    policy_review_status: str
    policy_review_feedback: str
    policy_revision_count: int
    approved_policy: dict

    # -- 第四步：清洗，标记缺失值 --
    cleaned_talents: list[dict]

    # -- 第五步：硬性筛选、同机构粉丝去重、最多 3 次放宽 --
    active_policy: dict
    screen_relax_count: int
    screened_talents: list[dict]
    excluded_talents: list[dict]
    deduped_talents: list[dict]
    screen_log: list[str]

    # -- 第六步：打分分档。明确档已写好标签；中间档等待扇出 --
    scored_talents: list[dict]
    reason_queue: list[dict]

    # -- 第七步：中间档一人一条。并行结果只能追加，不能整表替换 --
    reason_results: Annotated[list[dict], add]
    evaluated_talents: list[dict]

    # -- 第八步审核：节点逐人处理后整体替换最新结果，不采用并行追加 reducer --
    # audit_results 只包含中间档的有效 AuditResult；明确档跳过与执行错误另有状态。
    audit_results: list[dict]
    audited_talents: list[dict]  # 全名单 + 审核状态；缺数放行或 held_for_review 暂存标记
    audit_pending: list[dict]   # 中间态为待处理任务；交接时为暂存问题，须随方案带至最终人工审核
    pricing_candidates: list[dict]  # 原 stance=recommend：正常候选及补数失败标记放行者
    supplement_results: list[dict]  # 成功/部分查回记录及失败放行记录，含值与来源，供后续人工审核
    supplement_history: list[dict]  # 每次查回结果与来源、原评估快照；不覆盖旧记录
    supplement_reassessment_required: bool  # 有新值时经原评估、汇总、再审核
    audit_context: dict  # 上一轮需求和策略快照；上下文改变时不复用旧审核
    audit_history: list[dict]  # 实际执行过的中间档审核：当时评估、审核结论及上下文
    audit_summary: dict  # 覆盖范围与计数，不将理由审校等同于完整业务验收
    audit_unverified_checks: list[dict]  # 策略人工检查项，无核实证据时保持 unverified，随方案交接
    manual_review_queue: list[dict]  # 兼容旧交接格式，始终为空；不再产生提前人工业务节点

    # -- 第九步：组合与定价；审核快照独立保留，不改变任何单人评估 --
    portfolio_plan: dict | None
    portfolio_config: dict
    portfolio_feedback: dict | str | None
    portfolio_config_patch: dict | None
    portfolio_error: dict | None
    portfolio_previous_plan: dict | None
    portfolio_review: dict
    portfolio_review_status: str
    portfolio_review_history: list[dict]
    audit_workflow_status: str
    audit_next_step: dict

    # -- 后续步骤需要的输入说明 --
    next_step: dict

    # needs_client_confirmation / requirements_ready / awaiting_policy_review /
    # policy_revision_requested / policy_approved / screening / scored /
    # reasons_ready / audit_rewrite_required / rewriting_reasons / audit_complete /
    # audit_complete_with_incomplete_data（补数失败标记放行）/
    # audit_needs_resolution / reevaluating_after_supplement（补数后重新评估）/
    # audit_complete_with_pending（部分暂存，其余继续）/
    # portfolio_ready / portfolio_best_effort / portfolio_no_candidates /
    # portfolio_needs_pricing / portfolio_needs_input
    workflow_status: str
