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

    # -- 后续步骤需要的输入说明 --
    next_step: dict

    # needs_client_confirmation / requirements_ready / awaiting_policy_review /
    # policy_revision_requested / policy_approved / screening / scored /
    # reasons_ready
    workflow_status: str
