"""需求阶段的 LangGraph 适配层。

这个文件只负责读取/写回 GraphState，业务逻辑由 agent.requirements 中的对象承担。
"""
from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from agent.domain import RequirementProfile
from agent.requirements import (
    LLMClient,
    RequirementAnalysisAgent,
    ScreeningPolicyAgent,
)


class RequirementNodes:
    """将两个面向对象的业务 Agent 暴露为两个清晰的图节点。"""

    def __init__(self, llm: LLMClient) -> None:
        self.analyzer = RequirementAnalysisAgent(llm)
        self.policy_builder = ScreeningPolicyAgent(llm)

    def analyze_requirements(self, state: dict[str, Any]) -> dict[str, Any]:
        profile = self.analyzer.analyze(
            brief=state.get("brief", {}),
            brief_raw=state.get("brief_raw", {}),
            client_answers=str(state.get("client_answers") or ""),
        )
        status = (
            "requirements_ready"
            if profile.is_complete
            else "needs_client_confirmation"
        )
        return {
            "requirement_profile": profile.to_dict(),
            "clarification_questions": profile.clarification_questions,
            "workflow_status": status,
        }

    def build_screening_policy(self, state: dict[str, Any]) -> dict[str, Any]:
        raw_profile = state.get("requirement_profile")
        if not isinstance(raw_profile, dict):
            raise ValueError("缺少 requirement_profile，不能生成筛选策略")
        profile = RequirementProfile.from_dict(raw_profile)
        policy = self.policy_builder.build(
            profile,
            review_feedback=str(state.get("policy_review_feedback") or ""),
            previous_policy=state.get("screening_policy"),
        )
        return {
            "screening_policy": policy.to_dict(),
            "policy_review_status": "pending",
            "policy_review_feedback": "",
            "workflow_status": "awaiting_policy_review",
        }

    def request_client_confirmation(self, state: dict[str, Any]) -> dict[str, Any]:
        """只在存在客户专属阻断问题时暂停，并接收客户补充信息。"""
        questions = state.get("clarification_questions", [])
        response = interrupt(
            {
                "type": "client_confirmation",
                "questions": questions,
                "message": "这些问题会影响筛选、合规或验收，需要客户答复后再继续。",
            }
        )
        if isinstance(response, dict):
            answer = str(response.get("answers") or "").strip()
        else:
            answer = str(response or "").strip()
        if not answer:
            raise ValueError("客户答复不能为空")
        return {
            "client_answers": answer,
            "clarification_questions": [],
            "workflow_status": "client_answered",
        }

    def review_policy(self, state: dict[str, Any]) -> dict[str, Any]:
        """等待顾问批准方案，或带着修改意见返回策略节点。"""
        policy = state.get("screening_policy", {})
        response = interrupt(
            {
                "type": "policy_review",
                "policy": policy,
                "message": "筛选方案已经生成。请确认采用，或直接写下需要修改的地方。",
            }
        )

        action = ""
        feedback = ""
        if isinstance(response, dict):
            action = str(response.get("action") or "").strip().lower()
            feedback = str(response.get("feedback") or "").strip()
        else:
            text = str(response or "").strip()
            action = text.lower()
            if action not in {"approve", "approved", "同意", "批准", "通过"}:
                action = "revise"
                feedback = text

        if action in {"approve", "approved", "同意", "批准", "通过"}:
            return {
                "approved_policy": policy,
                "policy_review_status": "approved",
                "workflow_status": "policy_approved",
            }

        if not feedback:
            raise ValueError("要求修改筛选方案时，需要提供具体修改意见")
        return {
            "policy_review_status": "revision_requested",
            "policy_review_feedback": feedback,
            "policy_revision_count": int(state.get("policy_revision_count") or 0)
            + 1,
            "workflow_status": "policy_revision_requested",
        }


def route_after_requirement_analysis(state: dict[str, Any]) -> str:
    """有客户阻断问题时暂停；否则继续生成筛选条件与权重。"""
    if state.get("workflow_status") == "requirements_ready":
        return "build_policy"
    return "client_confirmation"


def route_after_policy_review(state: dict[str, Any]) -> str:
    if state.get("policy_review_status") == "approved":
        return "approved"
    return "revise"


def prepare_next_step(state: dict[str, Any]) -> dict[str, Any]:
    """顾问批准后进入清洗。下一步的占位说明由理由汇总节点写成审核核对。"""
    return {"workflow_status": "policy_approved"}
