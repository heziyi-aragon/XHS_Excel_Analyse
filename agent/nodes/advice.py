"""第十步节点：读取投放方案并返回内容建议。"""
from pathlib import Path
from uuid import uuid4

from agent.advice import AdviceAgent
from agent.export_table import export_recommendations
from agent.requirements import LLMClient


PREVIEW_DIR = Path(__file__).resolve().parents[2] / "output"


def make_advice_node(llm: LLMClient, *, output_dir: str | Path | None = None):
    agent = AdviceAgent(llm)

    def write_advice(state: dict) -> dict:
        recommendations = agent.generate_from_portfolio(
            brief=state["brief"],
            portfolio_plan=state["portfolio_plan"],
            audited_talents=state["audited_talents"],
        )
        failed = any(
            item["advice_status"] == "failed"
            for item in recommendations
        )
        preview_path = None
        preview_error = None
        if recommendations:
            directory = Path(output_dir) if output_dir is not None else PREVIEW_DIR
            path = directory / f"step10_advice_review_preview_{uuid4().hex}.xlsx"
            try:
                preview_path = str(export_recommendations(
                    recommendations, path, review_status="pending", preview=True
                ))
            except (OSError, ValueError) as exc:
                # 导出失败仍返回已生成的建议，避免丢失需人工处理的内容。
                preview_error = str(exc)
        return {
            "recommendations": recommendations,
            "advice_preview_path": preview_path,
            "advice_preview_error": preview_error,
            "workflow_status": (
                "advice_needs_review" if failed or preview_error else "advice_ready"
            ),
            "next_step": {
                "name": "human_review",
                "display_name": "最终人工审核",
                "implemented": True,
                "status": "interface_ready",
                "required_state": [
                    "recommendations",
                    "advice_preview_path",
                    "advice_preview_error",
                    "portfolio_plan",
                    "audited_talents",
                    "audit_pending",
                    "supplement_results",
                    "audit_unverified_checks",
                    "audit_summary",
                ],
                "message": (
                    "预览表导出失败，建议已保留，请修正导出问题并人工审核。"
                    if preview_error
                    else "部分建议生成失败，请人工补充并审核。"
                    if failed
                    else "建议已生成，等待最终人工审核。"
                ),
            },
        }

    return write_advice


def route_to_advice(state: dict) -> str:
    """有可执行投放项且无报价错误时，进入写建议。"""
    if state.get("portfolio_error"):
        return "end"

    plan = state.get("portfolio_plan")
    if not isinstance(plan, dict) or not plan.get("lines"):
        return "end"

    return "advice"
