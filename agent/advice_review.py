"""命令行最终人工审核与正式Excel导出。"""
from datetime import datetime, timezone
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

from agent.export_table import export_recommendations
from agent.nodes.advice import make_advice_node
from agent.nodes.portfolio import make_portfolio_handoff_node
from agent.runtime_logging import LOGGER, logged_node


def review_and_export(result: dict, output_dir: str | Path) -> bool:
    """明确输入y才批准当前JSON建议；不读取预览表中的手工修改。"""
    rows = result.get("recommendations")
    if not rows:
        return False
    result["review_status"] = "pending"
    print("\n请人工审核本轮报价、内容建议和待核实事项。")
    if result.get("advice_preview_path"):
        print(f"未审核预览表：{result['advice_preview_path']}")
    if result.get("advice_preview_error"):
        print(f"预览导出问题：{result['advice_preview_error']}")
    print("本次确认针对当前JSON数据；预览Excel中的手工修改不会自动回写。")
    if any(row.get("advice_status") == "failed" for row in rows):
        print("注意：部分建议生成失败，缺失内容将保留在表格中。")
    try:
        decision = input("请人工审核：y 通过并导出，n 打回重新定价，其他输入暂存：\n> ")
    except EOFError:
        decision = ""
    if decision.strip().lower() == "n":
        try:
            comments = input("请输入重新定价的修改意见：\n> ").strip()
        except EOFError:
            comments = ""
        if not comments:
            print("未提供修改意见，保留待审核状态，暂不重新定价。")
            return False
        result["portfolio_feedback"] = {"comments": comments, "feedback_id": uuid4().hex}
        result["review_status"] = "rejected"
        LOGGER.info("human_review.reject portfolio_revision=%s", (result.get("portfolio_plan") or {}).get("revision"))
        result.setdefault("human_review_history", []).append({
            "decision": "rejected", "source": "cli_explicit_n",
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "portfolio_revision": (result.get("portfolio_plan") or {}).get("revision"),
            "feedback": comments,
        })
        return True
    if decision.strip().lower() != "y":
        print("未确认通过，保留未审核预览，不导出正式表格。")
        return False
    result["review_status"] = "approved"
    LOGGER.info("human_review.approve portfolio_revision=%s", (result.get("portfolio_plan") or {}).get("revision"))
    result["human_review"] = {
        "decision": "approved", "source": "cli_explicit_y",
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "portfolio_revision": (result.get("portfolio_plan") or {}).get("revision"),
    }
    path = Path(output_dir) / f"recommendations_approved_{uuid4().hex}.xlsx"
    try:
        exported = export_recommendations(rows, path, review_status="approved")
    except (OSError, ValueError) as exc:
        result.update(workflow_status="export_failed", final_export_error=str(exc))
        print(f"审核确认已记录，但正式表格导出失败：{exc}")
        return False
    result.update(final_export_path=str(exported), final_export_error=None,
                  workflow_status="export_complete")
    result["next_step"] = {
        "name": "complete", "display_name": "审核通过，表格已导出",
        "implemented": True, "status": "complete",
    }
    print(f"人工审核已通过，正式表格已导出：{exported}")
    return False


def review_until_complete(result: dict, output_dir: str | Path, llm, *, save=None) -> None:
    """每轮报价交接后重新生成建议，直到人工确认或暂存。"""
    reprice = logged_node("build_portfolio", make_portfolio_handoff_node(llm))
    advise = logged_node("write_advice", make_advice_node(llm, output_dir=output_dir))
    while result.get("recommendations"):
        revise = logged_node("human_review", lambda state: review_and_export(state, output_dir))(result)
        if save is not None:
            save(result)
        if not revise:
            return
        previous = deepcopy(result["recommendations"])
        feedback = result["portfolio_feedback"]["comments"]
        result.setdefault("advice_history", []).append({
            "portfolio_revision": (result.get("portfolio_plan") or {}).get("revision"),
            "recommendations": deepcopy(previous),
            "preview_path": result.get("advice_preview_path"),
            "pricing_feedback": feedback,
            "reviewed_at": result["human_review_history"][-1]["reviewed_at"],
        })
        # 原文件保留，但新一轮不得沿用旧建议、审批或正式交付路径。
        for field in ("recommendations", "human_review", "final_export_path",
                      "final_export_error", "advice_preview_path", "advice_preview_error"):
            result.pop(field, None)
        result["review_status"] = "pending"
        if save is not None:
            save(result)
        print("正在按修改意见重新定价；原评分和审核记录保持不变。")
        result.update(reprice(result))
        if result.get("portfolio_error") or not (result.get("portfolio_plan") or {}).get("lines"):
            print("新报价暂不能交接写建议：" + result["next_step"]["message"])
            if save is not None:
                save(result)
            return
        result.update(advise(result))
        if save is not None:
            save(result)
