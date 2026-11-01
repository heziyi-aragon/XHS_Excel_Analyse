"""只读复查已有 step8 JSON；不调用 API，不覆盖原文件。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent.audit_reporting import audit_report_fields
from agent.reasons import person_key


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    state = json.loads(args.path.read_text(encoding="utf-8"))
    if not isinstance(state, dict) or not isinstance(state.get("audited_talents"), list):
        parser.error("输入必须是包含 audited_talents 的审核结果")
    report = audit_report_fields(state)
    summary = report["audit_summary"]
    print(summary["scope"])
    print(f"进入审核 {summary['entered_count']} 人；模型审校 {summary['model_reviewed_count']} 人；"
          f"明确档跳过模型审校 {summary['skipped_reason_review_count']} 人。")
    print(f"可用推荐 {summary['usable_recommendations']} 人；已处理不推荐 {summary['rejected_count']} 人；"
          f"暂存 {summary['held_count']} 人；数据不全放行 {summary['flagged_release_count']} 人。")
    for row in state["audited_talents"]:
        print(f"\n{row.get('nickname')}：{row.get('stance')}，{row.get('audit_status')}，理由版本 {row.get('attempt')}")
        if row.get("band") in ("clear_yes", "clear_no"):
            print("  明确档由评分代码给标签；必要数据仍检查，模型理由审校按流程跳过。")
        for entry in state.get("audit_history") or []:
            if person_key(entry["assessment"]) != person_key(row):
                continue
            print(f"  版本 {entry['assessment']['attempt']}：{entry['result']['verdict']}")
            print("    被审理由：" + str(entry["assessment"].get("sentence")))
            for issue in entry["result"]["issues"]:
                print("    " + issue["code"] + "：" + issue["message"])
        if row.get("audit_error"):
            print("  " + row["audit_error"]["message"])
    print("\n尚无核实证据的人工检查项：")
    for item in report["audit_unverified_checks"]:
        print("  · " + item["description"])
    if summary["policy_changed_after_approval"]:
        print("实际执行策略与获批策略不同；本报告不确认策略变更符合客户要求。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
