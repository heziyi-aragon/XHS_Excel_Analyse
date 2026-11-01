r"""人工触发的真实 API 联调；不由 pytest 收集，不使用客户表中的达人。

运行：.\.venv\Scripts\python.exe -m scripts.audit_smoke
预期至少三次逻辑调用：审核错误理由、重写、再审核。格式纠正可能增加调用。
报告写入已被 Git 忽略的 output/audit_smoke.json，不保存 API Key。
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import platform
import time
from urllib.parse import urlsplit

from langgraph.graph import END, START, StateGraph

from agent.nodes.audit import AuditNodes
from agent.nodes.audit_handoff import finalize_audit
from agent.nodes.evaluation import EvaluationNodes, collect_reasons, route_reasons
from agent.nodes.rewrite import prepare_reason_rewrites, route_after_audit
from agent.nodes.supplement import route_after_supplement, supplement_data
from agent.requirements import LLMClient
from agent.state import GraphState


BASE = Path(__file__).resolve().parents[1]


def synthetic_state() -> dict:
    """800 故意写成 99999；全部为联调构造数据，不代表真实达人或行业基准。"""
    talent = {
        "nickname": "联调模拟达人", "xhs_id": "audit-smoke-only", "mcn": None,
        "is_new_kol": "Y", "content_type": "Beauty", "tier": "T3", "main_format": "图文为主",
        "price_image": 8000, "price_video": 10000, "fans_wan": 10,
        "ratio_female": 0.9, "ratio_age_18_24": 0.45, "ratio_age_over_44": 0.05,
        "ctr": 0.2, "cpv": 0.8, "cpe": 10, "cpuv": 10,
        "exposure_med": 50000, "read_med": 10000, "interact_med": 800,
        "visit_store_med": 800, "ratio_discover": 0.7, "ratio_search": 0.2, "ads_count_90d": 8,
    }
    parts = [
        {"field": "visit_store_med", "operator": "maximize", "label": "进店中位数", "raw": 800,
         "normalized": 0.8, "weight": 0.6, "weight_effective": 0.6, "contribution": 0.48},
        {"field": "cpuv", "operator": "minimize", "label": "进店成本", "raw": 10,
         "normalized": 0.2, "weight": 0.4, "weight_effective": 0.4, "contribution": 0.08},
    ]
    record = {
        "xhs_id": talent["xhs_id"], "nickname": talent["nickname"], "band": "mid", "score": 0.56,
        "score_parts": parts, "missing_fields": [], "talent": talent, "attempt": 1,
        "stance": "recommend", "tag": "进店好",
        "sentence": "进店中位数达到99999，但进店成本还有优化空间。",
        "error": None, "source": "smoke_fixture", "evidence": {"visit_store_med": 800, "cpuv": 10},
    }
    criteria = [
        {"field": "visit_store_med", "operator": "maximize", "hard": False, "weight": 0.6,
         "reason": "本次模拟需求优先比较进店量"},
        {"field": "cpuv", "operator": "minimize", "hard": False, "weight": 0.4,
         "reason": "本次模拟需求兼顾进店成本"},
    ]
    policy = {"name": "单人联调策略", "criteria": criteria, "manual_checks": []}
    return {
        "brief": {"raw": {
            "投放目标": "依据进店量与进店成本比较候选，不在单人阶段承诺总目标达成",
            "达人要求": "按给定进店指标权衡，无额外账号排除项", "合规要求": "不得编造数据或承诺效果",
        }, "compliance": "不得编造数据或承诺效果"},
        "requirement_profile": {
            "objective": "比较进店表现", "kpi": "进店量和进店成本", "hard_requirements": [],
            "soft_preferences": ["优先考虑进店量，兼顾进店成本"], "exclusions": [],
        },
        "approved_policy": policy, "active_policy": deepcopy(policy),
        "scored_talents": [deepcopy(record)], "evaluated_talents": [deepcopy(record)],
        "reason_results": [{name: record[name] for name in (
            "xhs_id", "nickname", "stance", "tag", "sentence", "evidence", "attempt", "error", "source"
        )}],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-seconds", type=float, default=45,
                        help="仅本次联调的单次请求超时，默认 45 秒，不修改 .env")
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error("timeout 必须大于 0")

    report = {"synthetic_data_only": True, "python_version": platform.python_version(), "calls": []}
    live = None
    try:
        live = LLMClient.from_env()
        if not live.api_key or not live.base_url or not live.model:
            raise ValueError("本地 API 配置不完整")
        live.timeout_seconds = args.timeout_seconds
        report.update({"model": live.model, "endpoint_host": urlsplit(live.base_url).hostname,
                       "timeout_seconds": live.timeout_seconds})

        def transport(system, payload):
            stage = "audit" if "assessment" in payload["candidate"] else "rewrite"
            event = {"stage": stage, "status": "started"}
            report["calls"].append(event)
            print(f"Starting {stage} request {len(report['calls'])}...", flush=True)
            started = time.monotonic()
            try:
                answer = live.complete_json(system, payload)
            except Exception:
                event["status"] = "failed"
                raise
            else:
                event["status"] = "response_received"
                return answer
            finally:
                event["elapsed_seconds"] = round(time.monotonic() - started, 2)

        client = LLMClient(transport=transport)
        audit, evaluation = AuditNodes(client), EvaluationNodes(client)
        graph = StateGraph(GraphState)
        graph.add_node("audit_reasons", audit.audit_reasons)
        graph.add_node("prepare_reason_rewrites", prepare_reason_rewrites)
        graph.add_node("write_one_reason", evaluation.write_one_reason)
        graph.add_node("collect_reasons", collect_reasons)
        graph.add_node("supplement_data", supplement_data)
        graph.add_node("finalize_audit", finalize_audit)
        graph.add_edge(START, "audit_reasons")
        graph.add_conditional_edges("audit_reasons", route_after_audit,
                                    {"rewrite": "prepare_reason_rewrites", "supplement": "supplement_data", "finish": "finalize_audit"})
        graph.add_conditional_edges("prepare_reason_rewrites", route_reasons)
        graph.add_conditional_edges("supplement_data", route_after_supplement)
        graph.add_edge("finalize_audit", END)
        graph.add_edge("write_one_reason", "collect_reasons")
        graph.add_edge("collect_reasons", "audit_reasons")
        result = graph.compile().invoke(synthetic_state(), config={"recursion_limit": 16})
        history = result["audit_history"]
        first = history[0]["result"] if history else {}
        latest = result["evaluated_talents"][0]
        final = result["audit_results"][0] if result["audit_results"] else {}
        checks = {
            "caught_invented_visit_number": first.get("verdict") == "reject" and any(
                issue["code"] == "data_conflict" and "visit_store_med" in issue["fields"]
                for issue in first.get("issues", [])
            ),
            "rewritten_exactly_once": latest["attempt"] == 2,
            "rewrite_generated_successfully": not latest.get("error") and bool(latest.get("sentence")),
            "invented_number_removed": "99999" not in (latest.get("sentence") or ""),
            "final_audit_passed": final.get("verdict") == "pass",
            "no_pending_tasks": not result["audit_pending"],
        }
        report.update({"checks": checks, "success": all(checks.values()), "audit_history": history,
                       "final_assessment": latest, "pending": result["audit_pending"],
                       "workflow_status": result["workflow_status"]})
    except Exception as exc:
        # 不打印异常正文或请求头，防止服务端回显认证信息；状态码与类型足够定位第一层故障。
        report["success"] = False
        causes, cause = [], exc
        while cause is not None and len(causes) < 5:
            causes.append(type(cause).__name__)
            cause = cause.__cause__
        report["error"] = {"type": type(exc).__name__, "http_status": getattr(exc, "status_code", None),
                           "cause_types": causes}

    output = BASE / "output" / "audit_smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if live is not None and live.api_key:
        text = text.replace(live.api_key, "[REDACTED]")
    output.write_text(text, encoding="utf-8")
    summary = {name: report[name] for name in ("success", "checks", "error", "calls") if name in report}
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    print(f"Report saved to {output}")
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
