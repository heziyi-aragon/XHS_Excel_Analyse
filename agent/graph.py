"""组装筛选、审核与组合定价，并在需求阶段支持暂停/恢复。"""
from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agent.nodes.audit import AuditNodes
from agent.nodes.audit_handoff import finalize_audit
from agent.nodes.evaluation import EvaluationNodes, collect_reasons, route_reasons
from agent.nodes.load_inputs import load_inputs
from agent.nodes.portfolio import make_portfolio_handoff_node
from agent.nodes.rewrite import prepare_reason_rewrites, route_after_audit
from agent.nodes.supplement import route_after_supplement, supplement_data
from agent.nodes.requirements import (
    RequirementNodes,
    prepare_next_step,
    route_after_policy_review,
    route_after_requirement_analysis,
)
from agent.nodes.screening import (
    preprocess_table,
    relax_screening,
    route_after_screen,
    score_talents,
    screen_talents,
)
from agent.requirements import LLMClient
from agent.state import GraphState


def build_graph(
    llm_client: LLMClient | None = None,
    checkpointer=None,
    *,
    audit_llm_client: LLMClient | None = None,
    enable_portfolio: bool = True,
    portfolio_llm_client: LLMClient | None = None,
):
    """默认继续组合；审核可独立运行，组合说明模型仅在显式注入时调用。"""
    g = StateGraph(GraphState)
    llm = llm_client or LLMClient.from_env()
    requirement_nodes = RequirementNodes(llm)
    evaluation_nodes = EvaluationNodes(llm)
    audit_nodes = AuditNodes(audit_llm_client if audit_llm_client is not None else llm)

    g.add_node("load_inputs", load_inputs)
    g.add_node("analyze_requirements", requirement_nodes.analyze_requirements)
    g.add_node(
        "request_client_confirmation",
        requirement_nodes.request_client_confirmation,
    )
    g.add_node("build_screening_policy", requirement_nodes.build_screening_policy)
    g.add_node("review_policy", requirement_nodes.review_policy)
    g.add_node("prepare_next_step", prepare_next_step)
    g.add_node("preprocess_table", preprocess_table)
    g.add_node("screen_talents", screen_talents)
    g.add_node("relax_screening", relax_screening)
    g.add_node("score_talents", score_talents)
    g.add_node("write_one_reason", evaluation_nodes.write_one_reason)
    g.add_node("collect_reasons", collect_reasons)
    g.add_node("audit_reasons", audit_nodes.audit_reasons)
    g.add_node("prepare_reason_rewrites", prepare_reason_rewrites)
    g.add_node("finalize_audit", finalize_audit)
    g.add_node("supplement_data", supplement_data)
    if enable_portfolio:
        g.add_node("build_portfolio", make_portfolio_handoff_node(portfolio_llm_client))

    g.add_edge(START, "load_inputs")
    g.add_edge("load_inputs", "analyze_requirements")
    g.add_conditional_edges(
        "analyze_requirements",
        route_after_requirement_analysis,
        {
            "client_confirmation": "request_client_confirmation",
            "build_policy": "build_screening_policy",
        },
    )
    g.add_edge("request_client_confirmation", "analyze_requirements")
    g.add_edge("build_screening_policy", "review_policy")
    g.add_conditional_edges(
        "review_policy",
        route_after_policy_review,
        {
            "approved": "prepare_next_step",
            "revise": "build_screening_policy",
        },
    )
    g.add_edge("prepare_next_step", "preprocess_table")
    g.add_edge("preprocess_table", "screen_talents")
    g.add_conditional_edges(
        "screen_talents",
        route_after_screen,
        {
            "score": "score_talents",
            "relax": "relax_screening",
        },
    )
    g.add_edge("relax_screening", "screen_talents")
    g.add_conditional_edges("score_talents", route_reasons)
    g.add_edge("write_one_reason", "collect_reasons")
    g.add_edge("collect_reasons", "audit_reasons")
    g.add_conditional_edges(
        "audit_reasons", route_after_audit,
        {"rewrite": "prepare_reason_rewrites", "supplement": "supplement_data", "finish": "finalize_audit"},
    )
    g.add_conditional_edges("prepare_reason_rewrites", route_reasons)
    g.add_conditional_edges("supplement_data", route_after_supplement)
    if enable_portfolio:
        g.add_edge("finalize_audit", "build_portfolio")
        g.add_edge("build_portfolio", END)
    else:
        g.add_edge("finalize_audit", END)

    return g.compile(checkpointer=checkpointer or InMemorySaver())
