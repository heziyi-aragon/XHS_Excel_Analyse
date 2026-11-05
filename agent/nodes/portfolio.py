"""Step-9 adapters: preserve audit evidence and hand the proposal to content advice."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from agent.portfolio import PortfolioPlanningError, build_portfolio_plan
from agent.portfolio_config import changed_patch, merge_config
from agent.state import GraphState


class PortfolioState(GraphState, total=False):
    """Backward-compatible standalone subgraph schema using the shared fields."""


def _validate_handoff(state: dict) -> None:
    required = {
        'brief': dict, 'requirement_profile': dict, 'active_policy': dict,
        'pricing_candidates': list, 'audited_talents': list, 'audit_pending': list,
        'supplement_results': list, 'audit_unverified_checks': list, 'audit_summary': dict,
    }
    invalid = [name for name, kind in required.items() if not isinstance(state.get(name), kind)]
    if invalid:
        raise ValueError('Incomplete step-8 handoff: ' + ', '.join(invalid))


def make_portfolio_node(llm_client: Any = None):
    def node(state: dict) -> dict:
        _validate_handoff(state)
        current = state.get('portfolio_config') or {}
        old = state.get('portfolio_plan') or state.get('portfolio_previous_plan') or {}
        if not isinstance(current, dict) or not isinstance(old, dict):
            raise ValueError('Portfolio configuration and previous plan must be mappings')
        patch = changed_patch(current, old.get('node_configuration') or {}) if old else deepcopy(current)
        if state.get('portfolio_config_patch') is not None:
            patch = merge_config(patch, state['portfolio_config_patch'])
        plan = build_portfolio_plan(state, patch, llm_client=llm_client,
                                    feedback=state.get('portfolio_feedback'))
        plan['node_configuration'] = deepcopy(current)
        return {'portfolio_plan': plan}
    return node


def build_portfolio(state: dict) -> dict:
    """Only the new result is returned; LangGraph retains upstream state unchanged."""
    return make_portfolio_node()(state)


def make_portfolio_handoff_node(llm_client: Any = None):
    """Main-graph adapter; validation failures preserve the completed audit for review."""
    node = make_portfolio_node(llm_client)

    def handoff(state: dict) -> dict:
        snapshot = {
            'audit_workflow_status': state.get('audit_workflow_status') or state.get('workflow_status'),
            'audit_next_step': deepcopy(state.get('audit_next_step') or state.get('next_step') or {}),
        }
        try:
            update = node(state)
        except ValueError as error:
            return {**snapshot, 'portfolio_plan': None,
                    'portfolio_previous_plan': deepcopy(state.get('portfolio_plan') or state.get('portfolio_previous_plan')),
                    'portfolio_error': {'type': 'needs_clarification' if isinstance(error, PortfolioPlanningError) else 'input_validation',
                                        'message': str(error), 'planning': getattr(error, 'planning', None)},
                    'workflow_status': 'portfolio_needs_input',
                    'next_step': {'name': 'build_portfolio', 'display_name': '投放组合，拟定价格',
                                  'implemented': True, 'status': 'needs_input',
                                  'message': '审核结果已保留；确认上述要求后重新运行本步骤。'}}
        plan = update['portfolio_plan']
        has_lines = bool(plan['lines'])
        message = ('组合草案已生成，须携带KPI缺口、缺数及未核实事项交给最终人工审核；写建议节点尚未实现。'
                   if has_lines else '当前没有可执行的已定价组合；候选及核价问题已保留，暂不能写投放建议。')
        return {**snapshot, **update, 'portfolio_error': None, 'portfolio_config_patch': None,
                'portfolio_previous_plan': None,
                'workflow_status': 'portfolio_' + plan['status'],
                'next_step': {'name': 'write_suggestions', 'display_name': '写建议',
                              'implemented': False,
                              'status': 'interface_ready' if has_lines else 'no_portfolio',
                              'required_state': ['portfolio_plan', 'audited_talents', 'audit_pending',
                                                 'supplement_results', 'audit_unverified_checks',
                                                 'audit_summary', 'brief', 'requirement_profile', 'active_policy'],
                              'message': message}}
    return handoff


def build_portfolio_graph(llm_client: Any = None):
    """Independently runnable graph for demonstration and later team integration."""
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(PortfolioState)
    graph.add_node('build_portfolio', make_portfolio_node(llm_client))
    graph.add_edge(START, 'build_portfolio')
    graph.add_edge('build_portfolio', END)
    return graph.compile()
