"""Step-9 adapters: preserve audit evidence and hand the proposal to content advice."""
from __future__ import annotations

from copy import deepcopy
import re
from typing import Any

from agent.portfolio import build_portfolio_plan
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
        _check_client_revisions(state)
        return {'portfolio_plan': build_portfolio_plan(
            state, state.get('portfolio_config'), llm_client=llm_client,
            feedback=state.get('portfolio_feedback'))}
    return node


def build_portfolio(state: dict) -> dict:
    """Only the new result is returned; LangGraph retains upstream state unchanged."""
    return make_portfolio_node()(state)


def _check_client_revisions(state: dict) -> None:
    """Do not silently price against old brief numbers after a numeric client reply."""
    answers = str(state.get('client_answers') or '')
    if not re.search(r'\d|[零一二三四五六七八九十百千万亿]', answers):
        return
    config = deepcopy(state.get('portfolio_config') or {})
    if not isinstance(config, dict):
        raise ValueError('portfolio_config must be a mapping')
    feedback = state.get('portfolio_feedback')
    if isinstance(feedback, dict) and isinstance(feedback.get('config_overrides'), dict):
        config.update(feedback['config_overrides'])
    if not isinstance(config.get('targets') or {}, dict):
        raise ValueError('portfolio_config.targets must be a mapping')
    unresolved = []
    if '预算' in answers and not isinstance(config.get('budget'), dict):
        unresolved.append('budget')
    for word, field in (('曝光', 'exposure'), ('互动', 'interactions'),
                        ('cpe', 'cpe'), ('搜索', 'brand_search_lift')):
        if word in answers.lower() and not isinstance((config.get('targets') or {}).get(field), dict):
            unresolved.append('targets.' + field)
    if unresolved:
        raise ValueError('客户补答涉及预算或KPI数字，请先在portfolio_config中提供带source的确认值：'
                         + ', '.join(unresolved))


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
                    'portfolio_error': {'type': 'input_validation', 'message': str(error)},
                    'workflow_status': 'portfolio_needs_input',
                    'next_step': {'name': 'build_portfolio', 'display_name': '投放组合，拟定价格',
                                  'implemented': True, 'status': 'needs_input',
                                  'message': '审核结果已保留；请修正组合输入后重新计算：' + str(error)}}
        plan = update['portfolio_plan']
        has_lines = bool(plan['lines'])
        message = ('组合草案已生成，须携带KPI缺口、缺数及未核实事项交给最终人工审核；写建议节点尚未实现。'
                   if has_lines else '当前没有可执行的已定价组合；候选及核价问题已保留，暂不能写投放建议。')
        return {**snapshot, **update, 'portfolio_error': None,
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
