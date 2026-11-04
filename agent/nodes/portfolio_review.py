"""A bounded return-to-pricing adapter for the team's downstream human review.

No copywriting, final campaign approval or export is implemented here.  The team
can pass its review callback; the standalone adapter otherwise pauses for input.
"""
from copy import deepcopy

from agent.nodes.portfolio import PortfolioState, make_portfolio_handoff_node


def portfolio_feedback_interrupt(state: dict) -> dict:
    from langgraph.types import interrupt
    response = interrupt({'type': 'portfolio_feedback',
                          'portfolio_plan': state.get('portfolio_plan') or state.get('portfolio_previous_plan'),
                          'requires_correction': bool(state.get('portfolio_error')),
                          'message': '价格方案可继续交下游，或提交改价/篇数/形式意见。此操作不代表最终投放审批。'})
    if not isinstance(response, dict):
        response = {'action': 'revise', 'feedback': str(response)}
    return {'portfolio_review': response}


def make_review_adapter(review_node=None, max_revisions: int = 3):
    if type(max_revisions) is not int or max_revisions < 1:
        raise ValueError('max_revisions must be a positive integer')
    callback = review_node or portfolio_feedback_interrupt

    def review(state: dict) -> dict:
        view = deepcopy(state)
        if not view.get('portfolio_plan') and view.get('portfolio_previous_plan'):
            view['portfolio_plan'] = deepcopy(view['portfolio_previous_plan'])
        output = callback(view)
        decision = output.get('portfolio_review', output) if isinstance(output, dict) else None
        if not isinstance(decision, dict) or decision.get('action') not in {'revise', 'continue', 'approve'}:
            raise ValueError('Portfolio review must return action revise or continue')
        history = deepcopy(state.get('portfolio_review_history') or [])
        identity = decision.get('review_id') or decision.get('feedback_id')
        if identity is not None and any(item.get('review_id') == str(identity) for item in history):
            return {'portfolio_review': deepcopy(decision), 'portfolio_review_status': 'duplicate_feedback_ignored'}
        entry = {'review_id': str(identity) if identity is not None else None,
                 'revision': (state.get('portfolio_plan') or state.get('portfolio_previous_plan') or {}).get('revision'),
                 'action': decision['action'], 'feedback': deepcopy(decision.get('feedback'))}
        history.append(entry)
        base = {'portfolio_review': deepcopy(decision), 'portfolio_review_history': history}
        if decision['action'] in {'continue', 'approve'}:
            if state.get('portfolio_error'):
                return {**base, 'portfolio_review_status': 'needs_clarification'}
            return {**base, 'portfolio_review_status': 'accepted_for_handoff'}
        if (state.get('portfolio_plan') or state.get('portfolio_previous_plan') or {}).get('revision', 0) >= max_revisions:
            return {**base, 'portfolio_review_status': 'revision_limit_reached',
                    'workflow_status': 'portfolio_revision_limit_reached',
                    'next_step': {'name': 'build_portfolio', 'display_name': '重新定价',
                                  'implemented': True, 'status': 'needs_review',
                                  'message': '已到本次自动改价轮数上限；保留历史和最新草案，等待明确的新执行安排。'}}
        feedback = decision.get('feedback')
        if isinstance(feedback, str) and feedback.strip():
            feedback = {'comments': feedback.strip()}
        if not isinstance(feedback, dict) or not feedback:
            raise ValueError('A revise action requires pricing feedback')
        feedback = deepcopy(feedback)
        if identity is not None:
            feedback['feedback_id'] = str(identity)
        return {**base, 'portfolio_feedback': feedback,
                'portfolio_review_status': 'reprice_requested'}
    return review


def route_after_portfolio(state: dict) -> str:
    return 'end' if state.get('portfolio_error') or not state.get('portfolio_plan') else 'review'


def route_after_review(state: dict) -> str:
    return 'reprice' if state.get('portfolio_review_status') == 'reprice_requested' else 'end'


def attach_review_loop(graph, *, review_node=None, max_revisions=3):
    from langgraph.graph import END
    graph.add_node('portfolio_review', make_review_adapter(review_node, max_revisions))
    graph.add_conditional_edges('build_portfolio', route_after_portfolio,
                                {'review': 'portfolio_review', 'end': END})
    graph.add_conditional_edges('portfolio_review', route_after_review,
                                {'reprice': 'build_portfolio', 'end': END})


def build_repricing_graph(review_node=None, llm_client=None, *, max_revisions=3, checkpointer=None):
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph
    graph = StateGraph(PortfolioState)
    graph.add_node('portfolio_entry', lambda state: {})
    graph.add_node('build_portfolio', make_portfolio_handoff_node(llm_client))
    graph.add_edge(START, 'portfolio_entry')
    graph.add_conditional_edges('portfolio_entry', lambda state: 'review' if state.get('portfolio_plan') or state.get('portfolio_previous_plan') else 'build',
                                {'review': 'portfolio_review', 'build': 'build_portfolio'})
    attach_review_loop(graph, review_node=review_node, max_revisions=max_revisions)
    return graph.compile(checkpointer=checkpointer or InMemorySaver())
