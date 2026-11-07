"""Receive downstream pricing feedback without rerunning upstream agents."""
import argparse
import json
from pathlib import Path
from uuid import uuid4

from langgraph.types import Command

from agent.nodes.portfolio_review import build_repricing_graph
from scripts.build_portfolio import load_portfolio_state, read_json, report_markdown


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='Step-9 flat state or standalone envelope')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--max-revisions', type=int, default=3)
    parser.add_argument('--llm', action='store_true')
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve() == args.input.resolve():
        parser.error('Choose a new output file; existing proposals are preserved')
    if args.report and (args.report.exists() or args.report.resolve() in {args.input.resolve(), args.output.resolve()}):
        parser.error('Choose a distinct new report path')
    client = None
    if args.llm:
        from agent.llm_routing import ModuleLLMClients
        client = ModuleLLMClients.from_env().portfolio
    source = read_json(args.input)
    graph = build_repricing_graph(llm_client=client, max_revisions=args.max_revisions)
    config = {'configurable': {'thread_id': 'reprice-' + uuid4().hex}, 'recursion_limit': 64}
    result = graph.invoke(load_portfolio_state(source), config=config)
    while result.get('__interrupt__'):
        plan = result.get('portfolio_plan') or result.get('portfolio_previous_plan') or {}
        if result.get('portfolio_error'):
            print('本轮意见需澄清：' + result['portfolio_error']['message'])
            print('以下显示上一次有效方案，新意见通过校验后才生成新版本。')
        print(f"报价版本 {plan.get('revision')}；建议总价 {(plan.get('budget') or {}).get('total_suggested')}。")
        answer = input('输入修改要求（如“打九折”“每人最多2篇”），或输入“继续”交给下游：\n> ').strip()
        if not answer:
            print('未收到报价意见，原文件未改。')
            return 0
        decision = ({'action': 'continue'} if answer.lower() in {'继续', 'continue'} else
                    {'action': 'revise', 'feedback': answer, 'review_id': uuid4().hex})
        result = graph.invoke(Command(resume=decision), config=config)
    result.pop('__interrupt__', None)
    if result.get('portfolio_error'):
        print('尚未形成新报价：' + result['portfolio_error']['message'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    if args.report and result.get('portfolio_plan'):
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report_markdown(result['portfolio_plan']), encoding='utf-8')
    print(f'已保存 {args.output}；此流程仅处理报价退回，不代表最终投放审批。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
