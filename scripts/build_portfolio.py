"""Read an existing step-8 JSON and write a separate step-9 proposal. Offline by default."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from agent.nodes.portfolio import make_portfolio_node


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(value, dict):
        raise ValueError(f'{path.name} must contain a JSON object')
    return value


def report_markdown(plan: dict) -> str:
    budget = plan['budget']
    rows = ['# 投放组合与建议定价', '',
            f"方案状态：`{plan['status']}`。求解状态：`{plan['solver_status']}`。版本：{plan['revision']}。",
            '', f"预算上限：¥{budget['limit']:,.2f}；建议总价：¥{budget['total_suggested']:,.2f}；区间上限总价：¥{budget['total_upper']:,.2f}。",
            '', '以下为待最终人工审核的方案，历史中位数用于估算，不代表实际投放承诺。', '',
            '| 达人 | 形式 | 篇数 | 平台单价 | 建议单价 | 单价区间 | 建议小计 |',
            '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for line in plan['lines']:
        name = str(line.get('nickname') or line['creator_key']).replace('|', '/')
        form = '图文' if line['format'] == 'image' else '视频'
        rows.append(f"| {name} | {form} | {line['quantity']} | {line['platform_reference_price']:,.2f} | {line['unit_price_suggested']:,.2f} | {line['unit_price_low']:,.2f}–{line['unit_price_high']:,.2f} | {line['line_cost_suggested']:,.2f} |")
    rows += ['', '| KPI | 目标 | 方案估算 | 状态 | 缺口 |', '| --- | ---: | ---: | --- | ---: |']
    labels = {'exposure': '曝光', 'interactions': '互动', 'cpe': '整体CPE（上限费用口径）', 'brand_search_lift': '品牌词搜索提升'}
    def fmt(value):
        return '无法评估' if value is None else f'{value:,.4f}'.rstrip('0').rstrip('.')
    for key, check in plan['kpi_checks'].items():
        rows.append(f"| {labels[key]} | {fmt(check['target'])} | {fmt(check['estimate'])} | {check['status']} | {fmt(check['gap'])} |")
    rows += ['', '搜索提升以比例记，0.3 表示 30%。CPE由总方案费用÷总预计互动计算。', '', '## 待核价及上游风险', '']
    rows += [f"- {item.get('nickname') or item['creator_key']}：{item['reason']}，未计入已承诺预算。" for item in plan['conditional_candidates']]
    risks = plan['upstream_risks']
    rows += [f"- 暂存问题 {len(risks.get('audit_pending') or [])} 项；待核实检查 {len(risks.get('audit_unverified_checks') or [])} 项。"]
    rows += ['', '## 警告', ''] + ['- ' + str(item) for item in plan['warnings']]
    rows += ['', '## 本方案假设', ''] + ['- ' + str(item) for item in plan['assumptions']]
    rows += ['', '完整上游审核、补数和未核实事项保存在同名 JSON，不能将本报告替代最终人工核实。', '']
    return '\n'.join(rows)


def run(input_path: Path, output_path: Path, *, config: dict | None = None,
        feedback=None, llm_client=None, report_path: Path | None = None) -> dict:
    if input_path.resolve() == output_path.resolve():
        raise ValueError('Step-9 output must not overwrite the step-8 input')
    if report_path is not None and report_path.resolve() in {input_path.resolve(), output_path.resolve()}:
        raise ValueError('Input, JSON output, and Markdown report must use different paths')
    if output_path.exists() or (report_path is not None and report_path.exists()):
        raise ValueError('Output already exists; choose a new output name for this revision')
    source = read_json(input_path)
    state = deepcopy(source)
    if config is not None:
        state['portfolio_config'] = config
    if feedback is not None:
        state['portfolio_feedback'] = feedback
    update = make_portfolio_node(llm_client)(state)
    # Use a new envelope: the exact upstream payload is evidence, never a mutable plan.
    result = {'upstream_state': source,
              'portfolio_config': update['portfolio_plan'].get('configuration', config or {}), **update}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report_markdown(update['portfolio_plan']), encoding='utf-8')
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='Existing output/step8_audit.json')
    parser.add_argument('--output', type=Path, default=Path('output/step9_portfolio.json'))
    parser.add_argument('--config', type=Path)
    parser.add_argument('--feedback', type=Path, help='JSON with config_overrides and/or comments')
    parser.add_argument('--previous', type=Path, help='Previous step-9 envelope, to increment only the portfolio revision')
    parser.add_argument('--report', type=Path)
    parser.add_argument('--llm', action='store_true', help='Explicitly enable optional model annotation with existing .env')
    args = parser.parse_args()
    try:
        config = read_json(args.config) if args.config else None
        feedback = read_json(args.feedback) if args.feedback else None
        client = None
        if args.llm:
            from agent.requirements import LLMClient
            client = LLMClient.from_env()
        if args.previous:
            # Preserve the step-8 source file by using an in-memory previous-plan field.
            previous_envelope = read_json(args.previous)
            previous = previous_envelope.get('portfolio_plan')
            if not isinstance(previous, dict):
                raise ValueError('Previous result has no portfolio_plan')
            inherited = deepcopy(previous.get('configuration') or previous_envelope.get('portfolio_config') or {})
            if config is not None:
                inherited.update(config)
            config = inherited
            source = read_json(args.input)
            source['portfolio_plan'] = previous
            if args.input.resolve() == args.output.resolve() or args.output.exists() or (args.report and args.report.exists()):
                raise ValueError('Choose new output paths; existing inputs and results are preserved')
            if args.report and args.report.resolve() in {args.input.resolve(), args.output.resolve()}:
                raise ValueError('Input, JSON output, and Markdown report must use different paths')
            state = deepcopy(source)
            if config is not None:
                state['portfolio_config'] = config
            if feedback is not None:
                state['portfolio_feedback'] = feedback
            update = make_portfolio_node(client)(state)
            result = {'upstream_state': read_json(args.input),
                      'portfolio_config': update['portfolio_plan'].get('configuration', config or {}), **update}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
            if args.report:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(report_markdown(result['portfolio_plan']), encoding='utf-8')
        else:
            result = run(args.input, args.output, config=config, feedback=feedback,
                         llm_client=client, report_path=args.report)
        plan = result['portfolio_plan']
        print(f"Saved {args.output}; status={plan['status']}; solver={plan['solver_status']}; upper_cost={plan['budget']['total_upper']:.2f}")
    except (ValueError, OSError, json.JSONDecodeError) as error:
        parser.exit(2, f'Portfolio error: {error}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
