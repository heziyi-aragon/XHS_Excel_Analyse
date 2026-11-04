# 投放组合与建议定价接口

## 职责与主图接线

第9步消费审核交接，只安排候选达人、形式、篇数和建议价格，不修改上游评分、推荐理由、审核结论或轮次。
默认 `build_graph(enable_portfolio=True, portfolio_llm_client=None)` 执行 `finalize_audit → build_portfolio → END`。写建议、最终人工审核和导出仍待实现。
`enable_portfolio=False` 保留审核结束即停止的原流程。
核心入口：`build_portfolio_plan(state, config=None, *, llm_client=None, feedback=None) -> dict`。
`build_portfolio` 节点仅返回 `{'portfolio_plan': plan}`；主图使用 `make_portfolio_handoff_node` 更新当前流程交接。

## 输入契约

| 状态字段 | 类型与用途 |
| --- | --- |
| `brief`、`requirement_profile`、`active_policy` | dict，原始需求、分析及实际执行策略 |
| `pricing_candidates` | list，仅从此名单选择 `stance=recommend` 者 |
| `audited_talents`、`audit_pending` | list，完整审核名单和暂存问题 |
| `supplement_results`、`audit_unverified_checks` | list，补数来源及未核实事项 |
| `audit_summary` | dict，审核范围和计数 |
| `portfolio_config`、`portfolio_feedback` | 可选配置和本轮报价反馈 |

候选的 `score/stance/attempt/data_revision` 在外层，报价和效果在 `record['talent']`。
以 `xhs_id` 优先、昵称备用匹配身份；重复或冲突身份报错。`held_for_review` 者不参与组合。
`flagged_release` 可继续，但保留缺数和审核风险。缺一种形式的报价不影响另一种；全部可用形式缺价时保留为条件候选，不按0元入预算。

## 配置与修订来源

```json
{
  "price_multiplier": 1.0,
  "max_posts_per_creator": 1,
  "per_creator_max_posts": {},
  "per_creator_allowed_formats": {"82319044": ["video"]},
  "max_search_nodes": 100000
}
```

默认每人最多1篇是开发假设，图文和视频数量合计受此上限约束；历史商单数不代表未来产能。
`per_creator_allowed_formats` 按账号、`creator_key` 或昵称指定非空、无重复的 `image/video` 列表；指定形式无价时返回 `not_available_for_requested_format` 条件候选。
预算默认取 `brief.total_budget`。覆盖示例：`"budget": {"value": 100000, "source": "客户本轮明确确认"}`。
KPI读取 `brief.kpi`、原文核心KPI及 `requirement_profile.kpi`。可用 `targets` 显式覆盖 `exposure/interactions/cpe/brand_search_lift`，各项必须包含 `value` 和非空 `source`。
示例：`"targets": {"exposure": {"value": 25000000, "source": "需求单核心KPI"}}`。搜索提升使用比例，0.3表示30%。
`client_answers` 中的预算或KPI修订不会自动改写原 `brief`；应通过有来源的 `portfolio_config` 明确落实后再报价，不把模型推测当客户确认。

## 定价、预算与组合

默认平台参考价P对应区间 `[0.8P, 1.2P]`，建议价P；`price_multiplier` 调整建议价。超区间必须提供 `exception_reason`，仍待最终人工批准。
预算按各行价格区间上限乘篇数累加；超上限的例外建议价也纳入区间上界。内部使用Decimal及整数分，独立核查总上界不超过预算。
没有来源的费用不擅自添加；本需求的预算不含投流，其他费用应按实际需求口径处理。
`industry_benchmark_status=not_provided` 表示没有行业绝对报价，不能用相对成本排名编造市场价。
标准库有界搜索先偏好可同时满足可估数值KPI的组合，再比较曝光/互动归一缺口、整体CPE超额、上游分数、形式偏好和保守费用；每人分数只计一次。
`optimal` 只表示给定候选、篇数和效果代理下的最优。达到搜索限额返回 `limit_reached`，不证明最优或无解。
图文为主属于软偏好，没有固定视频占比。相同效果代理和较低图文价可能产生全图文方案；指定视频请使用形式配置，不编造视频增益。

## KPI与审批语义

曝光/互动使用历史中位数乘篇数，不是未来承诺或去重触达。数据不区分图文/视频效果，不额外假设效果增益。
整体CPE按总费用除以总预计互动，主要检查采用费用上界，另提供建议价口径；不平均历史CPE，也不要求每人单独满足整体上限。
缺效果时完整估算为空，另列已知部分。品牌搜索提升缺基线和模型时为 `not_evaluable`，不能以搜索页阅读占比替代。
`proxy_met` 仅表示效果代理满足目标，`verified_delivery=false`。`ready` 也不是审批通过，`approval_status` 始终为 `pending_final_human_review`。

## 输出与当前流程状态

`portfolio_plan` 包含 `schema_version/status/solver_status/lines/budget/kpi_checks`，以及条件候选、假设、警告、上游风险、最终配置、修订号和可选模型说明。
方案状态为 `ready/best_effort/no_candidates/needs_pricing`。无候选或全无价都返回明确状态，不伪造成功组合。
主图顶层 `workflow_status='portfolio_'+plan.status`；旧审核状态及交接分别保存在 `audit_workflow_status/audit_next_step`，审核明细保持不变。
顶层 `next_step` 指向 `write_suggestions`，`implemented=false`；有投放行时为 `interface_ready`，无投放行时为 `no_portfolio`。
主图遇到报价输入的 `ValueError` 时保留审核结果、保存 `portfolio_error`，并设置 `workflow_status=portfolio_needs_input`，不补造缺失预算或价格。
核心函数和独立只读节点仍对非法输入抛 `ValueError`，供调用方明确处理。

## 保存与独立运行

`main.save_result` 在第9步写 `output/step9_portfolio.json`，包含原审核字段和新增报价字段，不覆盖既有step8文件；审核停止模式仍写 `output/step8_audit.json`。
独立入口只读已完成审核的step8文件，另存带 `upstream_state` 的step9包裹结构和可选报告：

```bash
python -m scripts.build_portfolio output/step8_audit.json --output output/step9_r1.json --report output/step9_r1.md
```

`--config` 读取配置JSON；`--feedback` 读取反馈JSON；`--previous` 读取旧step9文件。每轮选新输出路径，不覆盖输入或已有结果。
修订顺序为旧方案 `configuration`、本轮显式配置、反馈 `config_overrides`；最终有效配置保存在新方案 `configuration`。
反馈示例：`{"comments":"尝试九折议价","config_overrides":{"price_multiplier":0.9}}`。自由文本只保留，不猜测执行。
只增加 `portfolio_plan.revision`，不修改上游 `attempt/data_revision`。最终人工审批仍由后续节点负责。

## 模型与验证边界

报价节点默认不增加模型调用，核心离线可运行。主图需显式注入 `portfolio_llm_client`，独立入口需显式 `--llm` 才启用辅助模型说明。
模型复用现有 `LLMClient`，只作有原文引用的约束翻译和方案解释；翻译为 `advisory_only`，不能改代码核验的预算、价格、候选身份或效果。
真实报价模型API尚未验证。离线固定回答和单元测试只能验证软件逻辑，不能证明模型准确率或实际投放结果。

```bash
python -m pytest tests/ -q
```