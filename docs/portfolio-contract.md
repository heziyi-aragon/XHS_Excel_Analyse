# 投放组合、定价与重定价接口

## 职责与接线

第9步只消费审核交接中的推荐候选，安排达人、形式、篇数和价格，不修改前序评分、理由、审核结论或轮次。
默认 `build_graph(enable_portfolio=True, portfolio_llm_client=None)` 接通审核、报价和后续交接；`enable_portfolio=False` 保留审核停止模式。
核心入口为 `build_portfolio_plan(state, config=None, *, llm_client=None, feedback=None)`；领域定价工具为 `price_for_creator(record, talent, form, config, state)`。
`build_portfolio` 只返回 `portfolio_plan` 增量；主图的 `make_portfolio_handoff_node` 同时更新当前流程状态并保留审核快照。
写内容建议、最终投放审批和导出仍由后续模块负责；报价复核接口只负责接收改价意见并返回本节点。

## 输入与配置

必需交接字段：`brief/requirement_profile/active_policy/audit_summary` 为dict；`pricing_candidates/audited_talents/audit_pending/supplement_results/audit_unverified_checks` 为list。
候选的分数、倾向、轮次在外层，报价和效果在 `record['talent']`。只选 `stance=recommend` 者，排除 `held_for_review`，保留 `flagged_release` 的缺数风险。
缺一种形式报价不影响另一种；全部允许形式均无价则进入条件候选，不按0元入预算，也不用行业价冒充缺失的平台锚点。

| 配置 | 规则 |
| --- | --- |
| `max_posts_per_creator` | 默认1，属于开发假设；同人图文与视频合计受限 |
| `per_creator_max_posts` | 按账号、creator_key或昵称设置单人上限 |
| `per_creator_allowed_formats` | 非空、无重复的image/video列表，指定形式无价时保留条件候选 |
| `price_multiplier` | 仅显式提供时覆盖差异化建议价，按平台价乘倍率 |
| `exception_reason` | 建议价超出平台±20%时必需，仍待最终人工批准 |
| `max_search_nodes` | 默认100000，达到限额不声称最优或无解 |
| `budget`、`targets` | 显式覆盖需value和非空source；targets含曝光、互动、CPE、品牌搜索提升 |
| `industry_benchmarks`、`pricing_as_of` | 有来源的可比报价观察及显式日期锚点 |

预算默认来自 `brief.total_budget`；KPI来自需求原文或有来源的配置。`targets`键为 `exposure/interactions/cpe/brand_search_lift`，搜索提升0.3表示30%。
客户补答中的明确修订由求解前规划处理；不能解析或相互矛盾时需要澄清，也可通过带source的明确配置落实，不回退旧数字冒充已处理。

## 行业观察与同行参考

`industry_benchmarks` 是观察记录列表。下列仅展示一条schema，数值为构造示例，不能充当真实行情：

```json
{
  "id": "demo-observation-1", "creator_id": "demo-peer-1",
  "content_type": "Beauty", "tier": "T3", "format": "image", "price": 10000,
  "source": "接口构造示例，非真实市场报价", "as_of": "2026-11-01", "synthetic": true
}
```

每条必须有唯一观察ID、内容类目、层级、image/video形式、正数价格、非空source及ISO日期as_of；creator_id可选，用于排除本人和重复达人。构造数据必须标synthetic=true。
仅匹配同类目、同层级、同形式；去重观察及已知达人身份，排除本人。至少3个独立可比观察，才采用IQR异常处理后保留样本的中位数及Q1–Q3区间。
真实观察由调用方提供；工具保留来源、日期、样本及排除原因，但不独立核实来源真实性。synthetic样本标为industry_fixture，不能显示为真实市场基准。
外部样本不足时，可使用 `state.talents` 中至少3位可比同行的平台价；状态为sample_peer_reference，明确是模拟输入的内部参考，非现实行业行情。
仍不足时回退平台走廊并标platform_fallback，不制造行业价。支持基准输入不等于已经取得真实行业数据或完成实盘估值。
日期优先config.pricing_as_of，其次state.pricing_as_of；只有显式锚点才排除未来观察。不读宿主日期，无锚点时标as_of_not_provided。

## 定价、预算与效果代理

有可靠可比样本时，以中位数确定逐人建议价，价格区间取Q1–Q3与平台 `[0.8P,1.2P]` 的交集。无交集时取最近的平台边界，保留conflict和解释供复核。
显式price_multiplier优先；若超出基准交集，显示区间扩展至包含建议价。超平台走廊另需exception_reason，不自动视为成交。
预算采用每行区间上限乘篇数累加，例外建议价也纳入上界；代码用Decimal及整数分独立核查，不依赖模型算术。
组合先偏好可同时满足可估数值KPI的方案，再比较曝光/互动归一缺口、整体CPE超额、每人一次的上游分数、软形式偏好及费用。
曝光/互动按历史中位数乘篇数估算，不是未来承诺或去重触达。不发明视频增益，也不把过去商单数当未来产能。
CPE按总费用上界除以总预计互动检查，另提供建议价口径；不平均历史CPE。缺效果保留空值与已知部分。
品牌搜索缺基线及模型时保持not_evaluable；搜索页阅读占比不能替代品牌词搜索指数。proxy_met不代表实际交付，verified_delivery=false。

## 求解前规划与模型边界

规划先读取brief、client_answers和本轮反馈，将支持的自然语言篇数、形式、折扣、预算及KPI修订转换成结构化配置，再调用定价和组合工具。
注入模型时，模型必须返回path/value/source_quote/source_kind；代码校验路径、候选身份、原文引用和可独立解析的值，拒绝编造事实、无依据数字及未知身份。
明确配置优先；顾问自由文本不能擅自放宽甲方预算/KPI。无法执行的意见保留pending，主图报告needs_clarification，不假装已重定价。
规划的accepted/rejected/pending及执行阶段写入结果。模型失效时可执行的确定性规则仍可运行，未解决意见须明确处理。
模型格式错误、错误数值、假引文等只进入rejected，不单独构成业务阻断；原始需求、客户补答或真实报价反馈中的未解要求才进入pending。因此首轮资料完整时，拒绝模型建议后仍按已核验条件生成组合，CLI提示回退情况，JSON保留原始诊断。
重复预算提案只能用原始“总预算”字段的完整金额核对，并与brief.total_budget一致；已知值记为accepted/action=retained_existing，不覆盖显式配置或后续客户修订。其他价格、金额片段不能冒充预算；原始预算与结构化预算明确冲突时暂停核对。
“允许上下浮动20%”是价格走廊，不是统一乘1.2的指令。否定、历史、条件或价格上下界的提及不作为已选倍率；引用片段还须通过完整来源上下文核对。暂不支持的明确视频篇数、形式占比或达人人数硬要求进入pending，软形式偏好不升级为硬配额。
库函数默认不隐式启用报价模型，可离线运行；`main.py` 命令行复用已配置客户端进行报价规划。独立入口用--llm显式启用。
新增报价规划及定价尚未运行真实模型API；离线测试不能证明真实模型准确率、行情真实性或投放结果。

## 修订、幂等与报价复核回路

重定价深合并旧方案configuration、本轮新patch及反馈config_overrides；嵌套字典保留未修改的兄弟项，列表按新值替换，避免改篇数时撤销旧折扣。
节点区分未变化的旧portfolio_config与本轮patch，支持portfolio_config_patch。自然语言反馈与结构化反馈均先校验后执行。
feedback_id/review_id用于幂等：同反馈标识、同输入及同有效配置重复提交不增加版本。真实新修订增加revision，并记录revision_history中的前轮配置、预算、反馈和指纹。
主图可传 `portfolio_review_node` 回调，返回 `action=revise` 及feedback后只回到报价；continue表示交接下游，approve兼容值也不代表最终投放审批。
回调可返回 `{'portfolio_review': {'action':'revise','review_id':'r2','feedback':{'comments':'按九折重新报价'}}}`；max_portfolio_revisions限制回路次数。
独立 `build_repricing_graph` 默认通过interrupt暂停，调用方用 `Command(resume=...)` 提交同样的决策；原审核、评分和attempt不重跑。
最终方案approval_status仍为pending_final_human_review；本接口没有实现第11步的最终人工审批。

## 状态、保存与运行

portfolio_plan保留方案、预算、KPI、逐行valuation、planning、有效配置、修订历史及完整上游风险；status为ready/best_effort/no_candidates/needs_pricing。
顶层workflow_status为portfolio_加方案状态，旧审核状态与交接保存在audit_workflow_status/audit_next_step。报价输入或规划错误保存portfolio_error及原审核，不乱补预算或报价。
需要澄清时，控制台给出中文原因和可操作示例，避免重复打印内部字典；完整planning及上一轮有效报价仍保留在状态中。成功回退不改变最终投放仍需人工审批的边界。
成功交接指向尚未实现的write_suggestions；有投放行为interface_ready，无投放行为no_portfolio。报价复核回路可在交接前请求新修订。
main保存 `output/step9_portfolio.json`，不覆盖旧step8文件；审核停止模式仍保存step8。独立工具另存新文件，保留upstream_state。

```bash
python -m scripts.build_portfolio output/step8_audit.json --output output/step9_r1.json --benchmarks input.json --pricing-as-of 2026-11-04
python -m scripts.reprice_portfolio output/step9_portfolio.json --output output/step9_r2.json
python -m pytest tests/ -q
```

build_portfolio的--config/--feedback/--previous分别读取配置、反馈和旧方案；--report可写可读报告。benchmarks文件包含观察列表，构造记录须标synthetic。
每轮使用新的输出路径；已保存的评分、理由、审核记录及原始输入不因改价而覆盖。
