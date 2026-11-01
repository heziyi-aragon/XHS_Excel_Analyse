# 第8步审核模块交接说明

## 1. 新增文件逐项说明与保留决定

“运行必需”表示当前审核功能会直接使用；“验收保留”表示主程序运行时无需加载，但接收方需要它验证和维护模块；“辅助保留”表示独立工具，可以不部署到生产环境。

### 运行模块：8个，均保留

| 文件 | 解决什么问题 | 必要性 |
| --- | --- | --- |
| `agent/audit.py` | 单人审核业务：组织原始指标和需求，调用审核模型，合并代码缺数结果并校验模型指控。它不修改被审理由 | 运行必需 |
| `agent/audit_contract.py` | 定义审核结果和问题类型，严格检查JSON结构，防止模型改写倾向、身份、标签或理由 | 运行必需 |
| `agent/audit_rules.py` | 不调用模型即可检查非法标签、理由生成失败、必要字段缺失；为所有档位查缺数 | 运行必需 |
| `agent/audit_reporting.py` | 汇总审核人数、模型审校覆盖、可用推荐和暂存数量；保留尚无证据的人工检查项 | 运行必需，末端交接与控制台共用 |
| `agent/nodes/audit.py` | 把单人审核接入整批状态：分档处理、匹配版本、复用未变化结果、保存历史并创建待处理任务 | 运行必需 |
| `agent/nodes/rewrite.py` | 决定下一步是重写、补数还是交接；只把被打回且未到上限的中间档送回原评估模块 | 运行必需 |
| `agent/nodes/supplement.py` | 按身份查回缺失值，更新工作副本，在同批候选中重新评分并返回评估；隔离新旧数据版本 | 运行必需 |
| `agent/nodes/audit_handoff.py` | 结束自动处理，区分正常推荐、缺数放行和暂存记录，生成下游交接状态；包含同批输入查值函数 | 运行必需 |

`agent/audit.py` 管“一位达人的审核判断”，`agent/nodes/audit.py` 管“整批名单在图中如何流转”，两者不是重复实现。结构校验、确定性规则、模型判断各自独立，接收方可以分别测试和替换。

### 辅助脚本：3个，均保留

| 文件 | 解决什么问题 | 必要性 |
| --- | --- | --- |
| `scripts/audit_acceptance.py` | 真实模型语义验收：预先写定正确答案，支持重复运行、数值变体、自定义题库，记录误报、漏放和不稳定 | 验收保留；不会由主程序或pytest自动调用真实API |
| `scripts/audit_smoke.py` | 真实API执行“审核打回→原评估重写→再次审核”的局部图，验证两个模型环节能实际协作；提供共用构造输入 | 验收保留；验收脚本也引用其构造输入，不能单独删掉 |
| `scripts/audit_report.py` | 只读已有step8 JSON，重新显示审核范围、历史和未核实项；无需再付费调用模型 | 辅助保留，便于用户和接收方复查结果 |

语义验收检查“判断是否符合预期”；局部图联调检查“打回和重写能否串起来”；报告脚本检查“已经发生了什么”。三者用途不同。

### 测试：9个，均保留

| 文件 | 防止哪类回归 | 必要性 |
| --- | --- | --- |
| `tests/test_audit_contract.py` | 非法输出、空打回、通过却带问题、擅改身份和理由 | 验收保留 |
| `tests/test_audit_rules.py` | 标签用错推荐侧、把0当缺失、无关缺项或错误的必要字段判断 | 验收保留 |
| `tests/test_audit.py` | 单人审核输入、重试边界、代码与模型问题合并、原记录只读 | 验收保留 |
| `tests/test_audit_nodes.py` | 档位分流、整批状态、候选生成、缓存和CLI保存接线 | 验收保留 |
| `tests/test_reason_rewrite.py` | 审核反馈丢失、无限重写、复审旧版本或丢掉其他达人 | 验收保留 |
| `tests/test_supplement_loop.py` | 错人补数、覆盖已有值、错误重评分、旧理由覆盖新数据、补后违反硬条件 | 验收保留 |
| `tests/test_audit_handoff.py` | 未解决者混进候选、缺数失败被假装通过、单人异常阻断其他人、候选为空仍声称就绪 | 验收保留 |
| `tests/test_audit_reporting.py` | 将审核通过误显示为推荐、将待处理为空误显示成人工核实完成、无法只读复查旧结果 | 验收保留 |
| `tests/test_audit_acceptance.py` | 验收工具自己算错成绩、空题库通过、执行错误当业务结论、自定义题库未校验就发请求 | 验收保留 |

这些测试用固定回答控制代码分支，不是196个真实模型样本。部分测试复用现有 `tests/test_requirements.py` 的构造输入和原Excel，因此应在完整项目内运行，不把审核测试复制到空目录独立执行。

### 文档：收敛为2个

| 文件 | 内容 | 决定 |
| --- | --- | --- |
| `docs/audit-contract.md` | 本交接说明：文件清单、接口、路由、依赖与责任边界 | 保留 |
| `docs/audit-testing.md` | 命令、自定义题库、最新验收证据、修复记录与业务边界 | 保留 |
| 原 `docs/audit-review.md` | 重复介绍接口和流程，并累积了多轮开发历史和旧测试数字 | 从交付文档移除，有效内容并入上述两份；原文留在Git忽略的本地归档 |

整理后新增交付文件共22个：8个运行模块、3个辅助脚本、9个测试、2份文档。不新增另一份“文件清单”或“总结”来重复本文。

## 2. 原有文件中的必要接线

以下文件不是本次新建。交接审核模块时必须同步保留其中的审核增量，不能只复制上述8个运行文件。组员原有的其他逻辑不属于本次整理范围。

| 原有文件 | 本次审核接线的作用 |
| --- | --- |
| `agent/graph.py` | 接上审核、重写、补数、末端交接；允许注入单独审核客户端 `audit_llm_client` |
| `agent/state.py` | 声明审核结果、历史、待处理、补数、汇总及下游候选等图状态 |
| `agent/evaluation.py` | 原评估模型接收审核意见重新写理由，并带回 `data_revision` |
| `agent/nodes/evaluation.py` | 重写沿用原Send扇出；按身份、数据版本和理由轮次汇总，避免新旧结果混用 |
| `main.py` | 保存step8结果、展示审核和补数状态；图执行步数预算为64 |
| `tests/test_requirements.py` | 原端到端测试的终点从第7步推进到审核交接，新增假审核客户端 |
| `README.md` | 保留原流程说明，并链接本交接文档和测试手册 |

`agent/requirements.py`、`agent/screening.py`、输入Excel、组员原有测试文件的其他内容和 `.vscode/settings.json` 不作为本次新增审核文件整理。

现有依赖仍使用项目的 `requirements.txt`，审核部分没有另建依赖清单或要求额外安装库。接收方使用自己的 `.env` 配置；配置文件不属于源码交付清单。

## 3. 实际处理路径

```text
第7步理由汇总 → 审核
  ├─ 有可修正的非缺数问题 → 原评估模块重写 → 汇总 → 再审核
  ├─ 仅有必要缺数 → 查回 → 成功：重评分、重新评估、再审核
  │                         失败：标记数据不全后交接
  └─ 自动处理结束 → finalize_audit → 投放组合接口
```

- 原README约定：明确推荐/不推荐档跳过模型理由审校，仍做必要数据检查；中间档做模型审校。
- 标签和原始指标是不同层次：标签合法不代表理由为真。必要缺数也不能掩盖已有指标中的错误。
- 每人最多两版理由（初版加一次重写）。审核回答格式错误可纠正一次，这不是增加理由轮次。
- 超过理由上限仍有非缺数问题，或连续无法产生有效审核回答：暂存该人，其他可用候选继续。
- 未变化的评估和上下文可复用结果；指标、需求或理由变化时重新核对。补数改变同批标尺时需要重评分。
- 不新增提前人工审批节点；完整名单、缺数标记、未解决问题及未核实事项须传至原定最终人工审核。

## 4. 上下游接口

### 输入

- `evaluated_talents`：三档全名单。每条至少有有效身份、正整数 `attempt`、`band`、`talent` 及评估结论 `stance/tag/sentence/error`。
- `brief`（保留原文）、`requirement_profile`、`approved_policy`；有实际执行策略时使用 `active_policy`，以及已有的 `client_answers`。
- 补数需要 `talents` 同批原始记录；成功回评估还需要完整 `scored_talents` 基线、评估结果及理由历史。不能只传被补的一个人后重新定义整批标尺。

单人业务入口是 `AuditAgent(llm).audit(record, brief=..., requirement_profile=..., approved_policy=..., active_policy=..., client_answers=...)`，只接收中间档。模型收到单人的完整指标原值、显示值和评分构成；不能仅凭摘录的 `evidence` 判断。

### 模型输出

```json
{
  "verdict": "reject",
  "issues": [
    {
      "code": "data_conflict",
      "message": "理由中的进店数与原始进店中位数不一致。",
      "fields": ["visit_store_med"]
    }
  ]
}
```

通过时为 `{"verdict":"pass","issues":[]}`。模型只能输出这两个字段，每个问题只能包含 `code/message/fields`。代码绑定上游身份与 `attempt`；模型不得改身份、指标、倾向、标签或理由。`pass` 是“理由审核通过”，不是“推荐该达人”。

| 问题类型 | 含义 |
| --- | --- |
| `invalid_tag` | 标签不属于该推荐倾向的固定清单，代码检查 |
| `generation_error` | 上游理由没有有效生成，代码检查 |
| `data_missing` | 当前判断必要的已有指标缺失，必须列出具体字段 |
| `data_conflict` | 说法与指标不符或缺少依据；外部事实无法表示时字段允许为空 |
| `client_exclusion` | 推荐者有证据违反客户当前生效的明确排除项 |

字段名必须来自现有 `FIELD_LABELS`；0是有效数值，空白字符串是缺失。默认不要求所有空字段、双报价或MCN都补齐。字段确有值时不能报告缺失；网络错误、非法回答不能伪装成某人的业务打回。

### 末端状态与下游必须接收的内容

| 字段 | 下游用途 |
| --- | --- |
| `pricing_candidates` | 本轮可用于组合的推荐者，可能包含有明确缺数标记的放行者 |
| `audited_talents` | 全名单，含原倾向、理由、审核状态、缺数或暂存标记 |
| `audit_pending` | 自动处理后仍未解决的暂存问题，不等于所有未核实业务事项 |
| `supplement_results` | 查回值、来源、部分成功或失败放行记录 |
| `audit_unverified_checks` | 策略列出的人工检查项，当前无证据时保持 `unverified`，不能自动视为完成 |
| `audit_summary` | 本批审核范围和计数，不是账号调查或KPI验收证明 |
| `brief/requirement_profile/active_policy` | 后续继续决策所用的原始需求和执行策略 |

- `audit_results` 只保留中间档最新的有效审核结果；`audit_history` 保留实际有效审核的评估和上下文快照。
- `audit_status=skipped_clear_band` 表示按流程跳过，不能伪造模型通过记录。`execution_error` 表示审核未成功取得有效结论。
- `release_status=flagged_release`：仅补数失败，附 `data_incomplete=true` 和具体缺项，保留原倾向；只有原本推荐者进入候选。
- `release_status=held_for_review`：非缺数问题未解决，暂不参与组合；全名单及问题仍交给最终人工审核。
- `attempt` 是理由版本；`data_revision` 是指标/评分版本，两者用途不同。
- 整批状态可能为 `audit_complete`、`audit_complete_with_incomplete_data` 或 `audit_complete_with_pending`。这些状态不表示所有人工检查已完成。
- `next_step` 指向 `build_portfolio`，有候选为 `interface_ready`，无候选为 `no_candidates`；当前投放组合实现为false。
- `manual_review_queue` 仅保留空数组兼容旧格式，不是另一条人工审批通道。

不能只把 `pricing_candidates` 交给下游而丢弃完整名单、暂存原因和待核实事项。

