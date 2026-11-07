# Implementation Plan: 模块级 LLM 主备线路

## Overview

在现有 `LLMClient` 之上增加配置注册表与进程内熔断客户端，再把五类客户端注入 LangGraph、CLI 和独立脚本。保持旧接口可用，不引入新依赖。

## Architecture Decisions

- 使用组合式 `FailoverLLMClient` 包装主、备 `LLMClient`，避免把业务模块与环境变量耦合。
- 模块校验器通过统一的 `report_validation_failure()` 通知路由器；普通 `LLMClient` 上该方法为空操作，保持显式注入兼容。
- `build_graph` 新增 `evaluation_llm_client` 和 `advice_llm_client` 可选参数；旧调用仍回退到原 `llm_client`。

## Task List

### Phase 1: 配置与熔断基础
- [x] 为六线路环境变量解析、缺失配置和进程内熔断编写失败测试
- [x] 实现配置注册表与 `FailoverLLMClient`

### Checkpoint: Foundation
- [x] `tests/test_llm_routing.py` 通过

### Phase 2: 模块接线
- [x] 将格式校验失败接入熔断通知
- [x] 将五类客户端接入主图、CLI 与模块脚本

### Checkpoint: Integration
- [x] 路由聚焦测试与相关 Agent 测试通过

### Phase 3: 配置示例与验证
- [x] 更新 `.env.example` 和 README
- [x] 运行完整测试并区分既有失败与新增回归（395 通过、17 个既有失败、0 个新增失败）

## Risks and Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| 格式校验发生在客户端外部 | 无法触发备用线路 | 提供统一失败通知并在五类 Agent 的既有重试点调用 |
| 批量理由调用反复访问故障线路 | 大量超时 | 每模块首次失败后进程内保持熔断 |
| 新参数破坏旧测试和调用方 | 回归 | 新参数全部可选，默认沿用原客户端 |

## Open Questions

- 无；用户已确认六线路、失败范围和熔断生命周期。
