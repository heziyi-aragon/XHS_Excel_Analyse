# 模块级 LLM 主备线路任务

## Task 1: 配置与熔断客户端

**Acceptance criteria:**
- [x] 六条线路独立解析五项配置
- [x] 主线路配置无效时使用备用，备用无效时明确失败
- [x] 主线路运行失败后当前模块保持熔断

**Verification:**
- [x] `.venv\Scripts\python.exe -m pytest tests\test_llm_routing.py -q`

**Dependencies:** None

## Task 2: 五类模块接线

**Acceptance criteria:**
- [x] 需求与策略共用 REQUIREMENTS
- [x] 评估、审核、组合、建议分别使用对应客户端
- [x] 格式校验失败能够触发熔断

**Verification:**
- [x] 相关 Agent 与图集成测试通过

**Dependencies:** Task 1

## Task 3: CLI、示例与回归验证

**Acceptance criteria:**
- [x] CLI 使用六线路注册表
- [x] `.env.example` 完整展示 5+1 配置
- [x] README 说明切换语义且不包含真实密钥

**Verification:**
- [x] 语法编译通过
- [x] 完整测试结果与既有基线对比：通过数 383 → 395，失败仍为原有 17 项

**Dependencies:** Task 2
