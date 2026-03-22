# TEST PROGRESS

本文档用于长期保存“应该执行的测试计划、已完成进度、当前未完成项和最近执行记录”。

任何涉及以下动作的任务，在开始前都必须先读取本文件：

- 功能开发后的回归测试
- 赛前 / 赛中套利能力验收
- Provider 接口调整后的复测
- 发布前验收
- 问题复盘后的补测

本文档与 [05_TEST_CASE.md](./05_TEST_CASE.md) 一起定义“该测什么”，与 [09_PROVIDER_VERIFICATION.md](./09_PROVIDER_VERIFICATION.md) 一起定义“Provider 如何做线上核验”。

## 1. 使用规则

1. 开始测试前，先阅读 `05_TEST_CASE.md`、`09_PROVIDER_VERIFICATION.md`、本文件。
2. 优先执行本文件中状态为 `待执行`、`部分验证`、`阻塞` 的项目，优先级按编号顺序处理。
3. 只要本轮改动触达某个模块，就必须重跑对应回归测试，即使该模块之前显示为 `已验证`。
4. 每次执行测试后，必须回写：
   - 执行日期
   - 执行命令
   - 结果摘要
   - 未解决风险或下一步动作
5. Provider 实时扫描、人工抽检、异常降级验证不能被离线单元测试替代。

## 2. 状态定义

- `已验证`：本轮或最近一次有效回归中已执行，结果通过，且没有未关闭阻塞项。
- `部分验证`：已有历史执行记录或只验证了部分子场景，仍需补测。
- `待执行`：尚未在该追踪文档中形成有效执行记录，或已知存在测试缺口。
- `阻塞`：当前无法可靠执行，需要先解决环境、数据或代码问题。

## 3. 测试矩阵

| 编号 | 测试域 | 目标 | 主要证据 / 命令 | 最新进度 | 状态 | 下一步 |
|---|---|---|---|---|---|---|
| TM-001 | 核心数学逻辑 | 验证套利 ROI、去水位、Kelly、EV、middle gap 与 stake split | `python -m pytest -q tests/test_arbitrage.py tests/test_ev.py tests/test_middles.py` | `2026-03-22` 已执行首轮高优先级回归，同批命令合计 `171 passed, 3 subtests passed` | 已验证 | 后续若修改数学逻辑或 stake 分配，需重跑 |
| TM-002 | 赛前套利回归 | 验证赛前盘口归类、赔率组合、回归样本和过滤逻辑 | `python -m pytest -q tests/test_prematch_arbitrage_additional.py tests/test_scanner_regressions.py` | `2026-03-22` 已执行首轮高优先级回归，同批命令合计 `171 passed, 3 subtests passed` | 已验证 | 后续若修改赛前筛选、合并或排序逻辑，需重跑 |
| TM-003 | 赛中套利回归 | 验证赛中盘口分段、实时赔率组合、误报回归 | `python -m pytest -q tests/test_inplay_arbitrage_additional.py tests/test_scanner_regressions.py` | `2026-03-22` 已执行首轮高优先级回归，同批命令合计 `171 passed, 3 subtests passed` | 已验证 | 后续若修改赛中识别、盘口归类或去重逻辑，需重跑 |
| TM-004 | 扫描接口与入参 | 验证 `/scan`、首页、runtime 接口、自动扫描输入保护 | `python -m pytest -q tests/test_app_scan_inputs.py tests/test_app_auto_scan.py` | `2026-03-22` 已补 Provider-only 无 key、Kelly 边界，并与历史端点联跑 `42 passed`；仍缺 `BD-001` 与 `EX-001/002` 专门覆盖 | 部分验证 | 补齐 `/scan` 剩余输入边界与错误分支回归 |
| TM-005 | 历史 / 统计 / 快照接口 | 验证 `/history`、`/history/stats`、provider snapshot、cross-provider report | `python -m pytest -q tests/test_app_history_endpoints.py` | `2026-03-22` 已补 `tests/test_app_history_endpoints.py`，目标端点回归 `14 passed` | 已验证 | 后续若修改历史或快照接口，需重跑 |
| TM-006 | 历史、通知、配置、副作用 | 验证历史写入、裁剪、通知失败不阻塞、默认配置推导 | `python -m pytest -q tests/test_history.py tests/test_notifier.py tests/test_config_markets.py` | `2026-03-22` 已补历史保存失败与通知失败不阻塞回归；相关 app 套件联跑 `42 passed` | 已验证 | 后续若修改历史、通知或默认配置链路，需重跑 |
| TM-007 | Provider 契约与解析回放 | 验证 provider 分段、契约回放、快照 golden、覆盖范围 | `python -m pytest -q tests/test_provider_arb_pipeline.py tests/test_provider_contract_replay.py tests/test_provider_market_segmentation.py tests/test_provider_snapshot_golden.py tests/test_provider_sport_coverage.py tests/test_provider_verification.py` | `2026-03-22` 已执行，`46 passed` | 已验证 | 后续若修改 provider 解析或快照结构，需重跑 |
| TM-008 | Provider 实时校验 | 线上验证 provider-only 扫描、结构、数量、结果抽检 | `python provider_verification.py --sport <sport_key>` | `2026-03-22` 已执行 `basketball_nba`，`tests.ran=true`、4 个 provider 全部 `ok`，但出现 3 条 middle negative EV 告警 | 部分验证 | 对告警样本做人工抽检，确认是否为低质量 middle 或排序问题 |
| TM-009 | 前端 helper 与浏览器流程 | 验证模板 helper、scan form JS、浏览器扫描流 | `node --test tests/frontend_market_line_formatting.test.js tests/frontend_scan_form_helpers.test.js`；`python -m pytest -q tests/test_browser_scan_flow.py tests/test_frontend_scan_form_helpers.py` | `2026-03-22` 已执行，Node `11/11` 通过；Python 浏览器流相关 `11 passed` | 已验证 | 后续若修改模板 helper 或前端扫描流程，需重跑 |
| TM-010 | 稳定性、异常与降级 | 验证长时间扫描稳定性、局部失败、partial 返回与非阻塞副作用 | `python -m pytest -q tests/test_scan_stability.py tests/test_scanner_regressions.py` + 手工异常注入 | `2026-03-22` 已执行 `tests/test_scan_stability.py` 与 `tests/test_scanner_regressions.py`，自动化回归通过；手工异常注入尚未补齐 | 部分验证 | 补做 EX-003 ~ EX-008 对应的异常注入或 mock 回归 |
| TM-011 | 扫描结果人工抽检 | 抽检顶部套利 / 中间盘 / +EV 结果，确认赛事、盘口、赔率来源、流动性合理 | 结合 `provider_verification_latest.*`、扫描快照、请求日志手工核对 | `2026-03-22` 已有实时扫描结果，但仅完成机器摘要，尚未逐条人工抽检告警样本 | 部分验证 | 先核对 3 条 negative EV middle 的赛事、盘口和流动性 |

## 4. 当前测试缺口

以下项目已经在 `05_TEST_CASE.md` 中定义，但仍缺少稳定的自动化覆盖或最近执行记录，后续测试时应优先补齐：

| 优先级 | 用例 ID | 场景 | 当前状态 | 建议动作 |
|---|---|---|---|---|
| P1 | BD-001 | `regions=[]` | 逻辑已实现，缺接口回归 | 补 `/scan` 参数校验测试 |
| P1 | EX-001 | 需要 The Odds API 但缺 API Key | 逻辑已实现，缺接口回归 | 补错误分支测试 |
| P1 | EX-002 | 未启用 API 且未启用任何 Provider | 逻辑已实现，缺接口回归 | 补错误分支测试 |
| P1 | EX-003 | Provider 预热失败 | 逻辑已实现，缺回归 | 模拟预热异常并验证首页不阻塞 |
| P0 | EX-008 | Provider 上游不可用 | 需要实时复测 | 通过 mock 或实时注入验证 `partial=true` |

## 5. 执行记录

| 日期 | 编号 | 命令 / 证据 | 结果 | 备注 |
|---|---|---|---|---|
| 2026-03-21 | TM-008 | `data/provider_verification/provider_verification_latest.md` | 部分通过 | 仅 `polymarket` 实时扫描；`tests.ran=false`，需补含 pytest 的 provider_verification |
| 2026-03-22 | TM-001 / TM-002 / TM-003 / TM-004 / TM-006 | `python -m pytest -q tests/test_arbitrage.py tests/test_ev.py tests/test_middles.py tests/test_app_scan_inputs.py tests/test_app_auto_scan.py tests/test_prematch_arbitrage_additional.py tests/test_inplay_arbitrage_additional.py tests/test_scanner_regressions.py tests/test_history.py tests/test_notifier.py tests/test_config_markets.py` | 通过 | 合计 `171 passed, 3 subtests passed in 1.38s` |
| 2026-03-22 | TM-005 | `python -m pytest -q tests/test_app_history_endpoints.py` | 通过 | `14 passed in 0.88s` |
| 2026-03-22 | TM-004 / TM-005 | `python -m pytest -q tests/test_app_scan_inputs.py tests/test_app_auto_scan.py tests/test_app_history_endpoints.py` | 通过 | `38 passed in 1.06s` |
| 2026-03-22 | TM-004 / TM-006 | `python -m pytest -q tests/test_app_scan_inputs.py` | 通过 | `17 passed in 1.12s` |
| 2026-03-22 | TM-004 / TM-005 / TM-006 | `python -m pytest -q tests/test_app_scan_inputs.py tests/test_app_auto_scan.py tests/test_app_history_endpoints.py` | 通过 | `42 passed in 1.20s` |
| 2026-03-22 | TM-009 | `node --test tests/frontend_market_line_formatting.test.js tests/frontend_scan_form_helpers.test.js` | 通过 | `11/11` 通过 |
| 2026-03-22 | TM-007 | `python -m pytest -q tests/test_provider_arb_pipeline.py tests/test_provider_contract_replay.py tests/test_provider_market_segmentation.py tests/test_provider_snapshot_golden.py tests/test_provider_sport_coverage.py tests/test_provider_verification.py` | 通过 | `46 passed in 0.95s` |
| 2026-03-22 | TM-009 / TM-010 | `python -m pytest -q tests/test_browser_scan_flow.py tests/test_frontend_scan_form_helpers.py tests/test_scan_stability.py` | 通过 | `11 passed in 9.79s` |
| 2026-03-22 | TM-008 | `python provider_verification.py --sport basketball_nba --summary-only` | 部分通过 | `success=true`、`partial=false`、4 个 provider 全部 `ok`；出现 3 条 middle negative EV 告警，需人工抽检 |

## 6. 当前剩余待执行顺序

建议下一轮优先从上到下清空：

1. `TM-008` Provider 实时校验告警样本人工抽检
2. `TM-010` 异常注入与降级场景补测
3. `TM-004` `/scan` 剩余边界与错误分支补测（`BD-001`、`EX-001/002`）
4. `TM-011` 顶部套利 / 中间盘 / +EV 人工抽检归档
5. 已验证项目在后续相关代码改动后按模块重跑
