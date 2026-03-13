# TEST CASE

本测试用例文档以上游文档 [PRD](./01_PRD.md)、[ARCH](./02_ARCH.md)、[API Doc](./03_API_DOC.md)、[SPEC](./04_SPEC.md) 为基准，并尽量映射到仓库中已经存在的测试文件。

## 1. 单元测试用例

| 用例 ID | 模块 | 场景 | 预期结果 | 现有映射 |
|---|---|---|---|---|
| UT-ARB-001 | `scanner.py` | 去水位 `_remove_vig` 输入合法双边赔率 | 返回 fair odds 和 vig 百分比 | `tests/test_arbitrage.py` |
| UT-ARB-002 | `scanner.py` | Kelly 公式在负 edge、0 bankroll、非法赔率下计算 | 返回 0 stake / 0 percent | `tests/test_arbitrage.py` |
| UT-ARB-003 | `scanner.py` | Edge/EV 计算 | 返回可重复的数值结果 | `tests/test_arbitrage.py`、`tests/test_ev.py` |
| UT-MID-001 | `scanner.py` | Spread/Total gap 识别 | 正确识别 gap 与无效 gap | `tests/test_middles.py` |
| UT-MID-002 | `scanner.py` | Middle 概率、分注、EV 计算 | 概率、stake split、EV 符合预期 | `tests/test_middles.py` |
| UT-UTIL-001 | `scanner.py` | 行值归一化、点位匹配、跨 Provider 对阵归一化 | 产出稳定签名 | `tests/test_utils.py` |
| UT-HIS-001 | `history.py` | 历史记录追加、裁剪、按模式读取 | JSONL 追加成功且超限裁剪 | `tests/test_history.py` |
| UT-HIS-002 | `history.py` | 机会对象扁平化 | 输出最小可存档结构 | `tests/test_history.py` |
| UT-NOT-001 | `notifier.py` | 通知阈值过滤与摘要格式化 | 仅发送高于阈值的机会 | `tests/test_notifier.py` |
| UT-CFG-001 | `config.py` | 不同体育的默认市场推导 | 返回与体育类型匹配的市场集合 | `tests/test_config_markets.py` |
| UT-PROV-001 | `providers/bookmaker_xyz.py` | 扫描期缓存启停 | 缓存命中与清理逻辑正确 | `tests/test_bookmaker_xyz_cache.py` |
| UT-PROV-002 | `providers/purebet.py` | 市场解析、事件归一化 | 返回统一事件/市场结构 | `tests/test_purebet_market_parsing.py` |
| UT-PROV-003 | `providers/polymarket.py` | 实时状态与异步抓取 | 运行时状态、异步获取可用 | `tests/test_polymarket_realtime.py` |

## 2. 接口测试用例

| 用例 ID | 接口 | 场景 | 预期结果 | 现有映射 |
|---|---|---|---|---|
| IT-APP-001 | `POST /scan` | 非法 JSON | 返回 `400` 和 `Invalid JSON payload` | `tests/test_app_scan_inputs.py` |
| IT-APP-002 | `POST /scan` | JSON 不是对象 | 返回 `400` 和 `Scan payload must be a JSON object` | `tests/test_app_scan_inputs.py` |
| IT-APP-003 | `POST /scan` | `sharpBook` 非字符串 | 回退到默认 Sharp Book | `tests/test_app_scan_inputs.py` |
| IT-APP-004 | `POST /scan` | 布尔字符串输入 | 正确解析 `allSports` / `allMarkets` / `includePurebet` / `saveScan` | `tests/test_app_scan_inputs.py` |
| IT-APP-005 | `POST /scan` | `includeProviders=[]` 且选中 Provider 型 bookmaker | 自动推导 Provider Key | `tests/test_app_scan_inputs.py` |
| IT-APP-006 | `POST /scan` | 嵌套结果结构写历史 | 从 `opportunities` 嵌套中提取并写历史 | `tests/test_app_scan_inputs.py` |
| IT-APP-007 | `GET /` | 首页访问 | 返回 `200`，并预热后台 Provider 服务 | `tests/test_app_scan_inputs.py` |
| IT-APP-008 | `GET /provider-runtime/polymarket` | 查询可用运行时 | 返回 `success=true` 和状态字段 | `tests/test_app_scan_inputs.py` |
| IT-APP-009 | `GET /provider-runtime/unknown` | 查询未知 Provider | 返回 `404` | `tests/test_app_scan_inputs.py` |
| IT-APP-010 | `GET /history` | 默认读取历史记录 | 返回 `success`、`records`、`count` | 待补充 |
| IT-APP-011 | `GET /history/stats` | 读取历史统计 | 返回 `enabled`、`dir`、`modes` | 待补充 |
| IT-APP-012 | `GET /provider-snapshots/{provider}` | 快照存在/不存在 | 正确返回 `200` 或 `404` | 待补充 |
| IT-APP-013 | `GET /cross-provider-report` | 报告存在/不存在 | 正确返回 `200` 或 `404` | 待补充 |

## 3. 边界用例

| 用例 ID | 场景 | 预期结果 | 现有映射 |
|---|---|---|---|
| BD-001 | `regions=[]` | `/scan` 返回 400：至少选择一个地区 | 已在扫描器逻辑中实现，待补接口测试 |
| BD-002 | 仅 Provider 扫描且无 API Key | 合法执行，不返回 API key required | 已在扫描器逻辑中实现，待补接口测试 |
| BD-003 | `kellyFraction < 0` 或 `> 1` | 归一化到 `[0, 1]` 区间 | 已在 `app.py` 中实现，待补接口测试 |
| BD-004 | `limit` 非法或越界 | `/history` 回退到默认值或最大值 | 已在 `app.py` 中实现，待补接口测试 |
| BD-005 | 历史文件超过上限 | 自动裁剪到 `HISTORY_MAX_RECORDS` | `tests/test_history.py` |
| BD-006 | 扫描结果为空 | 仍返回成功结构和 0 数量 summary | `tests/test_scanner_regressions.py` 覆盖部分场景 |

## 4. 异常用例

| 用例 ID | 场景 | 预期结果 | 现有映射 |
|---|---|---|---|
| EX-001 | 需要 The Odds API 但未传 API Key | 返回 400：`API key is required` | 已在扫描器逻辑中实现，待补接口测试 |
| EX-002 | 未启用 API 且未启用任何 Provider | 返回 400：`No enabled providers selected` | 已在扫描器逻辑中实现，待补接口测试 |
| EX-003 | Provider 预热失败 | 记录 warning，不阻塞首页和扫描接口 | 代码已实现，待补回归测试 |
| EX-004 | 历史保存失败 | 记录 warning，不阻塞扫描返回 | 代码已实现，待补回归测试 |
| EX-005 | 通知发送失败 | 记录 warning，不阻塞扫描返回 | 代码已实现，待补回归测试 |
| EX-006 | 快照文件 JSON 损坏 | `/provider-snapshots/{provider}` 返回 500 | 代码已实现，待补接口测试 |
| EX-007 | 跨 Provider 报告 JSON 损坏 | `/cross-provider-report` 返回 500 | 代码已实现，待补接口测试 |

## 5. 测试覆盖率要求

- 核心数学逻辑必须覆盖：
  - 套利 ROI、去水位、Kelly、EV
  - Middle gap、概率、EV、分注
  - +EV edge 和 fair odds
- 接口层必须覆盖所有公开 JSON 接口的成功与失败分支。
- 每个自定义 Provider 至少覆盖一种解析/分段/运行时场景。
- 历史记录、通知、请求日志等非主链路副作用必须覆盖“不阻塞主流程”的行为。
- 当前仓库未内置覆盖率门禁配置；如后续接入 `pytest-cov`，应以“核心数学逻辑 + 全部公开接口 + 全部已注册 Provider 的关键路径”作为最低门槛。
