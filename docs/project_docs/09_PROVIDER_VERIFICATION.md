# Provider 接口核验

本文档用于约束和记录“自定义 Provider 接口是否正确”的核验过程。该流程是 [00_WORKFLOW.md](./00_WORKFLOW.md) 的强制补充，不是可选项。

## 1. 核验流程

1. 先跑本地 Provider 相关测试，确认解析和回归逻辑没有明显问题。
2. 再执行 Provider-only 实时扫描，确认在线端点仍可访问、返回仍符合当前实现假设。
3. 如果发现空数据、状态码异常、字段漂移、盘口错配、赔率异常，必须回到官方文档复核。
4. 最后抽查顶部套利 / 中间盘 / +EV 结果，确认赛事、盘口、赔率来源、最大下注额和链接合理。

## 1.1 重复执行脚本

仓库已提供脚本：

```bash
python provider_verification.py --sport basketball_nba
```

Windows 可直接使用：

```powershell
.\run_provider_verification.ps1
```

```bat
run_provider_verification.bat
```

本地自动化脚本：

```powershell
.\scheduled_provider_verification.ps1 --sport basketball_nba --fail-on-alert
.\register_provider_verification_task.ps1 -DailyAt 09:00 -Sport basketball_nba -FailOnAlert
```

仓库也提供 GitHub Actions 工作流：

- `.github/workflows/provider-verification.yml`
- 定时任务默认上传 `data/provider_verification/` 产物
- 手动触发时可选择 strict 模式，在有告警时直接失败

脚本会输出两类文件到 `data/provider_verification/`：

- `provider_verification_<timestamp>.json`
- `provider_verification_<timestamp>.md`
- `provider_verification_latest.json`
- `provider_verification_latest.md`

常用参数：

- `--skip-tests`
- `--providers betdex bookmaker_xyz sx_bet polymarket`
- `--all-markets`
- `--out-dir data/provider_verification`
- `--summary-only`
- `--json-stdout`
- `--fail-on-alert`

说明：

- `run_provider_verification.ps1` 和 `run_provider_verification.bat` 默认使用 `--summary-only`
- 当需要接自动化时，建议使用 `--fail-on-alert`
- Windows 包装脚本会透传 Python 退出码，因此命中告警时会按设计返回非零状态
- `scheduled_provider_verification.ps1` 会额外写出调度日志：
  - `provider_verification_scheduler_<timestamp>.log`
  - `provider_verification_scheduler_latest.log`

## 2. 当前核验记录

核验日期：2026-03-13

### 2.1 本地测试

- 已执行：
  - `tests/test_provider_market_segmentation.py`
  - `tests/test_provider_sport_coverage.py`
  - `tests/test_polymarket_realtime.py`
  - `tests/test_bookmaker_xyz_cache.py`
  - `tests/test_scanner_regressions.py`
- 结果：关键 Provider 回归通过

### 2.2 实时扫描

- 执行方式：Provider-only，`sports=["basketball_nba"]`
- 实时结果摘要：
  - `betdex`: 返回 8 个事件
  - `bookmaker_xyz`: 返回 8 个事件
  - `sx_bet`: 返回 8 个事件
  - `polymarket`: 返回 54 个事件
  - `plus_ev`: 0
  - `arbitrage`: 1
  - `middles`: 609

## 3. 官方来源与结论

| Provider | 当前实现端点 | 官方来源 | 结论 |
|---|---|---|---|
| BetDEX | `https://www.betdex.com/api/session`、`https://production.api.monacoprotocol.xyz/events`、`/markets`、`/market-prices` | `https://docs.betdex.com/`、`https://docs.api.monacoprotocol.xyz/` | 当前实现与线上请求日志一致，端点可用 |
| bookmaker.xyz | `https://api.onchainfeed.org/api/v1/public/market-manager/*` + `@azuro-org/dictionaries` | `https://docs.bookmaker.xyz/guides/sportsbook`、`https://api.onchainfeed.org/api/v1/public/gateway/docs` | 当前链路已切到 Azuro 官方公共 `market-manager`，并使用官方 dictionaries 解析盘口 |
| SX Bet | `https://api.sx.bet/summary/upcoming/{baseToken}/{sportId}`、`/leagues/active`、`/markets/active`、`/orders/odds/best`、`/orders` | `https://api.docs.sx.bet/` | 当前实现与公开文档中的端点形态一致，实时请求成功 |
| Polymarket | `https://gamma-api.polymarket.com/events`、`https://clob.polymarket.com/book`、`wss://ws-subscriptions-clob.polymarket.com/ws/market` | `https://docs.polymarket.com/developers/gamma-markets-api/get-events`、`https://docs.polymarket.com/developers/CLOB/introduction` | 当前实现与官方文档一致，实时请求成功 |
## 4. 本轮发现

1. Provider 快照在旧实现里会被后续事件合并污染，导致 `bookmaker_xyz` 快照中混入 `sx_bet` 盘口；该问题已在本轮修复。
2. 顶部套利结果存在极高 ROI，但往往伴随很小的 `max_stake`，需要按“低流动性 / 短时错价”处理，不能直接视为稳定机会。
3. `bookmaker.xyz` 主链路已改为官方 `market-manager`，`basketball_nba` 与 `soccer_epl` 当前都能从官方实时 feed 返回事件，不再依赖旧 GraphQL 快照。
4. `bookmaker.xyz` 同时支持动态 Azuro sport key：`azuro__<sport_slug>__<league_slug>__<country_slug>`，用于覆盖当前公共 feed 中的新增联赛。

## 5. 扫描结果抽检结论

本轮抽检 `basketball_nba` 的顶部结果后，结论如下：

- 顶部套利主要来自 `SX Bet` 与 `bookmaker.xyz` 的价差组合。
- 赛事匹配本身是正确的，球队名和开赛时间能在快照里对上。
- 但最高 ROI 机会受到 `SX Bet` 很小的 `max_stake` 限制，说明更可能是低深度盘，不适合作为稳定套利样本。
- 中间盘结果大多是 `BetDEX` 与 `SX Bet` 的 totals 差值，结构上合理，但前几条 EV 仍为负值，使用时应按排序和过滤规则继续筛选。

## 6. 后续执行要求

- 每次改动任一 Provider 取数逻辑，都必须更新本文档。
- 如果官方文档或线上端点有变化，先更新本文档，再改代码和测试。
- 如果再次出现“结果看起来不对”，优先顺序固定为：
  - 请求日志
  - Provider 快照
  - 官方文档
  - 实时复测
