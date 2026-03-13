# PRD

项目名称：Edge Scanner

## 1. 项目目标

Edge Scanner 是一个本地运行的体育投注赔率扫描工具，目标是把 The Odds API 与自定义 Provider 的赔率数据统一收敛到一个 Web 界面中，帮助用户快速发现三类机会：

- 套利机会（Arbitrage）
- 中间盘机会（Middles）
- 正期望机会（+EV）

项目当前版本同时提供筛选、历史记录、通知、Provider 快照与跨 Provider 对账能力，定位为“发现机会的分析工具”，不负责自动下注。

## 2. 核心功能

1. 扫描赔率
   - 通过 `/scan` 接口触发扫描。
   - 支持按体育项目、地区、Bookmaker、Sharp Book、Stake、Bankroll、Kelly 系数等参数过滤。
2. 机会识别
   - 识别套利机会并输出 ROI、分注结果、最佳赔率来源。
   - 识别中间盘机会并输出 gap、EV、概率、分注结果。
   - 识别 +EV 机会并输出 fair odds、edge、EV、Kelly 建议下注额。
3. 多数据源整合
   - 支持 The Odds API。
   - 支持 `purebet`、`betdex`、`bookmaker_xyz`、`sx_bet`、`polymarket`、`dexsport_io`、`sportbet_one`。
4. 结果留存与诊断
   - 扫描结果可写入历史记录。
   - 可选保存最新扫描请求/结果。
   - 可选写入请求日志、Provider 快照、跨 Provider 匹配报告。
5. 前端交互
   - 首页提供参数配置、立即扫描、自动扫描、通知、历史展示与诊断入口。

## 3. 用户流程

1. 用户启动 `python app.py`，浏览器自动打开首页 `/`。
2. 用户在页面输入 API Key，或使用环境变量 / `settings.json` 预置配置。
3. 用户选择地区、体育、Bookmaker、Stake、Sharp Book 等参数。
4. 用户点击“Scan Now”，前端向 `/scan` 发送 JSON 请求。
5. 后端执行统一扫描流程，汇总 API 与自定义 Provider 数据。
6. 后端返回套利、中间盘、+EV 三类结果以及统计、时序、Provider 状态。
7. 前端展示结果；后台异步写入历史记录，并按配置发送通知。
8. 用户可查看 `/history`、`/history/stats`、`/provider-snapshots/{provider}`、`/provider-runtime/{provider}`、`/cross-provider-report` 等诊断信息。

## 4. 非功能需求（性能、安全、稳定性）

### 性能

- 单次扫描需返回结构化 `timings`，用于展示和排查性能瓶颈。
- Provider 拉取与市场处理支持并发，避免串行阻塞。
- 通知发送必须异步，不阻塞 `/scan` 响应。

### 安全

- 保存扫描请求时必须脱敏 `apiKey` / `apiKeys`。
- 请求日志必须限制正文长度，并对敏感字段做清洗。
- 服务默认本地运行，不暴露额外鉴权机制。

### 稳定性

- 输入非法时必须返回结构化错误对象，而不是未处理异常。
- 某些体育项目或 Provider 失败时，应尽量返回 `partial=true` 的部分结果，而非整次崩溃。
- Polymarket 实时服务初始化失败时，只影响相关能力，不应阻断首页和主扫描流程。

## 5. 约束条件

- 当扫描包含 The Odds API 数据源时，必须提供可用 API Key；仅 Provider 扫描时可不提供。
- 至少选择一个地区；系统会为 Sharp Book 自动补齐必要地区。
- 数据持久化当前为本地文件模式，未引入数据库。
- 部署形态以单机本地 Flask 服务为主，仓库未提供容器编排或生产部署脚本。
- 仅支持文档中已注册的 Provider Key 与别名，不支持未注册 Provider。

## 6. 验收标准

1. 首页 `/` 可正常渲染，并回填默认配置项。
2. `/scan` 在合法输入下返回：
   - `success=true`
   - `scan_time`
   - `arbitrage`
   - `middles`
   - `plus_ev`
   - `timings`
3. `/scan` 在非法 JSON 或非对象 JSON 时返回 400，并包含 `success=false`、`error`、`error_code`。
4. 启用历史记录后，扫描成功结果会异步写入历史文件，`/history` 与 `/history/stats` 可读取。
5. 启用快照后，可通过 `/provider-snapshots/{provider_key}` 与 `/cross-provider-report` 读取最新扫描诊断产物。
6. `/provider-runtime/polymarket` 可返回运行时状态；未知 Provider 返回 404。
