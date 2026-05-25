# 面向中国大陆友好性的 Provider 候选调研

> 这是技术可行性调研，不是法律意见。不要用本项目绕过地域限制、KYC、平台条款或当地法律。凡是官方明确限制中国大陆，或需要 VPN、代理、身份规避才能访问的平台，都不应作为可执行套利平台接入。

## 结论

本轮不新增 Betfair 或 Matchbook。

当前更值得做的不是继续堆传统投注平台，而是：

1. 先加强现有 provider 的“机会是否真的可执行”判断。
2. 优先补强已有的 Azuro/bookmaker.xyz 路径，因为项目已经有 `bookmaker_xyz` provider。
3. Dexsport 只进入调研队列，必须先确认稳定、公开或合作 API。
4. 数据聚合 API 只能作为校验源，不能直接标记为可执行投注平台。

## 候选平台判断

| 候选 | 类型 | 当前结论 | 原因 |
| --- | --- | --- | --- |
| Azuro / bookmaker.xyz | 链上体育投注流动性协议 / 前端 | 继续补强现有 provider | 项目已经支持 `bookmaker_xyz`。Azuro V3 文档提供 backend、graph、WebSocket、注单计算、订单管理、cashout、历史数据和实时更新等接口方向。 |
| Dexsport | Web3 sportsbook / 流动性池式平台 | 仅调研，暂不实现 | 本轮看到的条款没有在命名禁止地区里列出中国，但用户仍必须遵守所在地法律；同时还没有确认稳定公开数据 API。 |
| ParlayAPI | 赔率 / 交易所数据聚合 API | 可作为校验源，不作为可执行平台 | 适合做数据健康度和跨源校验，但它本身不是用户下注或成交的平台。 |
| Polymarket | 预测市场 CLOB | 保留现有 provider，并按体育市场手续费计算净 ROI | 体育套利需要扣除平台 taker fee，不能只看 CLOB 原始 best ask 折算出的十进制赔率。项目已有 provider，后续重点是费用、深度和成交可用性校验。 |
| Overtime | 链上体育 AMM | 可作为后续候选，但需要用户确认访问与条款责任 | 技术上匹配，且用户反馈当前可用；接入前仍要确认 API 权限、市场覆盖、流动性、结算规则和目标使用环境。 |
| Fairlay | P2P betting exchange | 不接入 | 官网显示服务正在关闭，并要求用户在 6 月 1 日前提现。 |
| Purebet | 多协议聚合前端 | 暂缓 | 当前环境下公开文档跳转到 Google 文档，条款和地域可用性没有足够确认。 |

## 为什么先做现有 Provider 质量

前面的真实扫描已经暴露出核心限制：不是“平台数量不够”，而是 provider 数据可用性、更新时间、流动性和最大下注额决定机会是否可执行。有些盘口会出现纸面 ROI，但因为报价时间缺失或流动性太小，不能直接当成真实套利机会。

项目现在已经为每个套利机会增加 `execution_quality`：

- `missing_quote_time`
- `missing_liquidity`
- `limited_by_liquidity`
- `below_min_executable_stake`
- `quote_time_skew`

后续判断机会是否可操作，应先看 `execution_quality.status`。`low` 不能当成真实可执行机会，只能作为线索继续人工核对。

## 实施建议

### 1. 先补强 Azuro/bookmaker.xyz

原因：它已经注册在 `providers/__init__.py`，增强这里可以复用现有 UI、诊断、provider verification 和历史扫描链路。

建议后续任务：

- 在 provider stats 中记录每条腿的来源端点和报价更新时间。
- 尽量透出 market 级别流动性或可执行下注额。
- 给 generic Azuro sport key 增加 provider verification 用例。
- 增加 provider health check，把 dictionary/API 新鲜度和 event 数量分开显示。

### 2. Dexsport 的调研门槛

Dexsport 在满足这些条件前不要实现：

- 是否有稳定公开 API 或合作 API 可取赛前 / 赛中盘口。
- 是否能拿到 event ID、market ID、line ID、odds、max stake、last update time。
- 目标用户所在地是否允许访问，且不需要任何地域规避。
- 结算规则是否足够明确，能和其他 provider 做同盘口匹配。

只要有一项不清楚，就不要把它放进可执行扫描器，也不要抓取私有或受保护接口。

### 3. 聚合 API 只做校验

ParlayAPI 这类数据源可以作为二级校验：

- 对比本地 provider 数据是否过期。
- 用 exchange/orderbook 字段辅助判断本地机会是否可信。
- 只有当用户可以在对应平台直接成交时，才允许把机会标记为可执行。

## 已拒绝或低优先级候选

### Overtime

Overtime 技术上有吸引力，因为它是链上体育 AMM，并且有开发者 API。用户反馈该平台当前可用，因此不再直接排除，但接入前仍需要完成下面几项核验：

- 官方文档说明其使用 Pool-vs-Peer AMM。
- protected API 需要审批 key，不适合在没有 key 的情况下硬接。
- 需要确认目标使用环境下访问、注册、KYC、交易和结算均符合平台条款与当地规则。
- 需要先建立只读 provider 原型，验证盘口 ID、赔率、流动性、最大下注额和更新时间是否足够稳定。

### Fairlay

Fairlay 当前不是好目标：

- 它自称 prediction market / exchange，并强调低佣金。
- 但官网也显示服务正在关闭，并要求用户在 6 月 1 日前提现。

## 来源

- Azuro API docs: https://gem.azuro.org/hub/apps/APIs
- Overtime protocol overview: https://docs.overtime.io/learn-about-overtime/how-overtime-works
- Overtime protected API docs: https://docs.overtime.io/overtime-v2-integration/overtime-v2-markets-protected
- Overtime terms: https://www.overtimemarkets.xyz/assets/overtime-terms-of-use-CK6ZVBha.pdf
- Fairlay homepage: https://fairlay.com/
- Dexsport general terms: https://dexsport.io/docs-general-terms/
- Dexsport betting rules: https://dexsport.io/docs-betting-rules/
- Polymarket geographic restrictions: https://help.polymarket.com/en/articles/13364163-geographic-restrictions
- ParlayAPI docs: https://parlay-api.com/docs
- PRC online gambling legal context: https://www.spp.gov.cn/spp/nbgz/201802/t20180201_363851.shtml
