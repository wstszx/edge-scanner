# API Doc

## 接口列表

| 接口 | 方法 | 说明 |
|---|---|---|
| `/` | `GET` | 渲染首页和默认配置 |
| `/scan` | `POST` | 执行扫描并返回套利 / 中间盘 / +EV 结果 |
| `/history` | `GET` | 读取最近历史记录 |
| `/history/stats` | `GET` | 读取历史存储统计 |
| `/provider-snapshots/{provider_key}` | `GET` | 读取某个 Provider 的最新快照 |
| `/provider-runtime/{provider_key}` | `GET` | 读取 Provider 运行时状态，当前仅支持 `polymarket` |
| `/cross-provider-report` | `GET` | 读取跨 Provider 对账报告 |

---

## 1. 首页

- 路径：`/`
- 方法：`GET`
- 请求参数：无
- 响应结构：
  - `text/html`
  - 页面中注入默认体育、地区、Bookmaker、Sharp Book、自动扫描、通知、主题等默认值
- 状态码：
  - `200`：渲染成功
- 错误示例：
  - 无专门 JSON 错误返回；模板异常会表现为服务端 500

---

## 2. 扫描接口

- 路径：`/scan`
- 方法：`POST`
- 请求参数：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `apiKey` | `string` | 条件必填 | 单个 The Odds API Key |
| `apiKeys` | `string` / `string[]` | 条件必填 | 多个 API Key，支持轮换 |
| `sports` | `string[]` | 否 | 指定体育项目 key |
| `allSports` | `bool` | 否 | 是否扫描全部可用体育 |
| `allMarkets` | `bool` | 否 | 是否扫描扩展市场 |
| `stake` | `number` | 否 | 总下注额，默认取配置 |
| `regions` | `string[]` | 否 | 区域列表，如 `us`、`eu` |
| `bookmakers` | `string[]` | 否 | 仅扫描指定 Bookmaker |
| `commission` | `number` | 否 | 百分比输入，例如 `5` 表示 5% |
| `includePurebet` | `bool` | 否 | 是否显式启用 / 禁用 Purebet |
| `includeProviders` | `string` / `string[]` | 否 | 指定自定义 Provider |
| `sharpBook` | `string` | 否 | +EV 参考 Sharp Book |
| `minEdgePercent` | `number` | 否 | +EV 最小 edge |
| `bankroll` | `number` | 否 | Kelly 计算 bankroll |
| `kellyFraction` | `number` | 否 | 0 到 1 之间 |
| `saveScan` | `bool` | 否 | 是否保存本次扫描请求和结果 |

说明：

- 当请求仅使用自定义 Provider，且不需要 The Odds API 数据时，`apiKey` / `apiKeys` 可省略。
- 若 `includeProviders` 未显式传入，但 `bookmakers` 中包含 Provider 型 Bookmaker，后端会自动推导对应 Provider Key。
- 非字符串 `sharpBook` 会回退到默认 Sharp Book。

- 响应结构：

```json
{
  "success": true,
  "scan_time": "2026-03-12T12:00:00Z",
  "arbitrage": {
    "opportunities": [],
    "opportunities_count": 0,
    "summary": {},
    "stake_amount": 100.0
  },
  "middles": {
    "opportunities": [],
    "opportunities_count": 0,
    "summary": {},
    "stake_amount": 100.0,
    "defaults": {
      "min_gap": 1.5,
      "sort": "ev",
      "positive_only": true
    }
  },
  "plus_ev": {
    "opportunities": [],
    "opportunities_count": 0,
    "summary": {},
    "defaults": {
      "sharp_book": "pinnacle",
      "min_edge_percent": 1.0,
      "bankroll": 1000.0,
      "kelly_fraction": 0.25
    }
  },
  "sport_errors": {},
  "partial": false,
  "regions": ["us", "eu"],
  "commission_rate": 0.05,
  "stale_event_filters": {},
  "purebet": {},
  "custom_providers": {},
  "provider_snapshot_paths": {},
  "cross_provider_match_report_path": "data/provider_snapshots/cross_provider_match_report.json",
  "timings": {
    "total_ms": 0,
    "steps": [],
    "sports": []
  },
  "request_log": {
    "enabled": true,
    "path": "data/request_logs/requests_20260312.jsonl",
    "requests_logged": 12
  },
  "scan_saved_path": "data/scans/scan_20260312_120000_abc123.json"
}
```

- 状态码：
  - `200`：扫描成功，或部分成功但已返回结果
  - `400`：请求参数不合法
  - `500`：扫描执行失败且没有可返回的有效结果

- 错误示例：

无效 JSON：

```json
{
  "success": false,
  "error": "Invalid JSON payload",
  "error_code": 400
}
```

非对象 JSON：

```json
{
  "success": false,
  "error": "Scan payload must be a JSON object",
  "error_code": 400
}
```

缺少地区：

```json
{
  "success": false,
  "error": "At least one region must be selected",
  "error_code": 400
}
```

缺少 API Key 且需要调用 The Odds API：

```json
{
  "success": false,
  "error": "API key is required",
  "error_code": 400
}
```

---

## 3. 历史记录接口

- 路径：`/history`
- 方法：`GET`
- 请求参数：

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `mode` | `string` | 否 | `arbitrage` / `middles` / `ev`，默认全部 |
| `limit` | `int` | 否 | 返回数量，默认 `200`，最大 `1000` |

- 响应结构：

```json
{
  "success": true,
  "records": [],
  "count": 0
}
```

- 状态码：
  - `200`：读取成功
  - `500`：读取历史失败

- 错误示例：

```json
{
  "success": false,
  "error": "history file read error"
}
```

---

## 4. 历史统计接口

- 路径：`/history/stats`
- 方法：`GET`
- 请求参数：无
- 响应结构：

```json
{
  "success": true,
  "enabled": true,
  "dir": "data/history",
  "modes": {
    "arbitrage": { "count": 0, "size_bytes": 0 },
    "middles": { "count": 0, "size_bytes": 0 },
    "ev": { "count": 0, "size_bytes": 0 }
  }
}
```

- 状态码：
  - `200`：读取成功
  - `500`：统计失败

- 错误示例：

```json
{
  "success": false,
  "error": "failed to read stats"
}
```

---

## 5. Provider 快照接口

- 路径：`/provider-snapshots/{provider_key}`
- 方法：`GET`
- 请求参数：

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `provider_key` | `string` | 是 | Provider 标识，如 `polymarket` |

- 响应结构：

```json
{
  "success": true,
  "provider_key": "polymarket",
  "snapshot": {
    "provider_name": "Polymarket",
    "sports": [],
    "events": []
  }
}
```

- 状态码：
  - `200`：读取成功
  - `400`：`provider_key` 非法
  - `404`：快照文件不存在
  - `500`：文件读取或 JSON 解析失败

- 错误示例：

```json
{
  "success": false,
  "error": "No snapshot found for provider 'polymarket'"
}
```

---

## 6. Provider 运行时接口

- 路径：`/provider-runtime/{provider_key}`
- 方法：`GET`
- 请求参数：

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `provider_key` | `string` | 是 | 当前仅 `polymarket` 有返回值 |

- 响应结构：

```json
{
  "success": true,
  "provider_key": "polymarket",
  "enabled": true,
  "started": true,
  "ready": true,
  "status": {
    "started": true
  }
}
```

- 状态码：
  - `200`：状态可用
  - `404`：不支持该 Provider 或状态不可用

- 错误示例：

```json
{
  "success": false,
  "error": "No runtime status is available for provider 'unknown'"
}
```

---

## 7. 跨 Provider 报告接口

- 路径：`/cross-provider-report`
- 方法：`GET`
- 请求参数：无
- 响应结构：

```json
{
  "success": true,
  "report": {
    "saved_at": "2026-03-12T12:00:00Z",
    "summary": {
      "providers_considered": [],
      "provider_event_counts": {},
      "total_raw_records": 0,
      "total_match_clusters": 0,
      "overlap_clusters": 0,
      "tolerance_minutes": 15
    },
    "clusters": [],
    "single_provider_samples": []
  }
}
```

- 状态码：
  - `200`：读取成功
  - `400`：报告路径非法
  - `404`：报告文件不存在
  - `500`：文件读取或 JSON 解析失败

- 错误示例：

```json
{
  "success": false,
  "error": "Cross-provider report file not found"
}
```
