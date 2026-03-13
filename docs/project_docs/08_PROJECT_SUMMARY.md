# 项目总结

## 1. 项目结构说明

Edge Scanner 采用“单体 Web 服务 + 多 Provider 适配器 + 本地文件持久化”的结构：

- `app.py`：Flask 入口、页面渲染、路由编排、后台服务管理
- `scanner.py`：扫描核心，负责数据拉取、归并、套利/中间盘/+EV 计算
- `providers/`：外部赔率源适配器
- `history.py`：历史机会记录与统计
- `notifier.py`：Webhook / Telegram 通知
- `templates/` + `static/`：前端页面和样式
- `data/`：运行期生成的扫描、历史、快照和日志文件
- `tests/`：回归测试和接口输入测试

## 2. 启动方式

1. 安装依赖：

```bash
pip install -r requirements.txt
```

2. 配置运行参数：
   - 通过 `.env`
   - 或通过 `settings.json`
   - 环境变量优先级高于 `settings.json`

3. 启动项目：

```bash
python app.py
```

可选端口：

```bash
python app.py --port 5050
```

启动后会创建 `data/` 目录，并尝试自动打开 `http://localhost:<port>/`。

## 3. 部署方式

- 当前仓库默认部署方式是“本地单机 Flask 服务”。
- 持久化依赖本地目录：
  - `data/history`
  - `data/scans`
  - `data/provider_snapshots`
  - `data/request_logs`
- 仓库当前未提供容器文件、进程守护配置或云部署脚本，因此文档确认的可交付部署方式仅为本地运行。

## 4. 维护说明

- 新增功能必须遵循本目录的固定工作流：`PRD -> ARCH -> API -> SPEC -> TEST CASE -> CODE -> TEST -> SUMMARY`
- 新增 Provider 时，必须同步更新注册表、测试和文档。
- 若调整配置项，需同步更新：
  - `settings.json`
  - `config.py`
  - `docs/project_docs/`
- 自定义 Provider 的维护不能只看单元测试；每次相关改动后，都要执行官方文档对照和一次实时扫描抽检。
- 若出现历史或请求日志膨胀，应优先检查：
  - `HISTORY_MAX_RECORDS`
  - `SCAN_SAVE_ENABLED`
  - `SCAN_REQUEST_LOG_ENABLED`

## 5. 常见问题

### Q1：为什么 `/scan` 提示 `API key is required`？

当本次扫描需要调用 The Odds API 且未提供 `apiKey` / `apiKeys` 时，会返回该错误。若只想跑自定义 Provider，需要显式选择 Provider 路径。

### Q2：为什么 `/provider-runtime` 只有 `polymarket` 可用？

当前实现中，运行时状态接口只接入了 `providers/polymarket.py` 的实时管理器，其它 Provider 没有暴露 runtime 状态。

### Q3：为什么 `/provider-snapshots/...` 或 `/cross-provider-report` 返回 404？

只有在启用快照写入并且完成过至少一次扫描后，对应文件才会生成。

### Q4：为什么明明没选 `eu`，结果里还是出现了 `eu`？

系统会为 Sharp Book 自动补齐必要地区，保证 +EV 参考来源可用。

### Q5：历史、通知写失败会不会影响扫描结果？

不会。当前实现中这两类能力都被设计为非阻塞副作用，失败只记录 warning 日志，不阻断主响应。
