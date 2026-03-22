# SPEC

本规范以 [PRD](./01_PRD.md)、[ARCH](./02_ARCH.md)、[API Doc](./03_API_DOC.md) 为唯一上游约束，后续实现必须与这三份文档保持一致。

## 1. 代码规范

- 后端统一使用 Python，公开函数与路由参数保持显式命名。
- 业务逻辑与 IO 逻辑分层：
  - `app.py` 负责路由编排与请求校验
  - `scanner.py` 负责扫描与计算
  - `providers/` 负责外部源接入
- 新增公共函数应优先补充类型注解，保持与现有代码风格一致。
- 路由层返回统一结构：
  - 成功：`jsonify(payload), 200`
  - 失败：`jsonify({"success": false, "error": "...", ...}), <status>`
- 不允许在未定义文档字段的情况下扩展响应结构。

## 2. 目录规则

- Web 入口固定放在项目根目录：`app.py`
- 扫描核心固定放在：`scanner.py`
- 配置固定放在：`config.py`、`settings.py`、`settings.json`
- Provider 适配器固定放在：`providers/<provider_name>.py`
- 前端模板固定放在：`templates/`
- 静态资源固定放在：`static/`
- 测试固定放在：`tests/`
- 运行期文件固定落在：`data/`
- 项目级设计文档固定落在：`docs/project_docs/`

## 3. 命名规范

- Python 函数、变量、模块名使用 `snake_case`
- 常量使用 `UPPER_SNAKE_CASE`
- Provider 统一使用规范化 key：
  - `betdex`
  - `bookmaker_xyz`
  - `sx_bet`
  - `polymarket`
- 接口路径使用资源语义命名，例如 `/history/stats`、`/provider-runtime/{provider_key}`
- 新增配置项命名必须与环境变量一致，采用大写下划线格式

## 4. 注释要求

- 复杂数学逻辑、概率估计、Provider 兼容分支允许写简短注释。
- 模块级文件保留简洁 docstring，描述职责和关键配置。
- 不写重复代码本身含义的注释。
- 文档或日志中出现敏感字段时，必须明确标识为脱敏值。

## 5. 异常处理

- `app.py` 必须先校验请求体是否为合法 JSON 对象。
- 参数转换失败时必须回退到默认值或返回结构化 400。
- 扫描器内部 Provider 失败应尽可能降级为 `partial=true`，而非中断全部结果。
- 后台通知、历史保存、Provider 预热失败只记录日志，不阻塞主请求返回。
- 前端自动扫描必须支持“暂停后续调度”而不强制中断当前扫描；当用户请求暂停时，当前轮次自然结束，后续自动扫描停止，并恢复设置可编辑状态。
- 不允许直接把未清洗的 API Key、Cookie、Bearer Token 输出到响应、日志或持久化文件。

## 6. 日志规范

- 常规异常统一使用 `logging.warning(..., exc_info=True)` 记录非阻塞失败。
- 开启 `SCAN_REQUEST_LOG_ENABLED` 时，请求日志落盘到 `data/request_logs/`，格式为 JSONL。
- 请求日志需要满足：
  - URL 和参数脱敏
  - 响应体截断
  - 记录耗时、状态码、Provider/接口来源
- 若后续新增日志字段，必须兼容现有 `request_log` 响应结构，不破坏 [API Doc](./03_API_DOC.md)。

## 7. 配置约束

- 配置优先级遵循当前实现：
  - 已存在环境变量 / `.env`
  - `settings.json` 中未被环境变量覆盖的项
  - 代码默认值
- 新增配置必须同步更新：
  - `settings.json`
  - `config.py` 或对应模块默认值
  - `docs/project_docs/01_PRD.md`
  - `docs/project_docs/03_API_DOC.md`（若影响接口）

## 8. Provider 扩展规范

- 新增 Provider 必须同时完成以下动作：
  - 新建 `providers/<provider>.py`
  - 在 `providers/__init__.py` 注册 fetcher、title、alias
  - 返回与扫描器兼容的标准化事件结构
  - 在 `tests/` 中补 Provider 解析或分段测试
  - 更新本目录下 `ARCH`、`API Doc`、`TEST CASE` 与 `PROJECT SUMMARY`

## 9. Provider 校验规范

- 自定义 Provider 的在线接口核验必须优先参考官方文档。
- 如果在线扫描出现以下任一情况，必须先核文档再改代码：
  - HTTP 4xx / 5xx / 52x
  - 响应字段缺失或类型漂移
  - 盘口映射明显不对
  - 扫描结果出现异常高 ROI 或明显错配赛事
- `bookmaker_xyz` 必须优先使用 Azuro 官方公开 `market-manager` + 官方 dictionaries 作为主取数来源；旧 GraphQL / 前端 bundle 只允许作为兼容性回退。
- Azuro 动态联赛 key 统一使用 `azuro__<sport_slug>__<league_slug>__<country_slug>`，不允许自由定义其它格式。
- Provider 快照应保存该 Provider 的原始视图，不应被后续跨 Provider 合并结果污染。
