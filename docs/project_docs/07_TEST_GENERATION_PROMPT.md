# 测试生成指令

请严格参考以下已确认文档：

1. [PRD](./01_PRD.md)
2. [架构文档](./02_ARCH.md)
3. [接口文档](./03_API_DOC.md)
4. [实现规范](./04_SPEC.md)
5. [测试用例](./05_TEST_CASE.md)

为对应代码生成单元测试和接口测试。

## 强制约束

- 测试范围必须完全对齐 [TEST CASE](./05_TEST_CASE.md)，不可随意删减或新增未定义场景。
- 测试风格与当前仓库保持一致：
  - 以 `unittest` 组织用例
  - 由 `pytest` 收集运行
- 接口测试优先覆盖：
  - `/scan`
  - `/history`
  - `/history/stats`
  - `/provider-snapshots/{provider_key}`
  - `/provider-runtime/{provider_key}`
  - `/cross-provider-report`
- 对外部依赖必须使用 mock、fixture 或本地样本文件，保证测试可直接运行。
- 所有失败分支必须校验状态码与错误字段。

## 交付要求

- 生成的测试文件放入 `tests/`
- 用例命名与 [TEST CASE](./05_TEST_CASE.md) 中的场景一一对应
- 测试说明中标明覆盖到的用例 ID
