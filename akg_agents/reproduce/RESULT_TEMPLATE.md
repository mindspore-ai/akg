# 复现结果记录模板

每次运行复现脚本后，按此模板记录结果，便于横向对比。

---

## 环境规范

| 项目 | 值 |
|------|---|
| arch | ascend910b4 |
| torch | 2.7.1 |
| triton_ascend | 3.2.0 |
| commit | (git short hash) |
| llm_model | (模型名称) |
| python | 3.10.0 |
| 日期 | 2026-xx-xx |

## 测试结果

| 算子(序号) | 结果 | 耗时(s) | speedup | 备注 |
|-----------|------|---------|---------|------|
| 05 StreamWrite | PASS/FAIL | | | |
| 19_ReLU | PASS/FAIL | | | |
