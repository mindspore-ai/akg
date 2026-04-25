---
name: performance-summary
description: 用于汇总和对比多个算子的性能测试结果。当用户请求汇总多个算子的性能结果、对比算子 benchmark、生成性能报告、分析多个 kernel 的加速比时使用此 Skill。
---

# 性能汇总

汇总和对比多个算子的性能指标。

## 工作流程

1. **收集性能数据**
   - 使用 `read_file` 读取各算子的 `verify_<op_name>/result.json`
   - 提取关键指标：latency_ms, throughput, memory_usage_mb, speedup

2. **生成对比表格**
   ```
   | 算子   | 延迟 (ms) | 吞吐量 | 内存 (MB) | 加速比 |
   |--------|-----------|--------|-----------|--------|
   | relu   | 0.15      | 1000   | 128       | 2.3x   |
   ```

3. **分析结果**
   - 找出最优/最差的算子
   - 标注性能异常点
   - 提出优化目标

4. **生成报告**
   使用模板：`## 测试环境` → `## 性能对比` → `## 分析` → `## 优化建议`

## 数据位置

性能结果存储在验证目录中：
- `verify_<op_name>/result.json` - 主要结果
- `verify_<op_name>/profiling.json` - 详细性能分析（如有）

## 关键指标

| 指标 | 说明 |
|------|------|
| latency_ms | 单次执行延迟（毫秒） |
| throughput | 每秒操作数 |
| speedup | 相对基准（torch）的加速比 |
| memory_usage_mb | 峰值内存占用（MB） |

## 示例

**用户**: "帮我汇总 relu 和 sigmoid 算子的性能"

**Agent 执行**:
1. 调用 `read_file("verify_relu/result.json")`
2. 调用 `read_file("verify_sigmoid/result.json")`
3. 提取指标，构建对比表格
4. 生成汇总和优化建议

## 脚本调用

可使用 `scripts/collect_metrics.py` 批量收集并打印性能汇总：

```python
from collect_metrics import collect_metrics, print_results

# 打印格式化的性能汇总结果
print_results()
```

## Skill 验证

使用 `--verify` 参数验证当前 skill 是否可用：

```bash
python scripts/collect_metrics.py --verify
```

验证内容包括：
- SKILL.md 文件是否存在且格式正确
- 核心函数是否可调用
- 依赖模块是否可用
- 模拟执行测试

**输出示例**:
```
============================================================
性能汇总结果
============================================================
| Operator | Latency (ms) | Throughput | Memory (MB) | Speedup | Correct |
|----------|--------------|------------|-------------|---------|---------|
| matmul   | 1.234        | 5000       | 256.0       | 1.80x   | ✅      |
| relu     | 0.150        | 10000      | 128.0       | 2.30x   | ✅      |
| sigmoid  | 0.180        | 8000       | 128.0       | 2.10x   | ✅      |
============================================================
```

