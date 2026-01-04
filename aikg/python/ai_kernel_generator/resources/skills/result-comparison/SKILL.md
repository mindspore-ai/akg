---
name: result-comparison
description: 对比分析不同代码版本、不同 SubAgent（codeonly vs evolve）或不同配置的生成结果。当用户请求对比生成结果的性能优化点、评估 codeonly 与 evolve 的输出的算子性能优化的策略时使用此 Skill。
---

# 结果对比

对比不同生成策略的输出结果。

## 工作流程

1. **确定对比对象**
   - 不同 SubAgent：codeonly vs evolve
   - 同一算子的不同版本
   - 不同配置参数

2. **收集对比数据**
   - 代码：`verify_<op_name>/generated_kernel.py`
   - 结果：`verify_<op_name>/result.json`
   - 日志：`verify_<op_name>/verify.log`

3. **多维度对比**
   - **正确性**：验证通过/失败，误差容限
   - **性能**：延迟，加速比
   - **代码质量**：代码行数，复杂度

4. **生成对比报告**

## 对比维度

| 维度 | 对比内容 |
|------|----------|
| 正确性 | pass/fail 状态，误差容限 (1e-5, 1e-6) |
| 性能 | latency_ms，speedup 比例 |
| 代码 | 代码行数，使用的优化技术 |

## SubAgent 对比

| SubAgent | 典型场景 | 耗时 |
|----------|----------|------|
| codeonly | 快速生成，标准算子 | 30-90秒 |
| evolve | 深度优化，复杂 kernel | 3-10分钟 |

## 示例

**用户**: "对比一下 codeonly 和 evolve 生成的 matmul 算子"

**Agent 执行**:
1. 调用 `read_file("verify_matmul_codeonly/result.json")`
2. 调用 `read_file("verify_matmul_evolve/result.json")`
3. 对比正确性和性能指标
4. 根据权衡给出推荐
