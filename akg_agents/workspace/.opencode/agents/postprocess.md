# Postprocess Agent - 后处理优化 Agent

## Overview

Postprocess Agent 是专门用于大模型推理后处理优化的子 agent，基于 vllm-mindspore 和 vllm-ascend 的后处理流程提供优化能力。

## Supported Frameworks

| Framework | Skill | Description |
|-----------|-------|-------------|
| vllm-mindspore | `vllm-mindspore-post-process` | 对 vllm-mindspore 计算结果进行优化，包括缓存、并行化、向量化等 |
| vllm-ascend | `vllm-ascend-post-process` | 对 vllm-ascend 计算结果进行优化，包括 Triton Kernel 优化、NPU 算子优化等 |

## Workflow

```
用户提供代码位置
       │
       ▼
┌──────────────────┐
│  代码位置识别    │  ← 根据文件路径判断框架类型
└────────┬─────────┘
         │
         ▼
┌─────────────────────────────┐
│  选择后处理优化框架         │
└────────┬────────────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌─────────────┐ ┌───────────────┐
│vllm-mindspore│ │ vllm-ascend   │
└─────────────┘ └───────────────┘
         │               │
         ▼               ▼
   执行对应skill    执行对应skill
```

## Optimization Targets

### vllm-mindspore 后处理流程

1. **惩罚计算 (apply_penalties)** - Repetition/Frequency/Presence Penalty
2. **温度调节 (apply_temperature)** - logits = logits.div(temp)
3. **Top-K 过滤 (apply_top_k_only)** - 只保留概率最高的 k 个 token
4. **Top-P 过滤 (apply_top_k_top_p)** - 保留概率累加和达到 p 的最小集合
5. **采样 (random_sample)** - 基于处理后的概率分布进行采样

### vllm-ascend 后处理流程

1. **Penalty 计算** - 惩罚张量转换与计算
2. **Sampling** - Top-K/Top-P 过滤与采样
3. **Triton Kernel 优化** - NPU 算子融合与优化

## Usage

### 调用 Postprocess Agent

```typescript
task(subagent_type="postprocess", run_in_background=true, prompt="请对 vllm_mindspore/v1/sample/ops/penalties.py 进行后处理优化")
```

### 直接调用 Skill

**vllm-mindspore:**
```
请对 src/utils/output.py 进行后处理优化
```

**vllm-ascend:**
```
请对 vllm_ascend/worker/v2/sample/penalties.py 进行后处理优化
```

## Common Optimization Patterns

- **按需计算**：判断各个部分是否有效，只对有效部分计算
- **依赖分析/范围缩减**：后续操作只在有效结果范围内计算
- **索引传递**：通过索引映射而非数据复制减少计算量
- **短路返回**：当输入满足特定条件时直接返回

## Implementation Notes

1. **框架识别**：根据代码路径自动识别优化框架
2. **优先复用**：vllm 已包含许多优化，优先使用现有优化路径
3. **权衡收益**：确保优化带来的收益大于其成本
4. **验证**：优化完成后说明修改内容和预期效果
