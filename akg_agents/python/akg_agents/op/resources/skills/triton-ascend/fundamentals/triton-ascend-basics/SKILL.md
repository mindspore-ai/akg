---
name: triton-ascend-basics
description: "Triton Ascend 编程基础，包括核心概念（program_id、block、grid）、内核函数结构、装饰器用法和标准代码模式。适用使用 Triton Ascend、需要了解基本语法结构的任意内核代码生成场景"
category: fundamental
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
  operator_patterns: "all"
---

# Triton Ascend 编程基础

## 标准内核结构（交错循环）

```python
@triton.jit
def kernel(
    output_ptr, input_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr, CORE_NUM: tl.constexpr,
):
    pid = tl.program_id(0)
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    for block_id in range(pid, num_blocks, CORE_NUM):
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        result = compute(data)
        tl.store(output_ptr + offsets, result, mask=mask)
```

## 内核启动模板

```python
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        try:
            self.VEC_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 40)
        except:
            self.VEC_CORE_NUM = 40

    def forward(self, x):
        out = torch.empty_like(x)
        BLOCK_SIZE = 1024
        grid = (self.VEC_CORE_NUM,)  # Ascend: 固定为核心数
        kernel[grid](out, x, x.numel(), BLOCK_SIZE=BLOCK_SIZE, CORE_NUM=self.VEC_CORE_NUM)
        return out
```

## 边界处理

```python
offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offsets < n_elements
data = tl.load(ptr + offsets, mask=mask, other=0.0)
result = tl.where(condition, true_val, false_val)
```

## Autotune 用法

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def kernel(..., BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pass
```

### Autotune 关键要点
1. **grid 必须使用 lambda**: `grid = lambda meta: (...)`
2. **调用时不传 configs 参数**: autotune 自动传入
3. **configs 参数必须是 constexpr**
4. **key 参数**: 指定哪些维度变化时重新 autotune
5. **Ascend 不支持调优**: num_warps / num_ctas / num_stages 等参数

## 核心数选择（重要）

Ascend NPU 有两类计算核心，必须根据算子类型正确选择：

- **VEC_CORE_NUM（向量核心）**：用于 element-wise、reduce、softmax、归一化等 **不含 tl.dot** 的算子
- **CUBE_CORE_NUM（矩阵核心）**：用于 matmul、attention 等 **包含 tl.dot** 的算子

**硬约束**：涉及 `tl.dot` / 矩阵乘法运算的算子**必须**使用 CUBE_CORE_NUM，混合运算（先 matmul 再 elementwise 后处理）也使用 CUBE_CORE_NUM。核心数获取代码和详细策略见 grid-config 文档。

## 输出张量创建

- 输出张量用 `torch.empty` / `torch.empty_like`（避免 `zeros`/`ones` 初始化开销）
- `torch.empty_like()` 创建的输出默认连续

## Ascend Triton 不支持的 API

以下 API 在 CUDA Triton 中存在，但在 Ascend Triton 中**不支持**，使用会导致编译错误：

| 不支持的 API | 替代方案 |
|-------------|---------|
| `tl.any` / `tl.all` | `tl.sum(mask.to(tl.int32)) > 0` |
| `tl.histogram` | 手动实现分桶逻辑 |
| `tl.sort` | 手动排序或分阶段比较 |
| `tl.gather` / `tl.scatter` (部分) | `tl.load` / `tl.store` + 索引计算 |
| `num_warps` / `num_ctas` / `num_stages` (autotune 参数) | Ascend 不需要，忽略即可 |
