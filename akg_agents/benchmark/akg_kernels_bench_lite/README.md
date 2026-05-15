# AKG Kernels Bench Lite

精简版 benchmark cases，用于 Ascend NPU 算子测试。

## 目录结构

```
akg_kernels_bench_lite/
├── t1/   (6 files)
├── t2/   (4 files)
└── t3/   (3 files)
```

## Cases 列表

### t1 (6 files)

| 文件 | 算子 |
|------|------|
| `gelu.py` | GELU 激活 |
| `fused_silu_and_mul.py` | SwiGLU (使用 `torch_npu.npu_swiglu`) |
| `matmul_basic.py` | BF16 矩阵乘 (32×8192×8192) |
| `matmul_biasadd.py` | FP16 Matmul+BiasAdd (4096×4096×4096) |
| `softmax.py` | Softmax |
| `sigmoid_scale_sum.py` | Sigmoid+Scale+Sum 融合 |

### t2 (4 files)

| 文件 | 算子 |
|------|------|
| `rope.py` | RoPE (使用 `torch_npu.npu_rotary_mul`) |
| `add_rmsnorm_cast.py` | Add+RMSNorm+Cast |
| `add_rmsnorm_quant.py` | Add+RMSNorm+Int8量化 |
| `moe_topk_softmax.py` | MoE TopK Softmax |

### t3 (3 files)

| 文件 | 算子 | 来源 |
|------|------|------|
| `causal_conv1d.py` | Causal Conv1D | Mamba (sgl-kernel-npu) |
| `decode_mla.py` | Paged MLA Decode | DeepSeek-V3 (sgl-kernel-npu) |
| `layernorm_gated.py` | Gated LayerNorm | FLA (sgl-kernel-npu) |

## 使用方法

### Case 文件结构

每个 case 文件包含：

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    """PyTorch 参考实现（Golden Reference）"""
    
    def __init__(self, ...):
        """参数与 get_init_inputs() 返回值对应"""
        super().__init__()
        ...
    
    def forward(self, ...):
        """参数与 get_inputs() 返回值对应"""
        ...
        return result

def get_inputs():
    """生成测试输入"""
    return [input1, input2, ...]

def get_init_inputs():
    """返回 Model 初始化参数"""
    return [param1, param2, ...]
```

### 实现要求

参赛者需要实现 `ModelNew` 类，要求：

```python
import torch
import torch.nn as nn
import triton
import triton.language as tl

class ModelNew(nn.Module):
    def __init__(self, ...):
        """参数与题目的 get_init_inputs() 返回值对应。"""
        super().__init__()
        ...

    def forward(self, ...):
        """参数与题目的 get_inputs() 返回值对应，输出需与 Model.forward() 一致。"""
        ...
        return result
```

### 标准调用方式

```python
from t1.gelu import Model, get_inputs, get_init_inputs

# 初始化
model = Model(*get_init_inputs())

# 运行
inputs = get_inputs()
output = model(*inputs)
```

### 正确性验证

```python
# 参考实现
ref_output = Model(*get_init_inputs())(*get_inputs())

# 你的实现
your_output = ModelNew(*get_init_inputs())(*get_inputs())

# 验证
assert torch.allclose(ref_output, your_output, atol=1e-2, rtol=1e-2)
```

## 评分标准

### 正确性验证

正确性是前提，不正确的 case 得 0 分。验证策略：

1. **输出类型检查**: 支持单个 Tensor 或 Tensor 列表/元组
2. **输出数量匹配**: `len(ref_outputs) == len(sol_outputs)`
3. **Shape 匹配**: 每个输出 Tensor 的 shape 必须完全一致
4. **数值精度检查**:
   - 计算绝对误差: `max_abs_diff = max(|ref - sol|)`
   - 计算相对误差: `max_rel_diff = max(|ref - sol| / (|ref| + 1e-8))`
   - 通过条件: `max_abs_diff <= atol (1e-2)` **且** `max_rel_diff <= rtol (1e-2)`

### 性能评分

- **Speedup 计算**: `speedup = baseline_time / solution_time`
- **评分规则**:
  - `speedup < 1.0`: 比 baseline 慢，线性折扣 `[0, 60)` 分
  - `speedup == 1.0`: 基础分 60 分
  - `speedup > 1.0`: 从 60 向 100 递增，`speedup >= 5.0` 时封顶 100 分
- **Tier 权重**: t1 (1.0x), t2 (1.5x), t3 (2.0x)
- **最终得分**: `raw_score * tier_weight`
