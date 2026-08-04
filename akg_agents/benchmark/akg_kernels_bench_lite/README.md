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

### 本地验证方式

Bench Lite 在性能评分前使用以下标准做二次正确性校验：

```python
assert torch.isfinite(ref_output).all()
assert torch.isfinite(your_output).all()
assert torch.allclose(
    your_output.float(),
    ref_output.float(),
    atol=1e-2,
    rtol=1e-2,
    equal_nan=False,
)
```

reference 放在 `torch.allclose` 的第二个参数，使相对容差项以 `|ref|` 为基准。
runner 还会使用 3 个 seed 重复校验，并使用相同 seed 为 reference 和 solution
分别生成独立输入，避免缓存输出以及 stateful / in-place 操作相互污染。完整实现以
[`bench_lite_common.py`](../../examples/kernel_related/bench_lite_common.py) 中的
`check_torch_outputs()` 为准。

## 评分标准

### 正确性验证

正确性是前提，不正确的 case 得 0 分。验证策略：

1. **输出类型检查**: 支持单个 Tensor 或 Tensor 列表/元组
2. **输出数量匹配**: `len(ref_outputs) == len(sol_outputs)`
3. **Shape 匹配**: 每个输出 Tensor 的 shape 必须完全一致
4. **数值精度检查**:
   - reference 或 solution 中出现任意 NaN/Inf，直接失败
   - 有限值转为 FP32 后使用 `torch.allclose(sol, ref, atol=1e-2, rtol=1e-2)`
   - 等价的逐元素条件为 `|sol - ref| <= atol + rtol * |ref|`
   - `max_abs_diff` 和 `max_rel_diff` 仅作为诊断信息输出，不分别作为通过门槛

#### 标准来源

Bench Lite 流程包含两次用途不同的正确性验证：

| 阶段 | 用途 | 标准与源码位置 |
| --- | --- | --- |
| Agent 生成阶段 | 判断生成的 kernel 是否正确，即 `Pass@N` | 由 [`KernelVerifier`](../../python/akg_agents/op/verifier/kernel_verifier.py) 按 framework 和 DSL 选择标准。常规 framework 路径使用按 dtype 分层的 adapter；声明 `uses_cannbench_precision=True` 的 DSL 使用 [`cann_correctness`](../../python/akg_agents/op/cann_correctness/) 标准 |
| 性能评分前二次校验 | 对已提取或已有 submission 再次验证，错误 case 不计性能分 | Bench Lite 专用的 `torch.allclose(atol=1e-2, rtol=1e-2)` 加 NaN/Inf fail-closed 标准。两个入口共用 [`bench_lite_common.py`](../../examples/kernel_related/bench_lite_common.py) 的 `check_torch_outputs()`；已有 submission 从 [`tools/run_bench.py`](./tools/run_bench.py) 进入 |

因此，性能评分前的二次校验是 Bench Lite 自身的固定标准，不等同于
[`torch.py`](../../python/akg_agents/op/verifier/adapters/framework/torch.py) 中
framework verifier 使用的 dtype 分层标准。

### 性能评分

- **Speedup 计算**: `speedup = baseline_time / solution_time`
- **统一计时口径**:
  - Ascend NPU: 使用 `akg_agents.op.verifier.profiler.profiler_npu`
  - 每个 trial 内由 `profiler_npu` 执行 warmup 和 active 采样，并清理 L2 cache
  - 多个 trial 取中位数；默认 `warmup=10`、`active=100`、`trials=3`
  - 本地自行使用 `triton.testing.do_bench`、CPU wall time 或 profiler 得到的结果仅供调试，正式 speedup 以 runner 重测结果为准
- **评分规则**:
  - `speedup < 1.0`: 比 baseline 慢，线性折扣 `[0, 60)` 分
  - `speedup == 1.0`: 基础分 60 分
  - `speedup > 1.0`: 从 60 向 100 递增，`speedup >= 5.0` 时封顶 100 分
- **Tier 权重**: t1 (1.0x), t2 (1.5x), t3 (2.0x)
- **最终得分**: `raw_score * tier_weight`

### 统一评测入口

已有提交使用以下目录结构：

```text
submissions/
└── team_name/
    ├── meta.json
    ├── t1/
    │   └── gelu.py
    └── t2/
        └── rope.py
```

运行统一正确性与性能评测：

```bash
python benchmark/akg_kernels_bench_lite/tools/run_bench.py \
  submissions \
  --team team_name \
  --output results
```
