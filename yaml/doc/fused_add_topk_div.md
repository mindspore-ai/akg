# fused_add_topk_div算子

## 描述
fused_add_topk_div算子实现了Sigmoid、Add、GroupTopk、Gather、ReduceSum、RealDiv、Muls算子的功能融合。支持两种模式：常规模式（物理专家模式）和逻辑专家模式。

在常规模式下，算子输出物理专家ID；在逻辑专家模式下，算子通过映射表将物理专家映射到逻辑专家，并输出逻辑专家ID。

## 输入参数

| Name | DType | Shape | Optional | Inplace | Format | Description |
|------|-------|-------|----------|---------|--------|-------------|
| x | Tensor | [a, b] | No | No | ND | 输入tensor，数据类型为float16/float32/bf16 |
| add_num | Tensor | [b] | No | No | ND | 输入tensor，用于与x相加，数据类型和格式与x一致 |
| group_num | int |  | No | No |  | 输入标量, 分组数量 |
| group_topk | int |  | No | No |  | 输入标量, 选择k个组 |
| n | int | [b] |  | No |  | 输入标量,组内选择n个最大值求和 |
| k | int | [b] |  | No |  | 输入标量,topk选择前k个值 |
| activate_type | int |  | No | No |  | 激活类型 |
| is_norm | bool | [b] | | No |  | 是否归一化 |
| scale | float | [b] |  | No |  | 归一化后的乘系数 |
| mapping_num | Tensor | [b] | Yes | No | ND | enableExpertMapping为true时输入，每个物理专家被映射到的逻辑专家数量，数据类型int32 |
| mapping_table | Tensor | [b, c] c<=128 | Yes | No | ND | enableExpertMapping为true时输入，物理专家/逻辑专家映射表，数据类型int32 |
| enable_expert_mapping | bool | | No | No |  | 是否使能物理专家向逻辑专家的映射。false时输入2个tensor，true时输入4个tensor。 |


注意：
- enableExpertMapping参数控制是否启用逻辑专家模式。当enableExpertMapping为false时，输入只有x和add_num；当为true时，输入包括x、add_num、mapping_num和mapping_table。
- a表示batch大小，b表示专家数量，c表示最大冗余专家数（最多128）。

## 输出参数

| Name | DType | Shape | Description |
|------|-------|-------|-------------|
| weight | Tensor | [a, k] | 输出tensor，数据类型float32 |
| indices | Tensor | [a, k] | 输出tensor，数据类型int32 |

## 特殊说明
- b必须为groupNum的整数倍。
- groupTopk <= groupNum。
- k <= b。
- b >= groupNum * n。
- b <= groupNum * 32。
- 若b >= 32，则groupNum = 8。
- mappingNum中的元素值范围：0 <= 元素值 < c。
- 不支持空tensor场景。

## 使用示例
### 基本使用示例（常规模式）
```python
import mindspore as ms
import numpy as np
import ms_custom_ops
import os

ms.set_device("Ascend")

def jit(func):
    @wraps(func)
    def decorator(*args, **kwargs):
        if ms.get_context("mode") == "PYNATIVE_MODE":
            return func(*args, **kwargs)
        return ms.jit(func, jit_level="O0", infer_boost="on")(*args, **kwargs)

    return decorator


class AsdFusedAddTopKDivCustom(ms.nn.Cell):
    def __init__(self):
        super().__init__()

    @jit
    def construct(
        self, x, add_num, group_num, group_topk, n, k, activate_type, is_norm, scale
    ):
        return ms_custom_ops.fused_add_topk_div(
            x, add_num, group_num, group_topk, n, k, activate_type, is_norm, scale
        )

a, b, group_num, group_topk, n, k = [8, 4, 2, 2, 2, 2]
activate_type = 0  # 算子只支持0
is_norm = True  # True时 会乘scale
scale = 2.5  # 暂时固定
os.environ["USE_LLM_CUSTOM_MATMUL"] = "off"
os.environ["INTERNAL_PRINT_TILING"] = "on"
os.environ["MS_DISABLE_INTERNAL_KERNELS_LIST"] = ""
os.environ["MS_ENABLE_INTERNAL_BOOST"] = "off"
context.set_context(mode=mode, device_target="Ascend")
context.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
x_np = np.random.randn(a, b)
add_num_np = np.random.randn(b)
x_t = Tensor(x_np).astype(ms.bfloat16)
add_num_t = Tensor(add_num_np).astype(ms.bfloat16)

net = AsdFusedAddTopKDivCustom() 
weight, indices = net(
                    x_t,
                    add_num_t,
                    group_num,
                    group_topk,
                    n,
                    k,
                    activate_type,
                    is_norm,
                    scale,
                )
```
