# SWFT 语言编程基础教程

SWFT是一款Ascend算子编译器，有着极简编写、高性能等特征。当前作为AIKG Ascend310P算子生成后端。

- SWFT python 表达更灵活，适合LLM代码生成
  - 支持基础Ascend语法同时，扩展出更高阶的封装（例如：数据搬移支持任意长度，在SWFT内部做实际的repeat、block、stride设置）
- 自动静态内存分配，无需现式控制buffer、设置pipeline等内容


## 使用示例
```python
from swft.core import *
from swft.api import *

OP_NAME = "tanh"

@sub_kernel(core_num=8)
def tanh_kernel(x, out):
    x_ub = move_to_ub(x)
    tanh_ub = tanh(x_ub)
    out.load(tanh_ub)
```
其中`@sub_kernel`指示了当前函数需要通过SWFT进行算子编译。`core_num`指示了编译该算子需要用到的昇腾AICore核数，`tanh_kernel`内部使用的API：`move_to_ub`，`tanh`，`Tensor.load`等，是SWFT对外提供的昇腾亲和的函数式API。

SWFT算子定义完成后，通过如下代码启动算子编译，并最终输出编译后的算子源码。
```python
def tanh_swft_numpy(device_id=0):
    set_context("310P") #指示编译的昇腾后端，当前仅支持310系列
    input0 = Tensor("GM", "FP16", [16, 256, 256], "ND", multi_core=False) # multi_core仅支持False
    output0 = Tensor("GM", "FP16", [16, 256], "ND", multi_core=False) # multi_core仅支持False
    tanh_kernel(input0, output0)

    # 使用动态路径
    current_dir = os.path.dirname(__file__)
    cce_path = os.path.join(current_dir, f"{OP_NAME}", f"{OP_NAME}.cce") # 指示算子编译输出文件的最终位置，输出为CCE代码。
    compile_kernel(cce_path, OP_NAME) # 编译算子
    exec_kernel(OP_NAME, locals(), inputs=['input0'], outputs=['output0'], device_id=device_id) # 执行算子
```
其中所有的输入输出Tensor必须按照input/output+index的方式命名，例如：input0, input1, output0, output1等，并且`multi_core`仅支持False。通过compile_kernel编译算子，exec_kernel执行算子，最终输出算子执行结果。

## SWFT 参考代码

```python
hidden = 7168
@sub_kernel(core_num=8)
def moe_token_unpermute_op_impl_npu(gm_permute_token, gm_sorted_idx, gm_probs, gm_output, tiling):
    block_idx = get_block_idx()

    # Initialize local output buffer
    ub_idx = move_to_ub(gm_sorted_idx)
    prob_ub = move_to_ub(gm_probs)
    ub_tiling = move_to_ub(tiling)
    token_num = move_to_scalar(ub_tiling[0])
    top_k = move_to_scalar(ub_tiling[1])
    tokens_per_core = move_to_scalar(ub_tiling[2])
    for i in dynamic_loop(tokens_per_core):
        start_token = block_idx * tokens_per_core + i
        tmp_s = Scalar("FP16", 0.0)
        local_out = vector_dup(tmp_s, [1, hidden], False)
        for k in dynamic_loop(top_k):
            # Load sorted_idx
            idx = move_to_scalar(ub_idx[k * token_num + start_token])
            # Load permute_token row
            dst_row = slice_to_ub(gm_permute_token, [idx, 0], slicesize=[1, hidden])
            # Load prob
            prob = move_to_scalar(prob_ub[start_token, k])
            # # Compute weighted row and accumulate
            weighted_row = vmuls(dst_row, prob)
            local_out = vadd(local_out, weighted_row)

        # Write back to GM
        insert_to_gm(gm_output, local_out, [start_token, 0], slicesize=[1, hidden])
```
