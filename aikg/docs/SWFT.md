# AIKG-SWFT

## SWFT 简介
SWFT是一款Ascend算子编译器，有着极简编写、高性能等特征。当前作为AIKG Ascend310P算子生成后端。

## SWFTCoder
AIKG对接SWFT算子python前端，通过直接生成SWFT python，利用SWFT直接编译成Ascend后端代码。AIKG与SWFT的对接主要集中在Coder部分，为此AIKG提供SWFTCoder作为专用的Coder来完成代码生成工作。


## AIKG-SWFT 分析

- SWFT python 表达更灵活，适合LLM代码生成
  - 支持基础Ascend语法同时，扩展出更高阶的封装（例如：数据搬移支持任意长度，在SWFT内部做实际的repeat、block、stride设置）
  - python写法与前端Sketch设计能大致对齐（`tile2slice, move2copy, vec_compute`）
- 自动静态内存分配，无需现式控制buffer、设置pipeline等内容
  - 优点：无需Designer分析调度，代码生成下限高

## 参考代码

AIKG生成的 `moe_token_unpermute_op` 示例如下：

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
