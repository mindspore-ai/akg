import numpy as np
from numpy.typing import NDArray


def moe_token_unpermute_op_impl_tiling(gm_permute_token: NDArray[np.float16], gm_sorted_idx: NDArray[np.int32],
                                       gm_probs: NDArray[np.float16], gm_output: NDArray[np.float16], tiling: list):
    BLOCK_DIM = 8
    WORKSPACE_SIZE = 0
    return BLOCK_DIM, WORKSPACE_SIZE


def moe_token_unpermute_op_impl_npu(gm_permute_token: NDArray[np.float16], gm_sorted_idx: NDArray[np.int32],
                                    gm_probs: NDArray[np.float16], gm_output: NDArray[np.float16], tiling: list):
    token_num = gm_probs.shape[0]
    top_k = gm_probs.shape[1]
    hidden = gm_permute_token.shape[1]

    BLOCK_DIM = 8
    tokens_per_core = (token_num + BLOCK_DIM - 1) // BLOCK_DIM

    # 并行轴切分
    for i in parallel_range(BLOCK_DIM):
        # 当前核处理的数据范围
        for j in range(tokens_per_core):
            current_i = i * tokens_per_core + j

            # 初始化输出buffer (hidden,)
            out_buffer = np.zeros((hidden,), dtype=np.float16)

            for k in range(top_k):
                # 计算索引位置 (scalar操作)
                idx = k * token_num + current_i
                src_row = np.copy(gm_sorted_idx[idx])  # scalar搬运

                # 加载permute_token行 (7168,)
                permute_row = np.copy(gm_permute_token[src_row, :])

                # 加载probs值 (scalar)
                prob_val = np.copy(gm_probs[current_i, k])

                # 向量乘加计算
                temp = prob_val * permute_row  # (7168,)
                out_buffer += temp  # (7168,)

            # 写回全局内存 (7168,)
            gm_output[current_i, :] = np.copy(out_buffer)
