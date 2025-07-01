import numpy as np
from numpy.typing import NDArray


def exp_adds_op_impl_npu(gm_input0: NDArray[np.float16], gm_output: NDArray[np.float16], tiling: list):
    # 输入输出shape: (40,256)
    total_rows, cols = gm_input0.shape

    # 并行轴：每个vector_core处理一行数据
    for row in parallel_range(0, total_rows):  # shape=(40,)
        # 从GM加载输入数据到vector_buffer
        input_part = np.copy(gm_input0[row, :])  # shape=(256,)

        # vector计算阶段
        tmp = np.exp(input_part)  # shape=(256,)
        output_part = tmp + 1.0   # shape=(256,)

        # 将结果写回GM
        gm_output[row, :] = np.copy(output_part)  # shape=(256,)


def exp_adds_op_impl_tiling(gm_input0: NDArray[np.float16], gm_output: NDArray[np.float16], tiling: list):
    BLOCK_DIM = 40  # 40 vector_core并行处理40行
    WORKSPACE_SIZE = 0
