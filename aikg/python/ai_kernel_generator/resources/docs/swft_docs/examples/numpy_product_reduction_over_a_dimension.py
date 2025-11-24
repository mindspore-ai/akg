from swft.core import *
from swft.api import *
import os

OP_NAME = "product_reduction_over_a_dimension"


@sub_kernel(core_num=8)
def product_reduction_over_a_dimension_kernel(gm_input, gm_output):
    BATCH_SIZE = 16
    DIM1 = 256
    DIM2 = 256
    SAMPLES_PER_CORE = BATCH_SIZE // 8

    block_idx = get_block_idx()

    # 注意：如果生成的cce过长（也就是所有嵌套for循环的乘积过大），将for循环中的range替换为dynamic_loop
    for i in dynamic_loop(SAMPLES_PER_CORE):
        current_batch = block_idx * SAMPLES_PER_CORE + i

        # 注意：为了保证结果的正确性，累乘操作必须转为fp32
        init_fp32 = Scalar("FP32", 1.0)
        reduce_buf_fp32 = vector_dup(init_fp32, [1, 1, DIM2], False)

        # 注意：当前SWFT仅支持最后一根轴是reduce轴，因此最后一根轴为非reduce轴时，只能通过for循环和vadd指令替代vcadd
        for j in range(DIM1):
            ub_input = slice_to_ub(gm_input, [current_batch, j, 0], [1, 1, DIM2])
            # 注意：为了保证结果的正确性，累乘操作必须转为fp32
            ub_fp32 = vconv(ub_input, "FP32")
            reduce_buf_fp32 = vmul(reduce_buf_fp32, ub_fp32)
        ub_output = vconv(reduce_buf_fp32, "FP16")
        insert_to_gm(gm_output, ub_output, [current_batch, 0], [1, DIM2])


def product_reduction_over_a_dimension_swft_numpy(device_id=0):
    set_context("310P")
    input0 = Tensor("GM", "FP16", [16, 256, 256], "ND", False)
    output0 = Tensor("GM", "FP16", [16, 256], "ND", False)
    product_reduction_over_a_dimension_kernel(input0, output0)

    # 使用动态路径
    current_dir = os.path.dirname(__file__)
    cce_path = os.path.join(current_dir, OP_NAME, OP_NAME + ".cce")
    compile_kernel(cce_path, OP_NAME)
    exec_kernel(OP_NAME, locals(), inputs=['input0'], outputs=['output0'], device_id=device_id)
