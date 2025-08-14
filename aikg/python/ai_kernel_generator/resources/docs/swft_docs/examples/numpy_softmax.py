from swft.core import *
from swft.api import *
import os

OP_NAME = "softmax"


@sub_kernel(core_num=8)
def softmax_kernel(gm_input, gm_output):
    BATCH_SIZE = 16
    DIM = 16384
    BLOCK_DIM = 8
    SAMPLES_PER_CORE = BATCH_SIZE // BLOCK_DIM

    core_idx = get_block_idx()
    start_batch = core_idx * SAMPLES_PER_CORE

    for i in range(SAMPLES_PER_CORE):
        current_batch = start_batch + i
        ub_input = slice_to_ub(gm_input, [current_batch, 0], [1, DIM])
        # 注意：为了保证结果的正确性，累加操作必须转为fp32
        ub_input_fp32 = vconv(ub_input, "FP32")

        ub_max = vcmax(ub_input_fp32, reduce_axis=-1)
        # 注意：为了充分利用UB，reduce后为标量情况下，使用向量-标量运算替代broadcast，注意仅仅在双目运算中使用
        ub_sub = vsubs(ub_input_fp32, move_to_scalar(ub_max))
        ub_exp = vexp(ub_sub)

        ub_sum = vcadd(ub_exp, reduce_axis=-1)
        # 注意：为了充分利用UB，reduce后为标量情况下，使用向量-标量运算替代broadcast，注意仅仅在双目运算中使用
        ub_div = vdivs(ub_exp, move_to_scalar(ub_sum))

        ub_result = vconv(ub_div, "FP16")
        insert_to_gm(gm_output, ub_result, [current_batch, 0], [1, DIM])


def softmax_swft_numpy(device_id=0):
    set_context("310P")
    input0 = Tensor("GM", "FP16", [16, 16384], "ND", False)
    output0 = Tensor("GM", "FP16", [16, 16384], "ND", False)
    softmax_kernel(input0, output0)

    # 使用动态路径
    current_dir = os.path.dirname(__file__)
    cce_path = os.path.join(current_dir, f"{OP_NAME}", f"{OP_NAME}.cce")
    compile_kernel(cce_path, OP_NAME)
    exec_kernel(OP_NAME, locals(), inputs=['input0'], outputs=['output0'], device_id=device_id)
