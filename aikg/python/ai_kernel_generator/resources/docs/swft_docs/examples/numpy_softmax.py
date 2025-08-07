from swft.core import *
from swft.api import *
import os

OP_NAME = "softmax"


@sub_kernel(core_num=8)
def softmax_kernel(gm_input, gm_output):
    # Hardcoded parameters from tiling
    BATCH_SIZE = 16
    DIM = 16384
    BLOCK_DIM = 8
    SAMPLES_PER_CORE = BATCH_SIZE // BLOCK_DIM  # 2

    # Get core index
    core_idx = get_block_idx()
    start_batch = core_idx * SAMPLES_PER_CORE

    # Process each sample in pipeline
    for i in range(SAMPLES_PER_CORE):
        current_batch = start_batch + i

        # Load input data to UB
        ub_input = slice_to_ub(gm_input, [current_batch, 0], [1, DIM])

        # 1. Find max value (along dim axis)
        ub_max = vcmax(ub_input, reduce_axis=-1)

        # 2. Subtract max value
        ub_sub = vsubs(ub_input, move_to_scalar(ub_max))

        # 3. Compute exp
        ub_exp = vexp(ub_sub)

        # 4. Sum exp values
        ub_sum = vcadd(ub_exp, reduce_axis=-1)

        # 5. Divide each element by sum
        ub_div = vdivs(ub_exp, move_to_scalar(ub_sum))

        # Write result back to GM
        insert_to_gm(gm_output, ub_div, [current_batch, 0], [1, DIM])


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
