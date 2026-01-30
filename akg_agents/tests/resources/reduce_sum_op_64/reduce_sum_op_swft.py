from core import *
from api import *


@sub_kernel(core_num=8)
def reduce_sum_op_impl_npu(gm_input, gm_output):
    total_rows = 64
    cols = 128
    BLOCK_DIM = 8

    block_idx = get_block_idx()
    rows_per_core = total_rows // BLOCK_DIM
    start_row = block_idx * rows_per_core

    # Load input data from GM to UB
    ub_input = slice_to_ub(gm_input, [start_row, 0], slicesize=[rows_per_core, cols])
    # ub_input_float = vconv(ub_input, "FP32")

    # Compute row-wise reduce_sum
    ub_output = vcadd(ub_input, reduce_axis=-1)
    # ub_output = vconv(ub_output_float, "FP16")

    # Store result back to GM
    insert_to_gm(gm_output, ub_output, [start_row], slicesize=[rows_per_core])


def reduce_sum_op_host_run():
    set_context("310P")

    # Define input and output tensors
    gm_input = Tensor("GM", "FP16", [64, 128], format="ND", multi_core=False)
    gm_output = Tensor("GM", "FP16", [64], format="ND", multi_core=False)

    # Execute the NPU kernel
    reduce_sum_op_impl_npu(gm_input, gm_output)


if __name__ == '__main__':
    reduce_sum_op_host_run()
    compile_kernel("./reduce_sum_op.cce")
