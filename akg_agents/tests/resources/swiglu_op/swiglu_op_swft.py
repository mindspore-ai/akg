from core import Tensor, set_context, sub_kernel, compile_kernel
from api import *

CORE_NUM = 8


@sub_kernel(core_num=CORE_NUM)
def swiglu_op_impl_npu(gm_input, gm_output):
    dim0 = 40
    dim1 = 256
    dim0_split = 5
    dim1_split = 128

    block_idx = get_block_idx()
    start_offset = block_idx * dim0_split

    # Load input data from GM to UB
    ub_input_half = slice_to_ub(gm_input, [start_offset, 0], slicesize=[dim0_split, dim1])

    # Split input into two parts and convert to FP32
    ub_input0_float = vconv(slice_to_ub(ub_input_half, [0, 0], slicesize=[dim0_split, dim1_split]), "FP32")
    ub_input1_float = vconv(slice_to_ub(ub_input_half, [0, dim1_split], slicesize=[dim0_split, dim1_split]), "FP32")

    # Compute SwiGLU: x * sigmoid(x) * y
    # sigmoid(x) = x / (1 + exp(-x))
    neg_tile = vmuls(ub_input0_float, -1.0)
    exp_tile = vexp(neg_tile)
    exp_tile = vadds(exp_tile, 1.0)
    sigmoid_tile = vdiv(ub_input0_float, exp_tile)
    result_tile = vmul(sigmoid_tile, ub_input1_float)

    # Convert result back to FP16 and store to GM
    ub_output_half = vconv(result_tile, "FP16")
    insert_to_gm(gm_output, ub_output_half, [start_offset, 0], slicesize=[dim0_split, dim1_split])


def swiglu_op_host_run():
    set_context("310P")
    gm_input = Tensor("GM", "FP16", [40, 256], format="ND", multi_core=False)
    gm_output = Tensor("GM", "FP16", [40, 128], format="ND", multi_core=False)

    swiglu_op_impl_npu(gm_input, gm_output)


if __name__ == '__main__':
    swiglu_op_host_run()
    compile_kernel("./swiglu_op.cce")
