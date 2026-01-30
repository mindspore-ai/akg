from core import *
from api import *


@sub_kernel(core_num=8)
def swiglu_quant_op_impl_npu(gm_x, gm_smooth_scales, gm_output, gm_scale, gm_swiglu_out):
    BLOCK_DIM = 8
    M, N = 512, 256

    # Define UB buffers
    # x0 = Tensor("UB", "FP16", [1, N], format="ND")
    # x1 = Tensor("UB", "FP16", [1, N], format="ND")
    # x0_f32 = Tensor("UB", "FP32", [1, N], format="ND")
    # x1_f32 = Tensor("UB", "FP32", [1, N], format="ND")
    # sigmoid_t = Tensor("UB", "FP32", [1, N], format="ND")
    # swiglu_t = Tensor("UB", "FP32", [1, N], format="ND")
    # swiglu_t_fp16 = Tensor("UB", "FP16", [1, N], format="ND")
    # q_tile = Tensor("UB", "FP32", [1, N], format="ND")
    # abs_tile = Tensor("UB", "FP32", [1, N], format="ND")
    # max_tile = Tensor("UB", "FP32", [1, 1], format="ND")

    # smooth_tile_f16 = Tensor("UB", "FP16", [N], format="ND")
    # smooth_tile_f32 = Tensor("UB", "FP32", [N], format="ND")

    block_idx = get_block_idx()
    per_core_rows = M // BLOCK_DIM

    # Load smooth scales
    smooth_tile_f16 = slice_to_ub(gm_smooth_scales, [0], slicesize=[N])
    smooth_tile_f32 = vconv(smooth_tile_f16, "FP32")

    for iter_idx in range(per_core_rows):
        row_start = block_idx * per_core_rows + iter_idx

        # Load x0 and x1
        x0 = slice_to_ub(gm_x, [row_start, 0], slicesize=[1, N])
        x1 = slice_to_ub(gm_x, [row_start, N], slicesize=[1, N])

        # Convert to FP32
        x0_f32 = vconv(x0, "FP32")
        x1_f32 = vconv(x1, "FP32")

        # Compute sigmoid
        x0_mul = vmuls(x0_f32, -1.0)
        x0_exp = vexp(x0_mul)
        x0_add = vadds(x0_exp, 1.0)
        sigmoid_t = vdiv(x0_f32, x0_add)

        # Compute swiglu
        swiglu_t = vmul(sigmoid_t, x1_f32)

        # Store swiglu output
        swiglu_t_fp16 = vconv(swiglu_t, "FP16")
        insert_to_gm(gm_swiglu_out, swiglu_t_fp16, [row_start, 0], slicesize=[1, N])

        # Quantization
        q_tile = swiglu_t  # Copy to q_tile
        q_mul = vmul(q_tile, smooth_tile_f32)  # Mul smooth scale

        # # Compute scale
        q_abs = vabs(q_mul)
        q_max = vcmax(q_abs, reduce_axis=-1)
        q_max_mul = vmuls(q_max, 1.0 / 127.0)
        insert_to_gm(gm_scale, q_max_mul, [row_start], slicesize=[1])

        # Quantize and store output
        q_max_brcb = vbrcb(q_max_mul, broadcast_axis=-1, broad_size=N)
        q_brcb_mul = vdiv(q_mul, q_max_brcb)
        # round_fp32_to_int8
        q_tile_int32 = vconv(q_brcb_mul, "INT32")
        q_tile_half = vconv(q_tile_int32, "FP16")
        q_tile_int8 = vconv(q_tile_half, "INT8")
        insert_to_gm(gm_output, q_tile_int8, [row_start, 0], slicesize=[1, N])


def swiglu_quant_op_host_run():
    set_context("310P")

    # Define tensors based on input sizes from numpy implementation
    gm_x = Tensor("GM", "FP16", [512, 512], format="ND", multi_core=False)
    gm_smooth_scales = Tensor("GM", "FP16", [256], format="ND", multi_core=False)
    gm_output = Tensor("GM", "INT8", [512, 256], format="ND", multi_core=False)
    gm_scale = Tensor("GM", "FP32", [512], format="ND", multi_core=False)
    gm_swiglu_out = Tensor("GM", "FP16", [512, 256], format="ND", multi_core=False)

    swiglu_quant_op_impl_npu(gm_x, gm_smooth_scales, gm_output, gm_scale, gm_swiglu_out)


if __name__ == '__main__':
    swiglu_quant_op_host_run()
    compile_kernel("./swiglu_quant_op.cce")
