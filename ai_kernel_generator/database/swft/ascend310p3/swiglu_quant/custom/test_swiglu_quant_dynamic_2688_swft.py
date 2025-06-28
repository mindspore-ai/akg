import os
from core import *
from api import *
from ai_kernel_generator.core.verifier.swft_kernel_verify import SWFTKernelVerify

BATCH_SIZE = 24
DIM = 2688
HALF_DIM = 1344
BLOCK_DIM = 8
@sub_kernel(core_num=BLOCK_DIM)
def swiglu_quant_op_impl_npu(gm_input, gm_scale, gm_offset, gm_output, tiling):
    ub_tiling = move_to_ub(tiling)
    batch_size_tile = move_to_scalar(ub_tiling[0])
    core_idx = get_block_idx()
    start_idx = core_idx * batch_size_tile

    for i in dynamic_loop(batch_size_tile):
        # Load input sample
        sample_idx = start_idx + i
        ub_input = slice_to_ub(gm_input, [sample_idx, 0], [1, DIM])
        
        # Split input into two parts
        ub_part0 = slice_to_ub(ub_input, [0, 0], [1, HALF_DIM])
        ub_part1 = slice_to_ub(ub_input, [0, HALF_DIM], [1, HALF_DIM])
        
        # Convert part0 to FP32
        ub_part0_fp32 = vconv(ub_part0, "FP32")
        
        # Compute Swish: x/(1+exp(-x))
        ub_neg_part0 = vmuls(ub_part0_fp32, Scalar("FP32", -1.0))
        ub_exp_neg = vexp(ub_neg_part0)
        ub_exp_add = vadds(ub_exp_neg, Scalar("FP32", 1.0))
        ub_swiglu = vdiv(ub_part0_fp32, ub_exp_add)
        
        # Convert part1 to FP32 and multiply with Swish
        ub_part1_fp32 = vconv(ub_part1, "FP32")
        ub_swiglu = vmul(ub_swiglu, ub_part1_fp32)

        # Load scale to UB
        ub_scale = slice_to_ub(gm_scale, [0], [HALF_DIM])
                
        # Apply scale
        ub_scaled = vmul(ub_swiglu, ub_scale)

        # Load offset to UB
        ub_offset = slice_to_ub(gm_offset, [0], [HALF_DIM])
        
        # Convert offset to FP32
        ub_offset_fp16 = vconv(ub_offset, "FP16")
        ub_offset_fp32 = vconv(ub_offset_fp16, "FP32")

        # Add offset
        ub_result = vadd(ub_scaled, ub_offset_fp32)
        
        # Convert to INT8
        ub_output_fp16 = vconv(ub_result, "FP16")
        ub_output_int8 = vconv(ub_output_fp16, "INT8")
        
        # Store result
        insert_to_gm(gm_output, ub_output_int8, [sample_idx, 0], [1, HALF_DIM])


def swiglu_quant_op_host_run():
    set_context("310P")

    gm_input = Tensor("GM", "FP16", [BATCH_SIZE, DIM], "ND", False)
    gm_scale = Tensor("GM", "FP32", [HALF_DIM], "ND", False)
    gm_offset = Tensor("GM", "INT8", [HALF_DIM], "ND", False)
    gm_output = Tensor("GM", "INT8", [BATCH_SIZE, HALF_DIM], "ND", False)
    tiling = Tensor("GM", "INT32", [128], format="ND", multi_core=False)

    op_name = "swiglu_quant_op"
    swiglu_quant_op_impl_npu(gm_input, gm_scale, gm_offset, gm_output, tiling)
    compile_kernel(f"./{op_name}_kernel.cce")
    
    tiling_value = [3]
    kernel_verify = SWFTKernelVerify(op_name)
    commom_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    op_task_path=os.path.join(commom_path, f"test_{op_name}_task.py")
    status = kernel_verify.run(input_list=[gm_input, gm_scale, gm_offset], tiling_value=tiling_value,
                      block_dim=BLOCK_DIM, workspace_size=0,
                      cce_path=f"./{op_name}_kernel.cce",
                      op_task_path=op_task_path)
    assert status.success, "Test failed"

if __name__ == '__main__':
    swiglu_quant_op_host_run()