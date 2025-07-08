from core import *
from api import *
import os
from ai_kernel_generator.core.verifier.swft_kernel_verify import SWFTKernelVerify

"""
场景：模拟reduce超大，一个UB存不下的情况，需要对reduce轴进行切分
"""

BLOCK_DIM = 8
reduce_axis = -1
@sub_kernel(core_num=BLOCK_DIM)
def reduce_sum_op_impl_npu(gm_input, gm_output, tiling):
    rows = 64
    static_cols = 128
    
    # ub_tiling = move_to_ub(tiling)
    # tile_cols = move_to_scalar(ub_tiling[0])

    block_idx = get_block_idx()
    rows_per_core = rows // BLOCK_DIM
    start_row = block_idx * rows_per_core
    
    ub_output_list = []
    # for iter_idx in dynamic_loop(tile_cols):
    ub_tmp = vector_dup(Scalar("FP32", 0.0), [rows_per_core, static_cols], False)
    for iter_idx in range(8):
        start_cols = iter_idx * static_cols
        # Load input data from GM to UB
        ub_input = slice_to_ub(gm_input, [start_row, start_cols], slicesize=[rows_per_core, static_cols])
        ub_input_fp32 = vconv(ub_input, "FP32")

        # Compute row-wise reduce_sum
        ub_tmp = vadd(ub_tmp, ub_input_fp32)

    ub_output1_fp32 = vcadd(ub_tmp, reduce_axis=reduce_axis)    
    ub_output = vconv(ub_output1_fp32, "FP16")

    # Store result back to GM
    insert_to_gm(gm_output, ub_output, [start_row], slicesize=[rows_per_core])


def reduce_sum_op_host_run():
    set_context("310P")

    # Define input and output tensors
    gm_input = Tensor("GM", "FP16", [64, 1024], format="ND", multi_core=False)
    gm_output = Tensor("GM", "FP16", [64], format="ND", multi_core=False)
    tiling = Tensor("GM", "INT32", [128], format="ND", multi_core=False)

    # Execute the NPU kernel
    op_name = "reduce_sum_op"
    reduce_sum_op_impl_npu(gm_input, gm_output, tiling)
    compile_kernel(f"./{op_name}_kernel.cce")
    
    tiling_value = [128]
    kernel_verify = SWFTKernelVerify(op_name)
    commom_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    op_task_path=os.path.join(commom_path, f"test_{op_name}_task.py")
    status = kernel_verify.run(input_list=[gm_input], attrs_list=[reduce_axis], tiling_value=tiling_value,
                               block_dim=BLOCK_DIM, workspace_size=0, cce_path=f"./{op_name}_kernel.cce",
                               op_task_path=op_task_path)
    assert status.success, "Test failed"

if __name__ == '__main__':
    reduce_sum_op_host_run()
