import os
from core import *
from api import *
from ai_kernel_generator.core.verifier.swft_kernel_verify import SWFTKernelVerify

"""
场景：仅有最后一根轴是reduce轴, 非reduce轴比较大，reduce轴较小
"""

BLOCK_DIM = 8
reduce_axis = -1
@sub_kernel(core_num=BLOCK_DIM)
def reduce_sum_op_impl_npu(gm_input, gm_output, tiling):
    cols = 128
    tile_rows = 16
    
    ub_tiling = move_to_ub(tiling)
    total_rows = move_to_scalar(ub_tiling[0])

    block_idx = get_block_idx()
    rows_per_core = total_rows // BLOCK_DIM // tile_rows
    
    for iter_idx in dynamic_loop(rows_per_core):
        start_row = block_idx * tile_rows * rows_per_core + iter_idx * tile_rows

        # Load input data from GM to UB
        ub_input = slice_to_ub(gm_input, [start_row, 0], slicesize=[tile_rows, cols])
        ub_input_fp32 = vconv(ub_input, "FP32")

        # Compute row-wise reduce_sum
        ub_output_fp32 = vcadd(ub_input_fp32, reduce_axis=reduce_axis)
        ub_output = vconv(ub_output_fp32, "FP16")

        # Store result back to GM
        insert_to_gm(gm_output, ub_output, [start_row], slicesize=[tile_rows])


def reduce_sum_op_host_run():
    set_context("310P")

    # Define input and output tensors
    gm_input = Tensor("GM", "FP16", [7168, 128], format="ND", multi_core=False)
    gm_output = Tensor("GM", "FP16", [7168], format="ND", multi_core=False)
    tiling = Tensor("GM", "INT32", [128], format="ND", multi_core=False)

    # Execute the NPU kernel
    op_name = "reduce_sum_op"
    reduce_sum_op_impl_npu(gm_input, gm_output, tiling)
    compile_kernel(f"./{op_name}_kernel.cce")
    
    tiling_value = [7168]
    kernel_verify = SWFTKernelVerify(op_name)
    commom_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    op_task_path=os.path.join(commom_path, f"test_{op_name}_task.py")
    status = kernel_verify.run(input_list=[gm_input], attrs_list=[reduce_axis], tiling_value=tiling_value,
                               block_dim=BLOCK_DIM, workspace_size=0, cce_path=f"./{op_name}_kernel.cce",
                               op_task_path=op_task_path)
    assert status.success, "Test failed"


if __name__ == '__main__':
    reduce_sum_op_host_run()
