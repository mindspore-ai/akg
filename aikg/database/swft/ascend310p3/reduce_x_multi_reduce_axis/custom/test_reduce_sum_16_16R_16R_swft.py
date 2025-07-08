import os
from core import *
from api import *
from ai_kernel_generator.core.verifier.swft_kernel_verify import SWFTKernelVerify

"""
场景：最后一个维度开始，向前连续多个维度都是reduce轴，除此之外，不存在其他reduce轴
"""

BLOCK_DIM = 8
first = 16
second = 16
last = 16
@sub_kernel(core_num=BLOCK_DIM)
def reduce_sum_op_impl_npu(gm_input, gm_output):
    num_per_core = first // BLOCK_DIM
    block_idx = get_block_idx()

    for i in range(num_per_core):
        start_idx = num_per_core * block_idx + i
        ub_input = slice_to_ub(gm_input, [start_idx, 0, 0], slicesize=[1, second, last])
        ub_input_reshape = reshape(ub_input, [second * last])
        ub_input_fp32 = vconv(ub_input_reshape, "FP32")
        ub_output = vcadd(ub_input_fp32, reduce_axis=-1)
        ub_output_fp16 = vconv(ub_output, "FP16")
        insert_to_gm(gm_output, ub_output_fp16, [start_idx], slicesize=[1])


def reduce_sum_op_host_run():
    set_context("310P")

    # Define input and output tensors
    gm_input = Tensor("GM", "FP16", [first, second, last], format="ND", multi_core=False)
    gm_output = Tensor("GM", "FP16", [first], format="ND", multi_core=False)

    # Execute the NPU kernel
    op_name = "reduce_sum_op"
    reduce_sum_op_impl_npu(gm_input, gm_output)
    compile_kernel(f"./{op_name}_kernel.cce")
    
    tiling_value = []
    kernel_verify = SWFTKernelVerify(op_name)
    commom_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    op_task_path=os.path.join(commom_path, f"test_{op_name}_task.py")
    status = kernel_verify.run(input_list=[gm_input], attrs_list=[1, 2], tiling_value=tiling_value,
                               block_dim=BLOCK_DIM, workspace_size=0, cce_path=f"./{op_name}_kernel.cce",
                               op_task_path=op_task_path)
    assert status.success, "Test failed"


if __name__ == '__main__':
    reduce_sum_op_host_run()
