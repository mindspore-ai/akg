import os
from core import *
from api import *
from ai_kernel_generator.core.verifier.swft_kernel_verify import SWFTKernelVerify

CORE_NUM = 8


@sub_kernel(core_num=CORE_NUM)
def swiglu_op_impl_npu(gm_input, gm_output, tiling):
    """
    完成swiglu函数值计算，输入为shape较大、数据类型为fp16的Tensor。输出的Tensor数据类型为fp16。共有8个核。
    记输入为Input，令[x, y] = Input，x和y的shape相同，则输出Output = x * sigmoid(x) * y。其中，sigmoid(x) = x / (1 + exp(x))。
    由于过程中有exp计算，所以中间计算过程数据类型要转为fp32保证精度。
    由于UB内存有限，所以需要对输入的Tensor做切分。并且，由于输入的shape较大，需要采用动态shape的方法进行计算加速，避免出现CCE文件过长生成失败的情况。
    """

    TILE_DIM0 = 2
    TILE_DIM1 = 2048

    ub_tiling = move_to_ub(tiling)
    M = move_to_scalar(ub_tiling[0])
    
    N = 4096
    N_split = N // 2
    tile_dim1_split = TILE_DIM1 // 2
    per_core_dim0 = M // CORE_NUM
    per_core_tile_dim0 = per_core_dim0 // TILE_DIM0
    per_core_tile_dim1 = N // TILE_DIM1

    block_idx = get_block_idx()

    for iter_tile_idx_dim0 in dynamic_loop(per_core_tile_dim0):
        dim0_start = block_idx * per_core_dim0 + iter_tile_idx_dim0 * TILE_DIM0

        for iter_tile_idx_dim1 in range(per_core_tile_dim1):
            dim1_start_0 = iter_tile_idx_dim1 * tile_dim1_split
            dim1_start_1 = N_split + iter_tile_idx_dim1 * tile_dim1_split

            # Load input data from GM to UB
            ub_input0_half = slice_to_ub(gm_input, [dim0_start, dim1_start_0], slicesize=[TILE_DIM0, tile_dim1_split])
            ub_input1_half = slice_to_ub(gm_input, [dim0_start, dim1_start_1], slicesize=[TILE_DIM0, tile_dim1_split])

            # Convert to FP32
            ub_input0_float = vconv(ub_input0_half, "FP32")
            ub_input1_float = vconv(ub_input1_half, "FP32")
            
            # Compute SwiGLU: x * sigmoid(x) * y
            # sigmoid(x) = x / (1 + exp(x))
            neg_tile = vmuls(ub_input0_float, 1.0)
            exp_tile = vexp(neg_tile)
            exp_tile = vadds(exp_tile, 1.0)
            sigmoid_tile = vdiv(ub_input0_float, exp_tile)
            result_tile = vmul(sigmoid_tile, ub_input1_float)

            # Convert result back to FP16 and store to GM
            ub_output_half = vconv(result_tile, "FP16")
            insert_to_gm(gm_output, ub_output_half, [dim0_start, dim1_start_0], slicesize=[TILE_DIM0, tile_dim1_split])


def swiglu_op_host_run():
    set_context("310P")

    # Define input tensors
    M = 4096

    gm_input = Tensor("GM", "FP16", [4096, 4096], format="ND", multi_core=False)
    gm_output = Tensor("GM", "FP16", [4096, 2048], format="ND", multi_core=False)
    tiling = Tensor("GM", "INT32", [128], format="ND", multi_core=False)

    op_name = "swiglu_op"
    swiglu_op_impl_npu(gm_input, gm_output, tiling)
    compile_kernel(f"./{op_name}_kernel.cce")
    
    tiling_value = [M]
    kernel_verify = SWFTKernelVerify(op_name)
    commom_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    op_task_path=os.path.join(commom_path, f"test_{op_name}_task.py")
    status = kernel_verify.run(input_list=[gm_input], tiling_value=tiling_value, block_dim=CORE_NUM, workspace_size=0,
                      cce_path=f"./{op_name}_kernel.cce",
                      op_task_path=op_task_path)
    assert status.success, "Test failed"

if __name__ == '__main__':
    swiglu_op_host_run()
