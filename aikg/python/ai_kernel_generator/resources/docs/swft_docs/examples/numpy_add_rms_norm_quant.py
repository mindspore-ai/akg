from swft.core import *
from swft.api import *
import os

OP_NAME = "add_rms_norm_quant_op"

BLOCK_DIM = 2


@sub_kernel(core_num=BLOCK_DIM)
def add_rmsnorm_quant_op_kernel(gm_input_a, gm_input_b, gm_gamma, gm_scale, gm_offset, gm_output1, gm_output2):
    # 硬编码参数
    BATCH_SIZE = 2
    DIM = 5120
    SAMPLES_PER_CORE = BATCH_SIZE // BLOCK_DIM

    # 获取当前核ID
    core_idx = get_block_idx()
    start_batch = core_idx * SAMPLES_PER_CORE

    # 加载常驻数据
    ub_gamma = slice_to_ub(gm_gamma, [0], [DIM])
    ub_scale = slice_to_ub(gm_scale, [0], [DIM])
    ub_offset = slice_to_ub(gm_offset, [0], [DIM])

    # 转换offset到float32
    ub_offset_fp32 = vconv(ub_offset, "FP32")

    for i in range(SAMPLES_PER_CORE):
        current_batch = start_batch + i

        # 1. 加载输入数据
        ub_input_a = slice_to_ub(gm_input_a, [current_batch, 0, 0], [1, 1, DIM])
        ub_input_b = slice_to_ub(gm_input_b, [current_batch, 0, 0], [1, 1, DIM])

        # 2. 执行Add操作
        ub_add_result = vadd(ub_input_a, ub_input_b)

        # 3. 计算平方
        ub_squared = vmul(ub_add_result, ub_add_result)

        # 4. 计算均方根(RMS)
        ub_rms_sum = vcadd(ub_squared, reduce_axis=-1)
        scalar_rms_mean = vdivs(ub_rms_sum, Scalar("FP16", DIM))
        scalar_rms_mean_eps = vadds(scalar_rms_mean, Scalar("FP16", 1e-5))
        scalar_rms = vsqrt(scalar_rms_mean_eps)

        # 5. 归一化
        scalar_rms_rec = move_to_scalar(vrec(scalar_rms))
        ub_normalized = vmuls(ub_add_result, scalar_rms_rec)

        # 6. 应用gamma缩放
        ub_scaled_output = vmul(ub_normalized, ub_gamma)

        # 7. 量化处理
        ub_scaled_fp32 = vconv(ub_scaled_output, "FP32")
        ub_out_scale = vmul(ub_scaled_fp32, ub_scale)
        ub_out_fp32 = vadd(ub_out_scale, ub_offset_fp32)
        ub_out_fp16 = vconv(ub_out_fp32, "FP16")
        ub_output2 = vconv(ub_out_fp16, "INT8")

        # 8. 写回结果
        insert_to_gm(gm_output1, ub_scaled_output, [current_batch, 0, 0], [1, 1, DIM])
        insert_to_gm(gm_output2, ub_output2, [current_batch, 0, 0], [1, 1, DIM])


def add_rms_norm_quant_op_swft_numpy(device_id=0):
    set_context("310P")

    input0 = Tensor("GM", "FP16", [2, 1, 5120], "ND", False)
    input1 = Tensor("GM", "FP16", [2, 1, 5120], "ND", False)
    input2 = Tensor("GM", "FP16", [5120], "ND", False)
    input3 = Tensor("GM", "FP32", [5120], "ND", False)
    input4 = Tensor("GM", "INT32", [5120], "ND", False)
    output0 = Tensor("GM", "FP16", [2, 1, 5120], "ND", False)
    output1 = Tensor("GM", "INT8", [2, 1, 5120], "ND", False)
    add_rmsnorm_quant_op_kernel(input0, input1, input2, input3, input4, output0, output1)

    # 使用动态路径
    current_dir = os.path.dirname(__file__)
    cce_path = os.path.join(current_dir, f"{OP_NAME}", f"{OP_NAME}.cce")
    compile_kernel(cce_path, OP_NAME)
    exec_kernel(OP_NAME, locals(), inputs=['input0', 'input1', 'input2', 'input3',
                'input4'], outputs=['output0', 'output1'], device_id=device_id)
