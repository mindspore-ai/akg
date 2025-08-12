from swft.core import *
from swft.api import *
import os

OP_NAME = "topk"

topk_len = 8
inf = 65500
length = 32


@sub_kernel(core_num=1)
def topk_kernel(x, indices, x_out, indices_out):
    ub_x = move_to_ub(x)
    ub_indices = move_to_ub(indices)
    inf_s = Scalar(ub_x.dtype, -inf)
    neg_one = Scalar("INT32", -1)
    for i in range(topk_len):
        res = inf_s.copy()
        index = neg_one.copy()
        for j in range(i, length):
            x = move_to_scalar(ub_x[j])
            if x > res:
                res.load(x)
                index.load(Scalar("INT32", j))
        tmp = move_to_scalar(ub_x[i])
        ub_x = move_scalar_to_ub(res, ub_x, i)
        ub_x = move_scalar_to_ub(tmp, ub_x, index)
        tmp_i = move_to_scalar(ub_indices[i])
        ub_indices = move_scalar_to_ub(index, ub_indices, i)
        ub_indices = move_scalar_to_ub(tmp_i, ub_indices, index)
    x_out.load(ub_x)
    ub_indices_s = slice_to_ub(ub_indices, [0], [topk_len])
    indices_out.load(ub_indices_s)


def topk_swft_numpy(device_id=0):
    set_context("310P")
    input0 = Tensor("GM", "FP16", [length], "ND", False)
    output0 = Tensor("GM", "FP16", [length], "ND", False)
    input1 = Tensor("GM", "INT32", [length], "ND", False)
    output1 = Tensor("GM", "INT32", [topk_len], "ND", False)
    topk_kernel(input0, input1, output0, output1)

    # 使用动态路径
    current_dir = os.path.dirname(__file__)
    cce_path = os.path.join(current_dir, f"{OP_NAME}", f"{OP_NAME}.cce")
    compile_kernel(cce_path, OP_NAME)
    exec_kernel(OP_NAME, locals(), inputs=['input0', 'input1'], outputs=['output0', 'output1'], device_id=device_id)
