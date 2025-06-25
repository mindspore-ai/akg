# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import os
import sys
from swft.core import *
from swft.api import *

OP_NAME = 'moe_token_unpermute_op'
os.system(f"mkdir -p temp/{OP_NAME}")
os.system(f"mkdir -p temp/{OP_NAME}/input")
os.system(f"mkdir -p temp/{OP_NAME}/output")


token_num = 11
BLOCK_DIM = token_num
top_k = 8
hidden = 7168


def moe_token_unpermute_op_impl(permute_token, sorted_idx, probs):
    token_num = probs.shape[0]
    top_k = probs.shape[1]
    hidden = permute_token.shape[1]
    out = np.zeros((token_num, hidden), dtype=np.float16)

    for i in range(token_num):
        for k in range(top_k):
            dst_row = permute_token[sorted_idx[i * top_k + k], :]
            out[i, :] += probs[i, k] * dst_row
    return out


def gen_data():
    # 生成输入数据
    permute_token = np.random.normal(0.0, 0.5, size=(
        token_num * top_k, hidden)).astype(np.float16)
    sorted_idx = np.random.permutation(
        np.arange(token_num * top_k, dtype=np.int32))
    probs = np.random.normal(0.0, 0.5, size=(
        token_num, top_k)).astype(np.float16)

    expected = moe_token_unpermute_op_impl(permute_token, sorted_idx, probs)

    tiling = np.zeros([128], dtype=np.int32)
    tiling[0] = token_num
    tiling[1] = top_k
    tiling.tofile(f"./temp/{OP_NAME}/input/tiling.bin")

    permute_token.tofile(f"./temp/{OP_NAME}/input/gm_permute_token.bin")
    sorted_idx.tofile(f"./temp/{OP_NAME}/input/gm_sorted_idx.bin")
    probs.tofile(f"./temp/{OP_NAME}/input/gm_probs.bin")
    expected.tofile(f"./temp/{OP_NAME}/output/gm_output_golden.bin")


@sub_kernel(core_num=BLOCK_DIM)
def moe_token_unpermute_op_impl_npu(gm_permute_token, gm_sorted_idx, gm_probs, gm_output, gm_tiling):
    block_idx = get_block_idx()

    # dynamic tiling from global memory
    ub_tiling = move_to_ub(gm_tiling)
    top_k = move_to_scalar(ub_tiling[1])
    curr_token = block_idx

    tmp_s = Scalar("FP16", 0.0)
    local_out = vector_dup(tmp_s, [1, hidden], False)

    ub_idx = slice_to_ub(gm_sorted_idx, [curr_token * top_k], slicesize=[64])

    for k in dynamic_loop(top_k):
        idx = move_to_scalar(ub_idx[k])

        # Load permute_token row
        dst_row = slice_to_ub(
            gm_permute_token, [idx, 0], slicesize=[1, hidden])

        # Load prob
        prob_ub = slice_to_ub(gm_probs, [curr_token * top_k], slicesize=[64])
        prob = move_to_scalar(prob_ub[k])

        # # Compute weighted row and accumulate
        weighted_row = vmuls(dst_row, prob)
        local_out = vadd(local_out, weighted_row)

    # Write back to GM
    insert_to_gm(gm_output, local_out, [curr_token, 0], slicesize=[1, hidden])


if __name__ == "__main__":
    gen_data()
    set_context("310P")

    gm_permute_token = Tensor(
        "GM", "FP16", [token_num * top_k, hidden], format="ND", multi_core=False)
    gm_sorted_idx = Tensor("GM", "INT32", [
                           token_num * top_k], format="ND", multi_core=False)  # [topk, token_num]
    gm_probs = Tensor(
        "GM", "FP16", [token_num * top_k], format="ND", multi_core=False)
    gm_output = Tensor(
        "GM", "FP16", [token_num, hidden], format="ND", multi_core=False)
    tiling = Tensor("GM", "INT32", [128], format="ND", multi_core=False)

    compile_func(moe_token_unpermute_op_impl_npu, globals())(
        gm_permute_token, gm_sorted_idx, gm_probs, gm_output, tiling)

    compile_kernel(f"./temp/{OP_NAME}/{OP_NAME}.cce", OP_NAME)
    exec_kernel(OP_NAME, locals(), prefix_path="temp", inputs=[
                'gm_permute_token', 'gm_sorted_idx', 'gm_probs', 'tiling'], outputs=['gm_output'])
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return_code = os.system(
        f'python3 {script_dir}/../verify_result.py ./temp/{OP_NAME}/output/gm_output_actual.bin ./temp/{OP_NAME}/output/gm_output_golden.bin float16 4e-2 1e-2 4e-3')
    sys.exit(return_code >> 8)
