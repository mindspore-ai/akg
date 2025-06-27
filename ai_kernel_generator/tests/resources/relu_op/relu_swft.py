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
from swft.core import *
from swft.api import *
import os

OP_NAME = "relu"

d0 = 16
CORE_NUM = d0
d1 = 16384

@sub_kernel(core_num=CORE_NUM)
def relu_op(gm_input, gm_output):
    block_idx = get_block_idx()
    ub_input = slice_to_ub(gm_input, [block_idx, 0], slicesize=[1, d1])
    ub_input_max = vrelu(ub_input)
    insert_to_gm(gm_output, ub_input_max, [block_idx, 0], slicesize=[1, d1])


def relu_swft(device_id=0):
    set_context("310P")
    input0 = Tensor("GM", "FP32", [d0, d1], format="ND", multi_core=False)
    output0 = Tensor("GM", "FP32", [d0, d1], format="ND", multi_core=False) 
    compile_func(relu_op, globals())(input0, output0)
    
    # 使用动态路径
    current_dir = os.path.dirname(__file__)
    cce_path = os.path.join(current_dir, f"{OP_NAME}", f"{OP_NAME}.cce")
    compile_kernel(cce_path, OP_NAME)

    exec_kernel(OP_NAME, locals(), inputs=['input0'], outputs=['output0'], device_id=device_id)


def relu_swft_mindspore(device_id=0):
    # 封装mindspore接口
    relu_swft(device_id)


def relu_swft_torch(device_id=0):
    # 封装torch接口
    relu_swft(device_id)


def relu_swft_numpy(device_id=0):
    # 封装numpy接口
    relu_swft(device_id)
