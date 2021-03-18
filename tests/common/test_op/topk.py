# Copyright 2019 Huawei Technologies Co., Ltd
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

"""operator dsl function:topk"""

import akg
import akg.tvm
from akg.utils import kernel_exec as utils
from akg.tvm.hybrid import script


@script
def compute_topk(output, input, temp):
    res = output_tensor(output.shape, output.dtype)
    # temp_tensor = output_tensor(temp.shape, temp.dtype)
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            res[i, j * 8 + 4] = input[i, j]

    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            # find topk
            res[i, j] = sort_k(res[i, j], res[i, j])

    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            input[i, j] = res[i, j * 8 + 4]

    return res


@script
def compute_get_last(input, output):
    res = output_tensor(output.shape, output.dtype)
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            res[i, j] = input[i, j * 8 + 4]
    return res


def topk(shape, k, dtype, kernel_name, attrs):
    check_list = ["float16", "int32"]
    if not (dtype.lower() in check_list):
        raise RuntimeError("tile_cce only support %s while dtype is %s" %
                           (",".join(check_list), dtype))
    if k > shape[-1]:
        raise RuntimeError("k should not be greater than shape[-1]")

    shape = (16, 16)
    out_shape = (16, 16)
    temp_shape = (16, 16 * 18)
    inputs = akg.tvm.placeholder(shape, name="input", dtype="float16")
    output = akg.tvm.placeholder(out_shape, name="output", dtype="float16")
    temp = akg.tvm.placeholder(temp_shape, name="temp", dtype="float16")

    values = compute_topk(output, inputs, temp)
    values1 = compute_get_last(values, temp)

    s = akg.tvm.create_schedule([values1.op])
    with akg.build_config(add_lower_pass=utils.debug_mode(0), dump_pass_ir=True):
        mod = akg.build(s, [inputs, values1],
                       "cce",
                       name=kernel_name,
                       attrs=attrs,
                       polyhedral=True)
        return mod
