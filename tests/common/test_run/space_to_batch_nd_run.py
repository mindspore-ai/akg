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

from akg.utils import kernel_exec as utils
from tensorio import compare_tensor
import numpy as np
from test_op import space_to_batch_nd


def space_to_batch_nd_run(input_shape, input_dtype, block, pad, kernel_name, attrs=None):
    op_attrs = [block, pad]
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(space_to_batch_nd.space_to_batch_nd, [input_shape], [input_dtype], op_attrs,
                                  kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            data, expect, output = gen_data(block, input_dtype, input_shape, pad)
            return mod, expect, (data, output)
        else:
            return mod
    else:
        data, expect, output = gen_data(block, input_dtype, input_shape, pad)
        mod = utils.op_build_test(space_to_batch_nd.space_to_batch_nd, [input_shape], [input_dtype], op_attrs,
                                  kernel_name=kernel_name, attrs=attrs)
        output = utils.mod_launch(mod, (data, output), expect=expect)
        return data, output, expect, compare_tensor(output, expect, atol=0.01)


def gen_data(block, input_dtype, input_shape, pad):
    support_list = {"float16": np.float16, "float32": np.float32, "int32": np.int32, "int8": np.int8, "uint8": np.uint8}
    if input_dtype == "uint8":
        data = np.random.randint(0, 5, input_shape).astype(support_list[input_dtype])
    elif input_dtype == "int8" or input_dtype == "int32":
        data = np.random.randint(-5, 5, input_shape).astype(support_list[input_dtype])
    else:
        data = np.random.random(input_shape).astype(support_list[input_dtype])
    lpad = list(pad)
    lpad.insert(0, (0, 0))
    lpad.append((0, 0))
    data_pad = np.pad(data, lpad, 'constant')
    data_pad_shape = data_pad.shape
    pad_rshape = []
    data_pad_shape_len = len(data_pad_shape)
    for i in range(data_pad_shape_len):
        if i == 0 or i == data_pad_shape_len - 1:
            pad_rshape.append(data_pad_shape[i])
        else:
            pad_rshape.append(data_pad_shape[i] // block[i - 1])
            pad_rshape.append(block[i - 1])
    y = np.reshape(data_pad, pad_rshape)
    tran_shape = []
    for i in range(2, len(pad_rshape), 2):
        tran_shape.append(i)
    tran_shape.append(0)
    for i in range(1, len(pad_rshape), 2):
        tran_shape.append(i)
    z = np.transpose(y, tran_shape)
    per_reshape = []
    x_shape_len = len(input_shape)
    for i in range(x_shape_len):
        if i == 0:
            mul_of_block = 1
            for j in range(len(block)):
                mul_of_block = mul_of_block * block[j]
            per_reshape.append(input_shape[i] * mul_of_block)
        elif i == x_shape_len - 1:
            per_reshape.append(input_shape[i])
        else:
            per_reshape.append(data_pad_shape[i] // block[i - 1])
    expect = np.reshape(z, per_reshape)
    output = np.full(expect.shape, np.nan, input_dtype)
    return data, expect, output
