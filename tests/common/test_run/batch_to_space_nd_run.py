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
from test_op import batch_to_space_nd


def batch_to_space_nd_benchmark(data, block_shape, crops):
    input_shape = data.shape
    # reshape input
    input_reshape = []
    mul_of_block = 1
    for i in range(len(block_shape)):
        input_reshape.append(block_shape[i])
        mul_of_block = mul_of_block * block_shape[i]

    for i in range(len(input_shape)):
        if i == 0:
            input_reshape.append(input_shape[i] // mul_of_block)
        else:
            input_reshape.append(input_shape[i])

    y1 = np.reshape(data, input_reshape)

    # permatue input reshape
    tran_axis = []
    tran_axis.append(len(block_shape))

    for i in range(len(block_shape)):
        tran_axis.append(len(block_shape) + i + 1)
        tran_axis.append(i)

    for i in range(len(tran_axis), len(input_reshape), 1):
        tran_axis.append(i)

    y2 = np.transpose(y1, tran_axis)

    # reshape permuted
    per_reshape = []
    for i in range(len(input_shape)):
        if i == 0:
            per_reshape.append(input_shape[i] // mul_of_block)
        elif i < len(block_shape) + 1:
            per_reshape.append(input_shape[i] * block_shape[i - 1])
        else:
            per_reshape.append(input_shape[i])

    y3 = np.reshape(y2, per_reshape)

    # crop
    index = []
    for i in range(len(per_reshape)):
        if i == 0:
            tmp = [0, per_reshape[i]]
            index.append(slice(*tmp))
        elif i < len(block_shape) + 1:
            tmp = [0 + crops[i - 1][0], per_reshape[i] - crops[i - 1][1]]
            index.append(slice(*tmp))
        else:
            tmp = [0, per_reshape[i]]
            index.append(slice(*tmp))

    index = tuple(index)
    expect = y3[index]
    return expect


def batch_to_space_nd_run(input_shape, input_dtype, block_shape, crops, kernel_name, attrs=None):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(batch_to_space_nd.batch_to_space_nd, input_shapes=[input_shape],
                                  input_types=[input_dtype], op_attrs=[block_shape, crops], kernel_name=kernel_name,
                                  attrs=attrs, tuning=t)
        if t:
            data, expect, output = gen_data(block_shape, crops, input_dtype, input_shape)
            return mod, expect, (data, output)
        else:
            return mod
    else:
        mod = utils.op_build_test(batch_to_space_nd.batch_to_space_nd, input_shapes=[input_shape],
                                  input_types=[input_dtype], op_attrs=[block_shape, crops], kernel_name=kernel_name,
                                  attrs=attrs)
        data, expect, output = gen_data(block_shape, crops, input_dtype, input_shape)
        output = utils.mod_launch(mod, (data, output), expect=expect)
        return data, output, expect, compare_tensor(output, expect, rtol=0.001)


def gen_data(block_shape, crops, input_dtype, input_shape):
    data = np.random.random(input_shape).astype(input_dtype)
    expect = batch_to_space_nd_benchmark(data, block_shape, crops)
    output = np.full(expect.shape, np.nan, input_dtype)
    return data, expect, output
