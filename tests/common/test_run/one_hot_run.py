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

import numpy as np
import itertools
from akg.utils import kernel_exec as utils
from akg.ops.array import one_hot
from tests.common.tensorio import compare_tensor
from tests.common.base import get_rtol_atol


def one_hot_execute(shape, depth, dtype, on_value, off_value, axis, attrs):
    if axis == -1:
        axis = len(shape)

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = one_hot_compile(shape, depth, dtype, on_value, off_value, axis, attrs, kernel_name=kernel_name, tuning=t)
        if t:
            expect, input, on_value_tensor, off_value_tensor, output = gen_data(axis, depth, dtype, shape, on_value, off_value)
            return mod, expect, (input, on_value_tensor, off_value_tensor, output)
        else:
            return mod
    else:
        if not utils.product_is_mini():
            attrs['enable_multicore'] = True
        shape_scalar = (1,)
        mod = one_hot_compile(shape, shape_scalar, depth, dtype, axis, attrs)
        expect, input, on_value_tensor, off_value_tensor, output = gen_data(axis, depth, dtype, shape, on_value, off_value)
        output = utils.mod_launch(mod, (input, on_value_tensor, off_value_tensor, output), expect=expect)
        rtol, atol = get_rtol_atol("one_hot", dtype)
        return input, output, expect, compare_tensor(output, expect, rtol=rtol, atol=atol, equal_nan=True)


def gen_data(axis, depth, dtype, shape, on_value = 1, off_value = 0):
    in_shape_idx = [i for i in range(len(shape))]
    in_shape_idx.append(len(shape))
    in_shape_idx[axis], in_shape_idx[len(shape)] = in_shape_idx[len(shape)], in_shape_idx[axis]
    input = np.random.randint(low = -1, high = depth, size=shape, dtype='int32')
    on_value_tensor = np.array([on_value]).astype(dtype)
    off_value_tensor = np.array([off_value]).astype(dtype)
    if axis < 0:
        axis = axis + len(shape) + 1
    outShape = [i for i in shape]
    outShape.insert(axis, depth)
    expect = np.full(outShape, off_value)
    dims = []
    for dim in shape:
        dims.append(list(range(dim)))
    indexs = [x for x in itertools.product(*tuple(dims))]
    indexs = [list(x) for x in indexs]
    temp = 0
    flatinput = input.flatten()
    for value in flatinput:
        indexs[temp].insert(axis, value)
        temp = temp + 1
    indexs = [tuple(x) for x in indexs]
    for loc in indexs:
        if loc[axis] >= 0:
            expect[loc]  = on_value
    expect = expect.astype(dtype)
    output = np.full(expect.shape, np.nan, dtype)
    return expect, input, on_value_tensor, off_value_tensor, output


def one_hot_compile(shape, shape_scalar, depth, dtype, axis, attrs, kernel_name="one_hot", tuning=False):
    return utils.op_build_test(one_hot.one_hot_v2, [shape, shape_scalar, shape_scalar], ['int32', dtype, dtype], op_attrs=[depth, axis],
                               kernel_name=kernel_name, attrs=attrs, tuning=tuning)
