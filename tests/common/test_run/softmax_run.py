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

"""
sqrt run define
"""

import math
import numpy as np
import akg
from akg.ops.nn import softmax
from akg.utils import kernel_exec as utils
from akg.utils.format_transform import get_bytes
from tensorio import compare_tensor
from base import get_rtol_atol
from gen_random import random_gaussian

def compute_blockdim(shape, axis, dtype):
    # strategy: all the shape before reduce axis can be used for multicore
    blockdim_limit = 2 if utils.product_is_mini() else 32
    blockdim = 1
    if isinstance(shape, int):
        shape = [shape]
    if axis < 0:
        axis += len(shape)
    if isinstance(shape, (list, tuple)):
        for i, sh in enumerate(shape):
            if not isinstance(sh, int):
                raise TypeError("Shape to compute blockdim must be a list/tuple of integer")
            if i == axis:
                if sh < 32 / get_bytes(dtype):
                    # when reduce axis is too small, multicore may not always increase performace
                    blockdim = 1
                break
            blockdim = blockdim * sh
    else:
        raise TypeError("Shape to compute blockdim must be a list/tuple of integer")
    return min(blockdim_limit, blockdim)


def softmax_execute(shape, dtype, axis, kernel_name, attrs):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = softmax_compile(shape, dtype, axis, kernel_name, attrs, tuning=t)
        if t:
            expect, inputs, output = gen_data(axis, dtype, shape)
            return mod, expect, (inputs, output)
        else:
            return mod
    else:
        mod = softmax_compile(shape, dtype, axis, kernel_name, attrs)
        expect, inputs, output = gen_data(axis, dtype, shape)
        args = [inputs, output]
        if attrs.get("dynamic"):
            for i in range(len(shape)):
                args.append(shape[i])
            blockdim = compute_blockdim(shape, axis, dtype)
            args.append(blockdim)
        acuOutput = utils.mod_launch(mod, args, outputs=(1,), expect=expect)
        rtol, atol = get_rtol_atol("softmax", dtype)
        testCaseRes = compare_tensor(acuOutput, expect, rtol=rtol, atol=atol, equal_nan=True)
        return inputs, acuOutput, expect, testCaseRes


def gen_data(axis, dtype, shape):
    if isinstance(axis, (list, tuple)):
        axis = axis[0]
    inputs = random_gaussian(shape, miu=1, sigma=0.1).astype(dtype)
    inputsSub = inputs - np.max(inputs, axis=axis, keepdims=True)
    inputsExp = np.exp(inputsSub)
    expect = inputsExp / np.sum(inputsExp, axis=axis, keepdims=True)
    outShape = expect.shape
    output = np.full(outShape, np.nan, dtype)
    return expect, inputs, output


def softmax_compile(shape, dtype, axis, kernel_name, attrs, tuning=False):
    if attrs is not None and attrs.get("dynamic"):
        var_shape = []
        for i in range(len(shape)):
            var_shape.append(akg.tvm.var("I" + str(i)))
        build_shape = var_shape
    else:
        build_shape = shape
    return utils.op_build_test(softmax.softmax, [build_shape], [dtype], op_attrs=[axis], kernel_name=kernel_name, attrs=attrs, tuning=tuning)
