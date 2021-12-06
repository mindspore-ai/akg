# Copyright 2019-2021 Huawei Technologies Co., Ltd
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
from akg.utils import kernel_exec as utils
from akg.ops.math.ascend import EqualCount
from akg import tvm
import math
def compute_blockdim(shape):
    size = 0
    if isinstance(shape, (list, tuple)):
        for i in shape:
            if isinstance(i, int):
                size = size * i
            elif isinstance(i, (list, tuple)):
                for ii in i:
                    if isinstance(ii, int):
                        size = size * ii
    elif isinstance(shape, int):
        size = shape
    else:
        size = 2
    return min(32, math.ceil(size / 8192 + 1))

def equal_count_run(shapes, dtype, kernel_name, attrs):
    # shape check
    if attrs is None:
        attrs = {}
    if attrs.get("dynamic"):
        var_size = tvm.var("I0")
        var_shape = []
        for shape in shapes:
            assert len(shape) == 1
            var_shape.append([var_size])
        build_shape = var_shape
    else:
        build_shape = shapes
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(EqualCount, build_shape, [dtype, dtype], kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            benchMark1, inputs1, output1 = gen_data(dtype, shapes)
            return mod, benchMark1, inputs1 + [output1]
        else:
            return mod
    else:
        mod = utils.op_build_test(EqualCount, build_shape, [dtype, dtype], kernel_name=kernel_name, attrs=attrs)
        benchMark1, inputs1, output1 = gen_data(dtype, shapes)
        if attrs.get("dynamic"):
            args = inputs1.copy()
            args.append(output1)
            for i in range(len(shape) - 1, -1, -1):
                args.append(shape[i])
            block_dim = compute_blockdim(shapes)
            args.append(block_dim)
        else:
            args = inputs1 + [output1]
        output1 = utils.mod_launch(mod, args, outputs=(2,), expect=benchMark1)
        return inputs1, output1, benchMark1, (output1[0] == benchMark1)


def gen_data(dtype, shapes, class_num=10):
    support_list = {"int32": np.int32}
    inputs1 = []
    for i in range(len(shapes)):
        shape = shapes[i]
        input = np.random.randint(low=0, high=class_num, size=shape).astype(support_list[dtype.lower()])
        inputs1.append(input)
    if len(inputs1) != 2:
        raise RuntimeError("inputs num should be 2")
    equal_result = np.equal(inputs1[0], inputs1[1])
    equal_count_num = np.sum(equal_result)
    output1 = np.full((1,), np.nan, dtype)
    return equal_count_num, inputs1, output1
