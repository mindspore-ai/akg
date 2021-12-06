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

from akg.utils import kernel_exec as utils
import numpy as np
from akg.topi.util import get_const_tuple
from akg.ops.nn.ascend import Relu
from tests.common.tensorio import compare_tensor
from tests.common.base import get_rtol_atol
from akg import tvm
import math

def compute_blockdim(shape):
    size = 0
    if isinstance(shape, (list, tuple)):
        for i in shape:
            size = size * i
    elif isinstance(shape, int):
        size = shape
    else:
        size = 2
    return min(32, math.ceil(size / 8192 + 1))

def relu_run(shape, dtype, rtol, attrs):
    if attrs is not None and attrs.get("dynamic"):
        build_shape = []
        attrs['enable_post_poly_loop_partition'] = False
        for i in range(len(shape)):
            build_shape.append(tvm.var("I" + str(i)))
    else:
        build_shape = shape
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(Relu, [build_shape], [dtype], kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            input_np, expect = gen_data(dtype, shape)
            return mod, (input_np, expect)
        else:
            return mod
    else:
        mod = utils.op_build_test(Relu, [build_shape], [dtype], kernel_name='relu', attrs=attrs)
        input_np, expect = gen_data(dtype, shape)
        output = np.full(expect.shape, np.nan, dtype=expect.dtype)
        args = [input_np, output]
        if attrs is not None and attrs.get("dynamic"):
            for i in range(len(shape)):
                args.append(shape[i])
            block_dim = compute_blockdim(shape)
            args.append(block_dim)
        output = utils.mod_launch(mod, args, outputs=(1,), expect=expect)
        rtol, atol = get_rtol_atol("relu", dtype)
        return input_np, output, expect, compare_tensor(output, expect, rtol=rtol, atol=atol)


def gen_data(dtype, shape):
    input_np = np.random.uniform(low=-1.0, high=1.0, size=get_const_tuple(shape)).astype(dtype)
    output_np = input_np * (input_np > 0)
    return input_np, output_np
