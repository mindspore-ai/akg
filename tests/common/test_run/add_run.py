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
import akg
import numpy as np
from akg import tvm
from akg.ops.math import Add
from tests.common.tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from tests.common.base import get_rtol_atol
from tests.common.gen_random import random_gaussian
from tests.common.test_utils import compute_blockdim
from akg.utils.result_analysis import target_profiling
from akg.utils.format_transform import to_tvm_nd_array

def add_run(shape1, shape2, dtype, kernel_name="add", scale=1.0, attrs_op={}, polyhedral=True, attrs={}):
    if type(scale) is not float or not int:
        if type(attrs_op) is not bool:
            scale, attrs_op = 1.0, scale
        else:
            scale, attrs_op, polyhedral = 1.0, scale, attrs_op

    op_attrs = [scale]
    if not polyhedral:
        op_attrs = op_attrs + [polyhedral, attrs_op]

    if attrs_op.get("dynamic"):
        attrs_op["enable_double_buffer"] = False
        if shape1 != shape2:
            raise TypeError("Input tensors have different shape. broadcast is't support for dynamic")
        var_shape = []
        for i in range(len(shape1)):
            var_shape.append(tvm.var("I" + str(i)))
        build_shape1 = var_shape
        build_shape2 = var_shape
    else:
        build_shape1 = shape1
        build_shape2 = shape2

    attrs.update(attrs_op)
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(Add, [build_shape1, build_shape2], [dtype, dtype], op_attrs,
                                  kernel_name=kernel_name, attrs=attrs, polyhedral=polyhedral,
                                  tuning=t)
        if t:
            args, expect, input1, input2 = gen_data(shape1, shape2, dtype, scale)
            return mod, expect, args
        else:
            return mod
    else:
        args, expect, input1, input2 = gen_data(shape1, shape2, dtype, scale)
        mod = utils.op_build_test(Add, [build_shape1, build_shape2], [dtype, dtype], op_attrs,
                                  kernel_name=kernel_name, attrs=attrs, polyhedral=polyhedral)
        if attrs.get("dynamic"):
            for i in range(len(shape1)):
                args.append(shape1[i])
            block_dim = compute_blockdim(shape1)
            args.append(block_dim)
        output = utils.mod_launch(mod, args, outputs=(2,), expect=expect)

        if attrs.get("profiling", False):
            target_name = attrs["target"].split()[0]
            data = to_tvm_nd_array(args, akg.tvm.context(target_name, 0))
            target_profiling(mod, *data, target=target_name, repeat_time=attrs["repeat_times"])

        rtol, atol = get_rtol_atol("add", dtype)
        return (input1, input2), output, expect, compare_tensor(output, expect, rtol=rtol, atol=atol, equal_nan=True)


def gen_data(shape1, shape2, dtype, scale):
    input1 = random_gaussian(shape1, miu=1, sigma=0.1)
    input2 = random_gaussian(shape2, miu=1, sigma=0.1)
    if (dtype == "int32"):
        input1 = input1.astype(np.int32)
        input2 = input2.astype(np.int32)
    elif (dtype == "float16"):
        input1 = input1.astype(np.float16)
        input2 = input2.astype(np.float16)
    elif dtype == "float32":
        input1 = input1.astype(np.float32)
        input2 = input2.astype(np.float32)
    expect = np.add(input1, input2 * scale)
    out_shape = expect.shape
    output = np.full(out_shape, np.nan, dtype)
    args = [input1, input2, output]
    return args, expect, input1, input2
