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

import math
import numpy as np
from tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from test_op import argmin
from akg.ops.math import argmax
from base import get_rtol_atol
from akg.utils.dsl_create import get_reduce_out_shape
from gen_random import random_gaussian
from akg import tvm
from test_utils import compute_blockdim


def common_run(shape, dtype, axis, attrs, method):
    if attrs is None:
        attrs = {}
    attrs["enable_algebra_simplify"] = True
    if attrs.get("dynamic"):
        build_shape = []
        for i in range(len(shape)):
            build_shape.append(tvm.var("I" + str(i)))
    else:
        build_shape = shape
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        if method is "min":
            mod = utils.op_build_test(argmin.argmin, [build_shape], [dtype], op_attrs=[axis], kernel_name=kernel_name,
                                      attrs=attrs, tuning=t)
        elif method is "max":
            mod = utils.op_build_test(argmax.argmax, [build_shape], [dtype], op_attrs=[axis], kernel_name=kernel_name,
                                      attrs=attrs, tuning=t)
        else:
            raise RuntimeError("not support " + method)
        if t:
            args, exp_output, input = gen_data(axis, dtype, method, shape)
            return mod, exp_output, args
        else:
            return mod
    else:
        if method is "min":
            mod = utils.op_build_test(argmin.argmin, [build_shape], [dtype], op_attrs=[axis], kernel_name="argmin",
                                      attrs=attrs)
        elif method is "max":
            mod = utils.op_build_test(argmax.argmax, [build_shape], [dtype], op_attrs=[axis], kernel_name="argmax",
                                      attrs=attrs)
        else:
            raise RuntimeError("not support " + method)
        args, exp_output, input = gen_data(axis, dtype, method, shape)
        if attrs.get("dynamic"):
            for i in range(len(shape)):
                args.append(shape[i])
            block_dim = compute_blockdim(shape)
            args.append(block_dim)
        res = utils.mod_launch(mod, args, outputs=(1,), expect=exp_output)
        acu_output = res.astype("int32")
        rtol, atol = get_rtol_atol("argmax_min_common", dtype)
        return input, acu_output, exp_output, compare_tensor(acu_output, exp_output, rtol=rtol, atol=atol, equal_nan=True)


def gen_data(axis, dtype, method, shape):
    support_list = {"float16": np.float16, "float32": np.float32, "int32": np.int32, "int8": np.int8}
    input = random_gaussian(shape, miu=1, sigma=100).astype(support_list[dtype])
    if dtype == "float32":
        input = np.around(input, 0)
    if method is "min":
        exp_output = np.argmin(input, axis=axis)
    elif method is "max":
        exp_output = np.argmax(input, axis=axis)
    else:
        raise RuntimeError("not support " + method)
    out_shape = get_reduce_out_shape(shape, axis=axis)
    output = np.full(out_shape, np.nan, np.int32)
    args = [input, output]
    return args, exp_output, input
