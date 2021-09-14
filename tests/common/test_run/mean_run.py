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
from akg import tvm
from tests.common.tensorio import compare_tensor
from tests.common.base import get_rtol_atol
from tests.common.gen_random import random_gaussian
from akg.ops.math import mean
from akg.utils import kernel_exec as utils
from akg.utils.result_analysis import akg_fp16_mean
from akg.utils.dsl_create import get_reduce_out_shape
from akg.utils.format_transform import get_bytes
from tests.common.test_utils import process_dynamic_shape

def compute_blockdim(shape, axis, dtype):
    # strategy: all the shape except reduce axis can be used for multicore
    blockdim_limit = 2 if utils.product_is_mini() else 32
    blockdim = 1
    if isinstance(shape, int):
        shape = [shape]
    if not isinstance(axis, list):
        axis = list(axis)
    for a in axis:
        if a < 0:
            a += len(shape)
    axis = sorted(axis)
    red_sh = 1
    if isinstance(shape, (list, tuple)):
        for i, sh in enumerate(shape):
            if not isinstance(sh, int):
                raise TypeError("Shape to compute blockdim must be a list/tuple of integer")
            if i in axis:
                red_sh *= sh
            else:
                blockdim = blockdim * sh
    else:
        raise TypeError("Shape to compute blockdim must be a list/tuple of integer")
    if red_sh < 32 / get_bytes(dtype):
        # when reduce axis is too small, multicore may not always increase performace
        blockdim = 1

    return min(blockdim_limit, blockdim)

def mean_execute(shape, dtype, axis, keepdims, kernel_name, attrs):
    if attrs is None:
        attrs = {}
    attrs["pragma_disable_whole_component"] = False
    attrs["pragma_disable_loop_reversal"] = False
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = mean_compile(shape, dtype, axis, keepdims, kernel_name, attrs, tuning=t)
        if t:
            expect, input, output = gen_data(axis, dtype, keepdims, shape)
            return mod, expect, (input, output)
        else:
            return mod
    else:
        mod = mean_compile(shape, dtype, axis, keepdims, kernel_name, attrs)
        expect, input, output = gen_data(axis, dtype, keepdims, shape)
        args = [input, output]
        if attrs.get("dynamic"):
            _, dynamic_args = process_dynamic_shape([shape], attrs, keep_axis=[4])
            args += dynamic_args
            block_dim = compute_blockdim(shape, axis, dtype)
            args.append(block_dim)
        output = utils.mod_launch(mod, args, outputs=(1,), expect=expect)  # unified launch
        rtol, atol = get_rtol_atol("mean", dtype)
        return input, output, expect, compare_tensor(output, expect, rtol=rtol, atol=atol, equal_nan=True)


def gen_data(axis, dtype, keepdims, shape):
    support_list = {"float16": np.float16, "float32": np.float32}
    input = random_gaussian(shape, miu=0.05, sigma=0.1).astype(support_list[dtype])
    if dtype == "float16":
        expect = akg_fp16_mean(input, axis=axis, keepdims=keepdims)
    else:
        expect = np.mean(input, axis=axis, keepdims=keepdims)
    out_shape = get_reduce_out_shape(shape, axis=axis, keepdims=keepdims)
    output = np.full(out_shape, 0, dtype)
    return expect, input, output


def mean_compile(shape, dtype, axis, keepdims, kernel_name, attrs, tuning=False):
    if attrs is None:
        attrs = {}
    if attrs.get("dynamic"):
        build_shape, _ = process_dynamic_shape([shape], attrs, keep_axis=[4])
        build_shape = build_shape[0]
        attrs["enable_post_poly_loop_partition"] = False
    else:
        build_shape = shape
    return utils.op_build_test(mean.mean, [build_shape], [dtype], op_attrs=[axis, keepdims], kernel_name=kernel_name, attrs=attrs, tuning=tuning)
