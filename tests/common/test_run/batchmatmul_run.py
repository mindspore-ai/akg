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
import logging
import numpy as np
import akg
from akg import tvm
from tensorio import compare_tensor
from base import get_rtol_atol
from akg.utils import kernel_exec as utils
from akg.ops.nn import batchmatmul
from akg.utils.result_analysis import result_compare
from gen_random import random_gaussian

logging.basicConfig(level=logging.DEBUG)


def compute_blockdim(batch, shape):
    blockdim_limit = 2 if utils.product_is_mini() else 32
    blockdim = 1
    if isinstance(batch, (list, tuple)) and len(batch) > 1:
        blockdim = batch[0]
    else:
        if isinstance(shape, (list, tuple)) and len(shape) == 2:
            if shape[0] > 32:
                blockdim = 32
            else:
                blockdim = shape[0]
        else:
            raise TypeError("Shape to compute blockdim must be a 2-D list/tuple")
    return min(blockdim_limit, blockdim)

def get_shape(bs, m, n, k, trans_a, trans_b):
    if trans_a:
        shape_A = (*bs, k, m)
    else:
        shape_A = (*bs, m, k)
    if trans_b:
        shape_B = (*bs, n, k)
    else:
        shape_B = (*bs, k, n)
    shape_C = (*bs, m, n)
    return shape_A, shape_B, shape_C


def double_fivethou_compare(actual, bench_mark, r_tol=5e-3, equal_nan=True):
    """
    Function for compare result according to double fivethou accuracy.

    Args:
        actual: actual.
        bench_mark: bench_mark.
        r_tol: Default: 5e-3.
        equal_nan: Default: True.

    Returns:
        res.
    """
    cmp_array = np.isclose(actual, bench_mark, rtol=r_tol, equal_nan=equal_nan)
    pass_num = cmp_array.astype("int").sum()
    total_num = np.size(cmp_array)
    fail_num = total_num - pass_num
    fail_rate = 1.0 * fail_num / (1.0 * total_num)
    res = False
    if fail_rate < 5e-3:
        print("Failed rate: {0} / {1} = {2}".format(fail_num, total_num, fail_rate))
        res = result_compare(actual, bench_mark, r_tol=r_tol)
    return res


def gen_data(bs, m, n, k, shape_bias, trans_a, trans_b, dtype):
    shape_a, shape_b, shape_out = get_shape(bs, m, n, k, trans_a, trans_b)
    matrix_a = random_gaussian(shape_a, miu=0.5, sigma=0.01).astype(dtype)
    matrix_b = random_gaussian(shape_b, miu=0.5, sigma=0.01).astype(dtype)
    if len(shape_bias) > 0:
        matrix_bias = random_gaussian(shape_bias, miu=0.5, sigma=0.01).astype(dtype)
    else:
        matrix_bias = np.zeros(shape_bias, dtype=dtype)

    # cast to float32 for fast compute in numpy
    if dtype == "float16":
        matrix_a_for_np = matrix_a.astype(np.float32)
        matrix_b_for_np = matrix_b.astype(np.float32)
        matrix_bias_for_np = matrix_bias.astype(np.float32)
    else:
        matrix_a_for_np = matrix_a
        matrix_b_for_np = matrix_b
        matrix_bias_for_np = matrix_bias
    if trans_a and trans_b:
        res = np.matmul(np.swapaxes(matrix_a_for_np, -1, -2), np.swapaxes(matrix_b_for_np, -1, -2))
    elif trans_a:
        res = np.matmul(np.swapaxes(matrix_a_for_np, -1, -2), matrix_b_for_np)
    elif trans_b:
        res = np.matmul(matrix_a_for_np, np.swapaxes(matrix_b_for_np, -1, -2))
    else:
        res = np.matmul(matrix_a_for_np, matrix_b_for_np)

    res = np.add(res, matrix_bias_for_np)

    if dtype == "float16":
        res = res.astype(dtype)
    return matrix_a, matrix_b, matrix_bias, res


def batchmatmul_execute(bs, m, n, k, bias_shape, dtype, trans_a, trans_b, kernel_name, attrs):
    # Generate data
    data_shape, weight_shape, out_shape = get_shape(bs, m, n, k, trans_a, trans_b)

    # this part is for auto-tuning
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = batchmatmul_compile(bs, m, n, k, bias_shape, dtype, trans_a, trans_b, kernel_name, attrs, t)
        if t:
            m_a, m_b, matrix_bias, expect = gen_data(bs, m, n, k, bias_shape, trans_a, trans_b, dtype)
            output = np.full(out_shape, np.nan, dtype=dtype)
            return mod, expect, (m_a, m_b, matrix_bias, output) if len(bias_shape) > 0 else (m_a, m_b, output)
        else:
            return mod

    mod, args = batchmatmul_compile(bs, m, n, k, bias_shape, dtype, trans_a, trans_b, kernel_name, attrs)
    m_a, m_b, matrix_bias, expect = gen_data(bs, m, n, k, bias_shape, trans_a, trans_b, dtype)
    output = np.full(out_shape, np.nan, dtype=dtype)
    launch_args = []
    outputs = []
    if len(bias_shape) > 0:
        launch_args = [m_a, m_b, matrix_bias, output]
        outputs = [3,]
    else:
        launch_args = [m_a, m_b, output]
        outputs = [2,]
    if attrs.get("dynamic"):
        launch_args = launch_args + args
        block_dim = compute_blockdim(bs, (m, n))
        launch_args.append(block_dim)
    output = utils.mod_launch(mod, launch_args, outputs=outputs, expect=expect)
    rtol, atol = get_rtol_atol("batchmatmul", dtype)
    print("-----------------")
    print(output.dtype)
    print(expect.dtype)
    res = compare_tensor(output, expect, rtol=rtol, atol=atol, equal_nan=True)
    # if not res:
    #     res = utils.double_fivethou_compare(output, expect, r_tol=5e-3, equal_nan=True)
    return (m_a, m_b), output, expect, res


def get_vars(input_shapes, input_types):
    shape_params = []
    for i, (shape, dtype) in enumerate(zip(input_shapes, input_types)):
        if shape and isinstance(shape, (list, tuple)) and isinstance(shape[0], (list, tuple)):
            tmp_input = []
            for j, tmp_shape in enumerate(shape):
                if isinstance(tmp, akg.tvm.expr.Var):
                    shape_params.append(tmp)
        elif shape and isinstance(shape, (list, tuple)) and isinstance(shape[0], akg.tvm.expr.Var):
            for tmp_shape in shape:
                if isinstance(tmp_shape, akg.tvm.expr.Var):
                    shape_params.append(tmp_shape)
        elif isinstance(shape, akg.tvm.tensor.Tensor):
            for tmp_shape in shape.shape:
                shape_params.append(tmp_shape)
    shape_var = []
    [shape_var.append(i) for i in shape_params if i not in shape_var]
    return shape_var


def batchmatmul_compile(bs, m, n, k, bias_shape, dtype, trans_a, trans_b, kernel_name, attrs, tuning=False):
    args_dict = {}
    if attrs.get("dynamic"):
        bs_vars = ()
        for i in range(len(bs)):
            tmp = tvm.var('I'+str(i))
            bs_vars = bs_vars + (tmp,)
            args_dict[tmp] = bs[i]

        bs = bs_vars
        index = len(bs)
        m_var = tvm.var('I'+str(index))
        n_var = tvm.var('I'+str(index + 1))
        k_var = tvm.var('I'+str(index + 2))
        args_dict[m_var] = m
        args_dict[n_var] = n
        args_dict[k_var] = k
        m = m_var
        n = n_var
        k = k_var

        index = index + 3
        bias_vars = ()
        if len(bias_shape) > 0:
            for i in range(len(bias_shape)):
                tmp = tvm.var('I'+str(i + index))
                args_dict[tmp] = bias_shape[i]
                bias_vars = bias_vars + (tmp,)
            bias_shape = bias_vars

    data_shape, weight_shape, out_shape = get_shape(bs, m, n, k, trans_a, trans_b)
    if len(bias_shape) > 0:
        input_shapes = [data_shape, weight_shape, bias_shape]
        input_types = [dtype, dtype, dtype]
    else:
        input_shapes = [data_shape, weight_shape]
        input_types = [dtype, dtype]

    op_attrs = [trans_a, trans_b]
    args = get_vars(input_shapes, input_types)
    args_value = []
    for item in args:
        if item in args_dict:
            args_value.append(args_dict[item])

    if len(bias_shape) > 0:
        return utils.op_build_test(batchmatmul.batchmatmul_bias, input_shapes, input_types, op_attrs,
                                   kernel_name, attrs, tuning=tuning), args_value
    else:
        return utils.op_build_test(batchmatmul.batchmatmul, input_shapes, input_types, op_attrs,
                                   kernel_name, attrs, tuning=tuning), args_value
