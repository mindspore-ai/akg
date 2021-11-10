# Copyright 2021 Huawei Technologies Co., Ltd
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

import akg.tvm
import numpy as np
from akg.utils import kernel_exec as utils
from akg.ops.math.ascend import MatMul
from tests.common.test_run.ascend.matmul_run import *

def get_matmul_fractal_shape(x, format='zN'):
    shape = x.shape
    m, n = shape[-2], shape[-1]
    m1, n1 = m // 16, n // 16
    m0, n0 = 16, 16
    needPad = m % 16 != 0 or n % 16 != 0

    if format == 'zN':
        transpose_axis = [2, 0, 1, 3]
        new_shape = [n1, m1, m0, n0]
    elif format == 'zZ':
        transpose_axis = [0, 2, 1, 3]
        new_shape = [m1, n1, m0, n0]
    elif format == 'nZ':
        transpose_axis = [0, 2, 3, 1]
        new_shape = [m1, n1, n0, m0]
    return new_shape

def transdata_matmul(x, y, b, out_dtype, left_format="zZ", right_format="nZ", out_format="zN", transpose_x=False,
                    transpose_y=False, attrs={}, target="cce"):
    x_fractal_shape = get_matmul_fractal_shape(x, 'zN')
    y_fractal_shape = get_matmul_fractal_shape(y, 'zN')

    func = akg.tvm.get_global_func("TransData")
    x = func([x], {"src_format" : "DefaultFormat", "dst_format" : "FRACTAL_NZ", "output_shape": x_fractal_shape})
    y = func([y], {"src_format" : "DefaultFormat", "dst_format" : "FRACTAL_NZ", "output_shape": y_fractal_shape})

    res, attrs = MatMul(x, y, b, out_dtype, left_format, right_format, out_format, transpose_x, transpose_y, attrs=attrs)
    return res, attrs

def transdata_matmul_compile(shape_x, shape_y, bias, left_format, right_format, output_format, adj_x, adj_y, dtype, bias_dtype, out_dtype, kernel_name, attrs, tuning=False):
    batch_tuple, m, k, n = extract_dim(shape_x, shape_y, adj_x, adj_y)
    m = (m + 15) // 16 * 16
    n = (n + 15) // 16 * 16
    k = (k + 15) // 16 * 16
    shape_xx, shape_yy, bias_shape, out_shape, k = get_converted_shapes(m, n, k, batch_tuple, adj_x, adj_y, bias,
                                                                        left_format, right_format, output_format)
    input_shapes = [shape_x, shape_y, bias_shape]
    input_types = [dtype, dtype, bias_dtype]
    has_bias = False
    if bias == 1:
        has_bias = True
    op_attrs = [out_dtype, left_format, right_format, output_format, adj_x, adj_y, attrs]
    if has_bias == False:
        input_shapes = [shape_x, shape_y]
        input_types = [dtype, dtype]
        op_attrs = [None, out_dtype, left_format, right_format, output_format, adj_x, adj_y, attrs]
    return utils.op_build_test(transdata_matmul, input_shapes, input_types, op_attrs, kernel_name, attrs=attrs, tuning=tuning)

def transdata_matmul_execute(shape_x, shape_y, bias, left_format, right_format, out_format, adj_x, adj_y, dtype, bias_dtype, out_dtype, kernel_name, attrs={}):
    batch_tuple, m, k, n = extract_dim(shape_x, shape_y, adj_x, adj_y)
    m = (m + 15) // 16 * 16
    n = (n + 15) // 16 * 16
    k = (k + 15) // 16 * 16
    shape_xx, shape_yy, bias_shape, out_shape, k = get_converted_shapes(m, n, k, batch_tuple, adj_x, adj_y, bias, left_format, right_format, out_format)
    mod = transdata_matmul_compile(shape_x, shape_y, bias, left_format, right_format, out_format, adj_x, adj_y, dtype, bias_dtype, out_dtype, kernel_name, attrs=attrs)
    # Generate data
    m_x_fractal, m_y_fractal, bench_mark, bias_data, m_x, m_y = gen_data_all(batch_tuple, m, k, n, adj_x, adj_y, dtype, bias_dtype, out_dtype, bias, left_format, right_format, out_format)

    # mod launch
    output = np.full(bench_mark.shape, np.nan, out_dtype)
    if bias == 0:
        output = utils.mod_launch(mod, (m_x, m_y, output), expect=bench_mark)
    elif bias == 1:
        output = utils.mod_launch(mod, (m_x, m_y, bias_data, output), expect=bench_mark)
    # compare result
    rtol, atol = get_rtol_atol("matmul", dtype)
    compare_result = compare_tensor(output, bench_mark, rtol=rtol, atol=atol, equal_nan=True)
    return (m_x, m_y), output, bench_mark, compare_result
