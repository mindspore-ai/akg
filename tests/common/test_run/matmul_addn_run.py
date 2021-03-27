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

import numpy as np
from akg.utils import kernel_exec as utils
from akg.ops.nn import matmul
from tests.common.test_run.matmul_run import *
from akg.ops.math import addn
from akg.ms.cce import add

def matmul_addn(x, y, adds, b, out_dtype, left_format="zZ", right_format="nZ", out_format="zN", transpose_x=False, transpose_y=False, attrs={}):
    matmul_res, attrs = matmul.matmul(x, y, b, out_dtype, left_format, right_format, out_format, transpose_x, transpose_y, attrs=None)
    attr = {}
    addn_res = addn.addn(adds)
    res, attr_map= add.TensorAdd(matmul_res, addn_res)
    return res, attrs


def matmul_addn_compile(shape_x, shape_y, bias, add_n, left_format, right_format, output_format, adj_x, adj_y, dtype, bias_dtype, out_dtype, kernel_name, attrs, tuning=False):
    batch_tuple, m, k, n = extract_dim(shape_x, shape_y, adj_x, adj_y)
    m = (m + 15) // 16 * 16
    n = (n + 15) // 16 * 16
    k = (k + 15) // 16 * 16
    shape_xx, shape_yy, bias_shape, out_shape, k = get_converted_shapes(m, n, k, batch_tuple, adj_x, adj_y, bias,
                                                                        left_format, right_format, output_format)
    shape_dy = out_shape
    addn_shapes = []
    for i in range(add_n):
        addn_shapes.append(out_shape)
    
    input_shapes = [shape_xx, shape_yy, addn_shapes, bias_shape]
    input_types = [dtype, dtype, out_dtype, bias_dtype]
    has_bias = False
    if bias == 1:
        has_bias = True
    op_attrs = [out_dtype, left_format, right_format, output_format, adj_x, adj_y, attrs]
    if has_bias == False:
        input_shapes = [shape_xx, shape_yy, addn_shapes]
        input_types = [dtype, dtype, out_dtype]
        op_attrs = [None, out_dtype, left_format, right_format, output_format, adj_x, adj_y, attrs]
    return utils.op_build_test(matmul_addn, input_shapes, input_types, op_attrs, kernel_name, attrs=attrs, tuning=tuning)

def matmul_addn_execute(shape_x, shape_y, bias , add_n, left_format, right_format, out_format, adj_x, adj_y, dtype, bias_dtype, out_dtype, kernel_name, attrs={}):
    batch_tuple, m, k, n = extract_dim(shape_x, shape_y, adj_x, adj_y)
    m = (m + 15) // 16 * 16
    n = (n + 15) // 16 * 16
    k = (k + 15) // 16 * 16
    shape_xx, shape_yy, bias_shape, out_shape, k = get_converted_shapes(m, n, k, batch_tuple, adj_x, adj_y, bias, left_format, right_format, out_format)
    mod = matmul_addn_compile(shape_x, shape_y, bias, add_n, left_format, right_format, out_format, adj_x, adj_y, dtype, bias_dtype, out_dtype, kernel_name, attrs={})
    # Generate data
    m_x, m_y, bench_mark, bias_data = matmul_data(batch_tuple, m, k, n, dtype, bias_dtype, out_dtype, bias, adj_x, adj_y, left_format, right_format, out_format)
    
    inputs = []
    for i in range(add_n):
        input = random_gaussian(out_shape, miu=1, sigma=0.1).astype(out_dtype)
        inputs.append(input)
    bench_mark = np.add(np.sum(inputs, axis=0), bench_mark)

    # mod launch
    output = np.full(out_shape, np.nan, out_dtype)
    
    if bias == 0:
        inputs.insert(0, m_y)
        inputs.insert(0, m_x)
        inputs.append(output)
        output = utils.mod_launch(mod, inputs, expect=bench_mark)
    elif bias == 1:
        inputs.insert(0, m_y)
        inputs.insert(0, m_x)
        inputs.append(bias_data)
        inputs.append(output)

        output = utils.mod_launch(mod, inputs, expect=bench_mark)
    # compare result
    rtol, atol = get_rtol_atol("matmul", dtype)
    compare_result = compare_tensor(output, bench_mark, rtol=rtol, atol=atol, equal_nan=True)
    return (m_x, m_y), output, bench_mark, compare_result
