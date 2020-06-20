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

from test_run.matmul_run import extract_dim, get_converted_shapes
from test_op import matmul4d_ad
import logging
from tensorio import compare_tensor
import numpy as np
from akg.utils import kernel_exec as utils
from gen_random import random_gaussian
logging.basicConfig(level=logging.DEBUG)


def extract_dim(shape_x, shape_y, adj_x, adj_y):
    rank = len(shape_x)
    m = shape_x[-2] if adj_x == False else shape_x[-1]
    k = shape_x[-1] if adj_x == False else shape_x[-2]
    n = shape_y[-1] if adj_y == False else shape_y[-2]
    batch_tuple = shape_x[:-2] if rank > 2 else (1,)
    return batch_tuple, m, k, n


def compute_expected(input_y, input_head, adj_x, adj_y, shape_xx):
    if adj_x:
        if adj_y:
            expected = np.tensordot(input_head, input_y, axes=([1, 4], [1, 4])).transpose(0, 3, 4, 1, 5, 2).reshape(shape_xx)
        else:
            expected = np.tensordot(input_head, input_y, axes=([1, 4], [2, 3])).transpose(0, 3, 4, 1, 5, 2).reshape(shape_xx)
    else:
        if adj_y:
            expected = np.tensordot(input_head, input_y, axes=([1, 4], [1, 4])).transpose(0, 3, 1, 4, 2, 5).reshape(shape_xx)
        else:
            expected = np.tensordot(input_head, input_y, axes=([1, 4], [2, 3])).transpose(0, 3, 1, 4, 2, 5).reshape(shape_xx)

    return expected


def matmul4d_ad_run(shape_x, shape_y, bias,  adj_x, adj_y, dtype, out_dtype, kernel_name, attrs):

    # calculate the shape in fractal type and create the data
    batch_tuple, m, k, n = extract_dim(shape_x, shape_y, adj_x, adj_y)

    m = (m + 15) // 16 * 16
    n = (n + 15) // 16 * 16
    k = (k + 15) // 16 * 16

    shape_xx, shape_yy, bias_shape, output_shape, k = get_converted_shapes(m, n, k, batch_tuple, adj_x, adj_y, bias)

    input_x = random_gaussian(shape_xx, miu=0.5, sigma=0.01).astype(np.float16)
    input_y = random_gaussian(shape_yy, miu=0.5, sigma=0.01).astype(np.float16)
    input_head = random_gaussian(output_shape, miu=0.5, sigma=0.01).astype(np.float16)

    dX_expected = compute_expected(input_y, input_head, adj_x, adj_y, shape_xx)

    input_shapes = [output_shape, shape_xx, shape_yy, bias_shape]
    input_types = [out_dtype, dtype, dtype, dtype]
    op_attrs = [out_dtype, adj_x, adj_y]
    if bias_shape is None:
        input_shapes = [output_shape, shape_xx, shape_yy]
        input_types = [out_dtype, dtype, dtype]
        op_attrs = [None, out_dtype, adj_x, adj_y]

    mod = utils.op_build_test(matmul4d_ad.matmul4d_ad, input_shapes, input_types, op_attrs, kernel_name, attrs)

    # calculate the backward kernel
    dX = np.full(shape_xx, np.nan, dtype)
    dX = utils.mod_launch(mod, (input_head, input_x, input_y, dX), expect=dX_expected)

    return (input_x, input_y, input_head), dX, dX_expected, compare_tensor(dX, dX_expected, rtol=0.01, equal_nan=True)
