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

"""Generating test data for operators"""
from typing import NamedTuple

import numpy as np
from tests.common.gen_json_data import gen_json_data
from tests.common.test_run import batchmatmul_run, conv_run, conv_backprop_input_run, conv_backprop_filter_run, matmul_run
from tests.prev_version_auto_tune.type_definitions import ConvDesc, ConvBackpropDesc, MatmulCubeDesc

def _gen_data_json(op_desc):
    """Generating test data for composite json"""
    input_for_mod, expect, output_indexes = gen_json_data(op_desc)
    return input_for_mod, expect, output_indexes

def _gen_data_conv(op_desc: ConvDesc):
    """Generating test data for conv"""
    fmap_data, filter_data, bias_data, expect = conv_run.gen_data(op_desc.fmap_shape, op_desc.filter_shape,
                                                                  op_desc.pad, op_desc.stride, op_desc.dilation,
                                                                  op_desc.use_bias)
    out_data = np.full(expect.shape, 0, 'float16')

    if op_desc.use_bias:
        args = (fmap_data, filter_data, bias_data, out_data)
    else:
        args = (fmap_data, filter_data, out_data)
    return args, expect


def _gen_data_conv_bn1(op_desc: ConvDesc):
    """Generating test data for conv_bn1"""
    fmap_data, filter_data, bias_data, conv_expect = conv_run.gen_data(op_desc.fmap_shape, op_desc.filter_shape,
                                                                       op_desc.pad, op_desc.stride, op_desc.dilation,
                                                                       op_desc.use_bias)
    axes = (0, 2, 3)
    conv_mean = np.mean(conv_expect, axis=axes, keepdims=True)
    conv_square = np.power(conv_expect, 2)
    conv_var_part = np.mean(conv_square, axis=axes, keepdims=True)

    expects = (conv_expect, conv_var_part, conv_mean)

    out_datas = [np.full(e.shape, 0, 'float16') for e in expects]
    out_datas[1] = out_datas[1].astype(np.float32)
    out_datas[2] = out_datas[2].astype(np.float32)

    if op_desc.use_bias:
        in_data = [fmap_data, filter_data, bias_data]
    else:
        in_data = [fmap_data, filter_data]

    args = in_data
    for out in out_datas:
        args.append(out)
    args = tuple(args)

    return {"args": args, 'outputs': (-3, -2, -1)}, expects


def _gen_data_conv_backprop_input(op_desc: ConvBackpropDesc):
    dout, w, dx = conv_backprop_input_run.gen_data(op_desc.fmap_shape, op_desc.filter_shape, op_desc.pad,
                                                   op_desc.stride, op_desc.dilation)
    out_data = np.full(dx.shape, 0, 'float16')

    args = (dout, w, out_data)
    return args, dx


def _gen_data_conv_backprop_filter(op_desc: ConvBackpropDesc):
    """Generating test data for conv_backprop_filter"""
    block_size = 16

    in_n, in_c, in_h, in_w = op_desc.fmap_shape
    cout, _, w_h, w_w = op_desc.filter_shape

    in_c = (in_c + block_size - 1) // block_size * block_size
    cout = (cout + block_size - 1) // block_size * block_size

    x_shape = (in_n, in_c, in_h, in_w)
    w_shape = (cout, in_c, w_h, w_w)

    dy_data, dx_data, expect = conv_backprop_filter_run.gen_data(x_shape, w_shape, op_desc.pad, op_desc.stride,
                                                                 op_desc.dilation)
    out_data = np.full(expect.shape, 0, 'float32')

    args = (dy_data, dx_data, out_data)
    return args, expect


def _gen_data_matmul_cube(op_desc: MatmulCubeDesc):
    """Generating test data for matmul_cube"""
    batch_tuple, m, k, n = matmul_run.extract_dim(op_desc.x_shape, op_desc.y_shape, op_desc.adj_x, op_desc.adj_y)
    m = (m + 15) // 16 * 16
    n = (n + 15) // 16 * 16
    k = (k + 15) // 16 * 16
    _, _, _, out_shape, k = matmul_run.get_converted_shapes(m, n, k, batch_tuple, op_desc.adj_x, op_desc.adj_y,
                                                            op_desc.bias, op_desc.left_format, op_desc.right_format,
                                                            op_desc.out_format)
    m_x, m_y, bench_mark, bias_data = matmul_run.matmul_data(batch_tuple, m, k, n, op_desc.dtype, op_desc.bias_dtype,
                                                             op_desc.out_dtype, op_desc.bias, op_desc.adj_x,
                                                             op_desc.adj_y, op_desc.left_format,
                                                             op_desc.right_format, op_desc.out_format)

    out_data = np.full(out_shape, np.nan, op_desc.out_dtype)

    if op_desc.bias:
        args = (m_x, m_y, bias_data, out_data)
    else:
        args = (m_x, m_y, out_data)
    return args, bench_mark


_gen_data_func = {
    'json': _gen_data_json,
    'conv': _gen_data_conv,
    'conv_bn1': _gen_data_conv_bn1,
    'conv_backprop_input': _gen_data_conv_backprop_input,
    'conv_backprop_filter': _gen_data_conv_backprop_filter,
    'matmul': _gen_data_matmul_cube,
}


def gen_data(op_type: str, op_desc: NamedTuple):
    """Generate test data for operator

    Parameters
    op_type: str
        operator name
    op_desc: NamedTuple
        operator definition parameters
    ----------
    """
    gen_func = _gen_data_func.get(op_type, None)
    if gen_func is None:
        raise ValueError('Unsupported op type for test data generating: %s' % op_type)
    return gen_func(op_desc)
