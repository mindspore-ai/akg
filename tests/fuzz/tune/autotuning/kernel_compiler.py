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

"""Compile kernel module for operator"""
import os
from typing import NamedTuple
from tests.common.base import TestBase
from akg import build_module
from akg.utils import kernel_exec as utils
from akg.utils import custom_tiling as ct_util
from akg.ops.nn import conv_bn1
from akg.ops.nn import conv, conv_backprop_input, conv_backprop_filter, batchmatmul
from akg.ops.nn import matmul
from tests.common.test_run import batchmatmul_run, matmul_run
from tests.fuzz.tune.autotuning.type_definitions import ConvDesc, ConvBackpropDesc, MatmulCubeDesc, ConvConfig, ConvBackpropInputConfig, ConvBackpropFilterConfig, MatmulCubeConfig


def gen_kernel_conv(op_desc: ConvDesc, input_shape, index_table,
                    config: ConvConfig = None, idx=None, gen_tiling_spaces=False):
    """Compile kernel module for conv"""
    if index_table is not None:
        raise RuntimeError('index_table should be none')
    kernel_name = "conv_poly"
    if idx is not None:
        kernel_name += str(idx)

    if config is None:
        attrs = {'dim': ""}
    else:
        tile_hh = config.tile_h
        tile_coco = config.tile_co
        tile_mm = config.tile_m
        tile_kk = config.tile_k
        tile_nn = config.tile_n
        tile_ww = config.tile_w
        tiling_param = [tile_hh, tile_coco, tile_mm, tile_kk, tile_nn, tile_ww]
        attrs = {'conv_tile': tiling_param, 'bypass': config.bypass}

    if op_desc.use_bias:
        shape = [input_shape[0], input_shape[1], input_shape[2]]
    else:
        shape = [input_shape[0], input_shape[1]]
    conv_dtype = 'float16'

    return utils.op_build(conv.conv, [shape], [conv_dtype],
                          op_attrs=[op_desc.fmap_shape, op_desc.filter_shape, op_desc.pad, op_desc.stride,
                                    op_desc.dilation, op_desc.use_bias, attrs],
                          kernel_name=kernel_name, attrs=attrs, polyhedral=True, tuning=gen_tiling_spaces)


def gen_kernel_conv_bn1(op_desc: ConvDesc, input_shape, index_table, config: ConvConfig = None,
                        idx=None, gen_tiling_spaces=False):
    """Compile kernel module for conv_bn1"""
    if index_table is not None:
        raise RuntimeError('index_table should be none')
    kernel_name = "conv_bn1_poly"
    if idx is not None:
        kernel_name += str(idx)

    if config is None:
        attrs = {'dim': ""}
    else:
        tile_hh = config.tile_h
        tile_coco = config.tile_co
        tile_mm = config.tile_m
        tile_kk = config.tile_k
        tile_nn = config.tile_n
        tile_ww = config.tile_w
        tiling_param = [tile_hh, tile_coco, tile_mm, tile_kk, tile_nn, tile_ww]
        attrs = {'conv_tile': tiling_param, 'bypass': config.bypass}

    if op_desc.use_bias:
        shape = [input_shape[0], input_shape[1], input_shape[2]]
    else:
        shape = [input_shape[0], input_shape[1]]
    conv_dtype = 'float16'

    return utils.op_build(conv_bn1.conv_bn1, [shape], [conv_dtype],
                          op_attrs=[op_desc.fmap_shape, op_desc.filter_shape, op_desc.pad, op_desc.stride,
                                    op_desc.dilation, op_desc.use_bias, attrs],
                          kernel_name=kernel_name, attrs=attrs, polyhedral=True, tuning=gen_tiling_spaces)


def gen_kernel_matmul_cube(op_desc: MatmulCubeDesc, _, index_table,
                           config: MatmulCubeConfig = None, idx=None, gen_tiling_spaces=False):
    """Compile kernel module for matmul_cube"""
    if index_table is not None:
        raise RuntimeError('index_table should be none')
    kernel_name = "matmul_cube_poly"
    if idx is not None:
        kernel_name += str(idx)
    if config is None:
        attrs = {'dim': ""}
    else:
        tiling_param = []
        for _ in range(len(op_desc.x_shape) - 2):
            tiling_param.append((1, 1))
        if config.n_l1 > 0:
            tiling_param.append((config.n_l1, config.n_l0))
        if config.m_l1 > 0:
            tiling_param.append((config.m_l1, config.m_l0))
        tiling_param.extend([(16, 16), (16, 16), (config.k_l1, config.k_l0)])
        dim_info = ct_util.set_dims(tuple(tiling_param))
        attrs = {'dim': dim_info, 'bypass': config.bypass}
    return matmul_run.matmul_compile(op_desc.x_shape, op_desc.y_shape, op_desc.bias, op_desc.left_format,
                                     op_desc.right_format, op_desc.out_format, op_desc.adj_x, op_desc.adj_y,
                                     op_desc.dtype, op_desc.bias_dtype, op_desc.out_dtype, kernel_name,
                                     attrs, tuning=gen_tiling_spaces)


def gen_kernel_conv_backprop_input(op_desc: ConvBackpropDesc, _, index_table, config: ConvBackpropInputConfig = None,
                                   idx=None, gen_tiling_spaces=False):
    """Compile kernel module for conv_backprop_input"""
    if index_table is not None:
        raise RuntimeError('index_table should be none')
    kernel_name = "conv_backprop_input_poly"
    if idx is not None:
        kernel_name += str(idx)

    if config is None:
        attrs = {'dim': ""}
    else:
        tile_hh = config.tile_h
        tile_coco = config.tile_co
        tile_mm = config.tile_m
        tile_kk = config.tile_k
        tile_nn = config.tile_n
        tile_ww = config.tile_w
        tiling_param = [tile_hh, tile_coco, tile_mm, tile_kk, tile_nn, tile_ww]
        attrs = {'conv_tile': tiling_param}

    conv_dtype = 'float16'
    block_size = 16

    in_n, in_c, in_h, in_w = op_desc.fmap_shape
    cout, _, w_h, w_w = op_desc.filter_shape

    in_c = (in_c + block_size - 1) // block_size * block_size
    cout = (cout + block_size - 1) // block_size * block_size

    pad_top, pad_bottom, pad_left, pad_right = op_desc.pad
    stride_h, stride_w = op_desc.stride

    out_n = in_n
    out_c = cout
    out_h = (in_h + pad_top + pad_bottom - w_h) // stride_h + 1
    out_w = (in_w + pad_left + pad_right - w_w) // stride_w + 1

    x_shape = (out_n, out_c, out_h, out_w)
    w_shape = (cout, in_c, w_h, w_w)
    in_nn, in_cc, in_hh, in_ww = x_shape
    input_shape_nc1hwc0 = (in_nn, in_cc // block_size, in_hh, in_ww, block_size)
    k_n, k_c, k_h, k_w = w_shape
    kernel_shape_nc1hwc0 = (k_n, k_c // block_size, k_h, k_w, block_size)
    k_n, _, k_h, k_w, _ = kernel_shape_nc1hwc0
    kernel_shape_fractal = (k_c // block_size * k_h * k_w, k_n // block_size, block_size, block_size)

    shape = [input_shape_nc1hwc0, kernel_shape_fractal]

    return utils.op_build(conv_backprop_input.conv_backprop_input, [shape], [conv_dtype],
                          op_attrs=[op_desc.fmap_shape, op_desc.filter_shape, op_desc.pad,
                                    op_desc.stride, op_desc.dilation, attrs],
                          kernel_name=kernel_name, attrs=attrs, polyhedral=True, tuning=gen_tiling_spaces)


def gen_kernel_conv_backprop_filter(op_desc: ConvBackpropDesc, _, index_table, config: ConvBackpropFilterConfig = None,
                                    idx=None, gen_tiling_spaces=False):
    """Compile kernel module for conv_backprop_filter"""
    if index_table is not None:
        raise RuntimeError('index_table should be none')
    kernel_name = "conv_backprop_filter_poly"
    if idx is not None:
        kernel_name += str(idx)

    if config is None:
        attrs = {'dim': ""}
    else:
        tile_cici = config.tile_ci
        tile_khkh = config.tile_kh
        tile_kwkw = config.tile_kw
        tile_coco = config.tile_co
        tile_bb = config.tile_batch
        tile_hh = config.tile_h
        tile_ww = config.tile_w
        tile_mm = config.tile_m
        tile_kk = config.tile_k
        tile_nn = config.tile_n
        tiling_param = [tile_cici, tile_khkh, tile_kwkw, tile_coco, tile_bb, tile_hh, tile_ww,
                        tile_mm, tile_kk, tile_nn]
        attrs = {'conv_tile': tiling_param}

    conv_dtype = 'float16'
    block_size = 16

    in_n, in_c, in_h, in_w = op_desc.fmap_shape
    cout, _, w_h, w_w = op_desc.filter_shape

    in_c = (in_c + block_size - 1) // block_size * block_size
    cout = (cout + block_size - 1) // block_size * block_size

    pad_top, pad_bottom, pad_left, pad_right = op_desc.pad
    stride_h, stride_w = op_desc.stride

    out_n = in_n
    out_c = cout
    out_h = (in_h + pad_top + pad_bottom - w_h) // stride_h + 1
    out_w = (in_w + pad_left + pad_right - w_w) // stride_w + 1

    x_shape = (in_n, in_c, in_h, in_w)
    y_shape = (out_n, out_c, out_h, out_w)
    in_n, in_c, in_h, in_w = x_shape
    input_shape_nc1hwc0 = (in_n, in_c // block_size, in_h, in_w, block_size)
    o_n, o_c, o_h, o_w = y_shape
    kernel_shape_nc1hwc0 = (o_n, o_c // block_size, o_h, o_w, block_size)
    o_n, o_c1, o_h, o_w, o_c0 = kernel_shape_nc1hwc0
    mo = (o_h * o_w + block_size - 1) // block_size
    mi = block_size
    kernel_shape_fractal = (o_n, o_c1, mo, mi, o_c0)

    input_shape = [kernel_shape_fractal, input_shape_nc1hwc0]

    return utils.op_build(conv_backprop_filter.conv_backprop_filter, [input_shape], [conv_dtype],
                          op_attrs=[op_desc.fmap_shape, op_desc.filter_shape, op_desc.pad,
                                    op_desc.stride, op_desc.dilation, attrs],
                          kernel_name=kernel_name, attrs=attrs, polyhedral=True, tuning=gen_tiling_spaces)


def gen_kernel_for_vector(op_desc, _, index_table=None, config: NamedTuple = None, idx=None, gen_tiling_spaces=False):
    """Compile kernel module for vector"""
    test_base = TestBase()
    test_base.params_init(op_desc[0][0:4] + str(idx), os.getcwd())
    kernel_name = "poly_"
    if idx is not None:
        kernel_name += str(idx)
    if config is None:
        attrs = {'dim': ""}
    else:
        tiling = [[getattr(config, name), 1] for name in getattr(config, '_fields') if name.startswith('tiling')]
        tiling_param = []
        for i, element in enumerate(tiling):
            tiling_param.append(index_table[i] + element)
        dim_info = ct_util.set_dims(tuple(tiling_param))
        attrs = {'dim': dim_info}
    _, func, args, kwargs = test_base.ana_args(op_desc)
    if 'attrs' in kwargs.keys():
        kwargs['attrs']['dim'] = attrs['dim']
        kwargs['attrs']['tuning'] = gen_tiling_spaces
        kwargs['attrs']['kernel_name'] = kernel_name
    else:
        for _, arg_ in enumerate(args):
            if isinstance(arg_, dict):
                arg_['dim'] = attrs['dim']
                arg_['tuning'] = gen_tiling_spaces
                arg_['kernel_name'] = kernel_name
                break
    try:
        if gen_tiling_spaces:
            mod, expect, param_for_mod = func(*args, **kwargs)
            mod = list(mod)
            mod.append(expect)
            mod.append(param_for_mod)
        else:
            mod = func(*args, **kwargs)
    except BaseException as e:
        print("Compile ERROR message:", e)
        print(func)
        print("Compile ERROR")
        raise Exception("Compile ERROR")

    return mod


_compile_kernel_func = {
    'conv': gen_kernel_conv,
    'conv_bn1': gen_kernel_conv_bn1,
    'conv_backprop_input': gen_kernel_conv_backprop_input,
    'conv_backprop_filter': gen_kernel_conv_backprop_filter,
    'matmul': gen_kernel_matmul_cube,
}


def compile_kernel(op_type: str, op_desc: NamedTuple, input_shape=None, index_table=None,
                   config_param: NamedTuple = None, idx: int = None, gen_tiling_spaces: bool = False):
    """Generate kernel module for operator

    Parameters
    op_type: str
        operator name
    op_desc: NamedTuple
        operator definition parameters
    config_param: NameTuple
        operator config  parameters
    idx: int
        operator idx(th) kernel
    gen_tiling_spaces: bool
        parameter passed to utils.op_build, whether to get spaces instead of stmt
    ----------

    Returns:
        kernel if gen_tiling_spaces == False else np.ndarray
    """
    gen_func = _compile_kernel_func.get(op_type, None)
    if gen_func is None:
        gen_func = gen_kernel_for_vector
    if gen_tiling_spaces:
        mod, key, expect, input_for_mod = gen_func(op_desc, input_shape, index_table, config_param,
                                                   idx, gen_tiling_spaces)
    else:
        mod = gen_func(op_desc, input_shape, index_table, config_param, idx, gen_tiling_spaces)
    return [build_module.tuning_spaces, key, expect, input_for_mod] if gen_tiling_spaces else mod
