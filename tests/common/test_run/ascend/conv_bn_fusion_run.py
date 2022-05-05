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

"""run function for fusion operation of splited bn2, add and relu"""
import time
import numpy as np
from akg.ops.nn.ascend import Conv, fused_bn1, fused_bn2, fused_bn3
from tests.common.gen_random import random_gaussian
from tests.common.tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from tests.common.test_run.ascend.conv_utils import conv_param_prepare, conv_shape_4d, \
    conv_forward_naive, conv_tensor_4d_to_5d
from tests.common.test_run.ascend.bn_split_run import get_compile_param as get_bn_split_param
from tests.common.test_run.ascend.bn_split_run import bn_benchmark

from tests.common.base import get_rtol_atol

def benchmark(x, w, bias, gamma, beta,
              running_mean, running_var, other_branch_data,
              pad, stride, dilation, momentum, eps, has_add, has_relu):
    """benchmark function"""
    conv_param = {'stride': stride, 'pad': pad, 'dilation': dilation}
    out = conv_forward_naive(
        x.astype(np.float32), w.astype(np.float32), bias, conv_param)
    _, _, _, conv_expect = conv_tensor_4d_to_5d(x, w, bias, out)

    axes = (0, 2, 3)
    conv_mean = np.mean(
        conv_expect.astype(np.float64), axis=axes, keepdims=True).astype(
            np.float32)
    mean_new = momentum * running_mean + (1 - momentum) * conv_mean
    var = np.var(
        conv_expect.astype(np.float64), axis=axes, keepdims=True).astype(
            np.float32)
    var_new = momentum * running_var + (1 - momentum) * var

    res = bn_benchmark(conv_expect, gamma, beta, running_mean, running_var, momentum, eps, 3)

    return conv_expect.astype(x.dtype), res[0].astype(x.dtype), \
        mean_new, var_new, conv_mean, var

def get_convbn1_compile_param(fm_shape, filter_shape, dtype,
                              pad, stride, dilation, use_bias):
    """get parameters for conv_bn1"""
    conv_dtype = "float16"

    conv_param = {'stride': stride, 'pad': pad, 'dilation': dilation}
    stride, pad, dilation = conv_param_prepare(conv_param)
    fm_shape_, w_shape, out_shape_4d = conv_shape_4d(
        fm_shape, filter_shape, pad, stride, dilation)
    input_n, input_channel, input_h, input_w = fm_shape_
    weight_n, weight_channel, weight_h, weight_w = w_shape
    block_size = 16

    if use_bias:
        in_shapes = [
            (input_n, input_channel // block_size,
             input_h, input_w, block_size),
            (weight_channel // block_size * weight_h * weight_w,
             weight_n // block_size, block_size, block_size),
            (1, weight_n // block_size, 1, 1, block_size)]
        in_dtypes = [conv_dtype] * 3
    else:
        in_shapes = [
            (input_n, input_channel // block_size,
             input_h, input_w, block_size),
            (weight_channel // block_size * weight_h * weight_w,
             weight_n // block_size, block_size, block_size)]
        in_dtypes = [conv_dtype] * 2

    shapes_4d = [fm_shape_, w_shape, (weight_n,)]
    op_attrs = [fm_shape, filter_shape, pad, stride, dilation, use_bias]
    assert len(out_shape_4d) == 4 and out_shape_4d[1] % block_size == 0, \
        "internal error."
    out_shape_5d = (out_shape_4d[0], out_shape_4d[1] // block_size,
                    out_shape_4d[2], out_shape_4d[3], block_size)

    return in_shapes, in_dtypes, op_attrs, shapes_4d, out_shape_5d

def get_compile_param(shape, dtype, momentum: float, eps: float,
                      is_1st_part, has_add, has_relu):
    """get parameters for fusion test compiling module"""
    mid_shape = (1, shape[1], 1, 1, shape[4])
    mid_dtype = "float32"

    if is_1st_part:
        in_shapes = [mid_shape] * 4
        in_dtypes = [mid_dtype] * 4
        op_attrs = [momentum]
    elif not has_add:
        in_shapes = [shape] + [mid_shape] * 4
        in_dtypes = [dtype] + [mid_dtype] * 4
        op_attrs = [eps]
    elif has_add and has_relu:
        in_shapes = [shape] * 2 + [mid_shape] * 4
        in_dtypes = [dtype] * 2 + [mid_dtype] * 4
        op_attrs = [eps]
    else:
        in_shapes = []
        in_dtypes = []
        op_attrs = []

    return in_shapes, in_dtypes, op_attrs

def gen_inputs_directly(in_shapes, in_dtypes, miu=1, sigma=0.3, is_conv=False):
    """gen inputs directly"""
    assert isinstance(in_shapes, (list, tuple)) \
        and isinstance(in_dtypes, (list, tuple)), \
        "parameters' type should be list or tuple"

    seed = int(time.time())
    inputs = []

    for i, (shape, dtype) in enumerate(zip(in_shapes, in_dtypes)):
        if not is_conv:
            inputs.append(random_gaussian(shape, miu=miu, sigma=sigma,
                                                seed=seed + i).astype(dtype))
        else:
            inputs.append(random_gaussian(shape, miu=miu,
                                          sigma=sigma).astype(dtype))
    return inputs[0] if len(inputs) == 1 else tuple(inputs)

def gen_convbn1_inputs(in_shapes, in_dtypes, shapes_4d, use_bias):
    """gen inputs for conv_bn1"""
    assert len(in_shapes) >= 2 and len(shapes_4d) >= 2 \
        and len(in_dtypes) >= 2, "internal error."
    inputs = list(gen_inputs_directly(shapes_4d[:2], in_dtypes[:2], is_conv=True))
    if use_bias:
        inputs.append(np.random.rand(*shapes_4d[2]).astype(
            np.float16, copy=False))
    else:
        inputs.append((np.array(np.zeros(*shapes_4d[2]))).astype(
            np.float16, copy=False))
    return inputs

def malloc_out_buffer(expects, full_value=0):
    """malloc buffer for outputs"""
    if not isinstance(expects, (list, tuple)):
        expects = [expects]
    return tuple([np.full(e.shape, full_value, e.dtype) for e in expects])

def fusion_gen_data(fm_shape, filter_shape, dtype, pad, stride, dilation,
                    use_bias, momentum=0.9, eps=1e-3,
                    has_add=False, has_relu=False):
    """Generate datas.

    Generate input datas, calculate expect results,
    and generate output_buffers.

    Args:
        fm_shape: Shape of convolution's input.
        filter_shape: Shape of convolution's filter.
        dtype: Data type of convolution's data.
        pad: list of 4 ints for convolution's pad parameters.
        stride: list of 2 insts for convolution's stride parameters.
        dilation: list of 2 ints for convolution's dilation parameters.
        use_bias: Whether convolution should consider bias.
        momentum: Momentum for moving average.
        eps: A small value for avoiding divide zero.
        other_branch_shape: Shape of data that comes from other branch and
                            will be added later.
        has_add: Whether this fusion function has add operator.
        has_relu: Whether this fusion function has relu operator.

    Returns:
        inputs: A tuple contain all generated input data.
        output_buffers: A tuple contain all generated output buffer.
        expects: A tuple contain expect results.
    """
    block_size = 16

    conv_in_shapes, conv_in_dtypes, _, shapes_4d, bn_shape = \
        get_convbn1_compile_param(fm_shape, filter_shape, dtype,
                                  pad, stride, dilation, use_bias)

    mid_shape = (1, bn_shape[1], 1, 1, bn_shape[4])
    mid_dtype = "float32"

    x_4d, filter_conv, bias = gen_convbn1_inputs(
        conv_in_shapes, conv_in_dtypes, shapes_4d, use_bias)
    gamma, beta, running_mean, running_var_tmp = \
        gen_inputs_directly([mid_shape] * 4, [mid_dtype] * 4)
    running_var = abs(running_var_tmp)

    inputs = []
    inputs_conv_bn1 = []
    in_n, in_channel, in_h, in_w = x_4d.shape
    inputs_conv_bn1.append(x_4d.reshape(
        in_n, in_channel // block_size, block_size, in_h, in_w).transpose(
            0, 1, 3, 4, 2).copy())
    weight_n, weight_channel, weight_h, weight_w = filter_conv.shape
    inputs_conv_bn1.append((
        filter_conv.reshape(
            weight_n, weight_channel // block_size, block_size, weight_h,
            weight_w).transpose(1, 3, 4, 0, 2).copy()
        ).reshape(
            weight_channel // block_size * weight_h * weight_w,
            weight_n // block_size, block_size, block_size))
    if use_bias:
        bias_n = bias.shape[0]
        inputs_conv_bn1.append(
            bias.reshape(1, bias_n // block_size, 1, 1, block_size))
    inputs.append(inputs_conv_bn1)

    if has_add:
        ob_data = gen_inputs_directly([bn_shape], [dtype])
        inputs_bn2_fusion = (ob_data, gamma, beta,
                             running_mean, running_var)
        to_pass_ins = (x_4d, filter_conv, bias, gamma, beta,
                       running_mean, running_var, ob_data)
    else:
        inputs_bn2_fusion = (gamma, beta,
                             running_mean, running_var)
        to_pass_ins = (x_4d, filter_conv, bias, gamma, beta,
                       running_mean, running_var, None)
    inputs.append(inputs_bn2_fusion)
    expects = benchmark(*to_pass_ins, pad, stride, dilation,
                        momentum, eps, has_add, has_relu)

    output_buffers = []
    output_buffers.append(tuple(
        [np.full(expects[0].shape, 0.0, dtype)] +
        [np.full(mid_shape, 0.0, mid_dtype)] * 2))
    output_buffers.append(tuple(
        [np.full(mid_shape, 0.0, mid_dtype)] * 3))
    output_buffers.append(malloc_out_buffer(expects[1]))
    return inputs, tuple(output_buffers), expects

def compare_result(res, expects, dtype="float16", rtol=None, atol=None):
    """compare list result"""
    if rtol is None or atol is None:
        rtol, atol = get_rtol_atol("bn_split", dtype)

    if not isinstance(res, tuple):
        res = (res,)

    return list(map(lambda x, y:
                    compare_tensor(x, y, rtol=rtol, atol=atol),
                    res, expects))

def conv_bn_fusion_run(fm_shape, filter_shape, dtype, pad, stride, dilation,
                       use_bias=False, momentum=0.9, eps=1e-3, attrs=None):
    """test run function for conv bn fusion"""
    ###########################################################################
    # compile each kernel
    ###########################################################################
    conv_in_shapes, conv_in_dtypes, conv_op_attrs, _, shape = \
        get_convbn1_compile_param(
            fm_shape, filter_shape, dtype, pad, stride, dilation, use_bias)

    # conv + bn1 + bn2 + bn3
    # conv
    mod_conv = utils.op_build_test(Conv, [conv_in_shapes], ['float16'],
                                   op_attrs=conv_op_attrs,
                                   kernel_name="conv_whole",
                                   attrs=attrs.copy())
    in_shapes_bn1, in_dtypes_bn1 = get_bn_split_param(shape, dtype, 1)
    mod_bn1 = utils.op_build_test(fused_bn1,
                                  in_shapes_bn1, in_dtypes_bn1,
                                  kernel_name="fused_bn1_whole",
                                  attrs=attrs.copy())

    in_shapes_bn2, in_dtypes_bn2 = get_bn_split_param(shape, dtype, 2)
    mod_bn2 = utils.op_build_test(fused_bn2,
                                  in_shapes_bn2, in_dtypes_bn2,
                                  op_attrs=[momentum],
                                  kernel_name="fused_bn2_whole",
                                  attrs=attrs.copy())

    in_shapes_bn3, in_dtypes_bn3 = get_bn_split_param(shape, dtype, 3)
    mod_bn3 = utils.op_build_test(fused_bn3,
                                  in_shapes_bn3, in_dtypes_bn3,
                                  op_attrs=[eps],
                                  kernel_name="fused_bn3_whole",
                                  attrs=attrs.copy())

    ###########################################################################
    # following run the kernel
    ###########################################################################
    inputs, output_buffers, expects = \
        fusion_gen_data(fm_shape, filter_shape, dtype, pad, stride, dilation,
                        use_bias, momentum, eps, False, False)

    inplace_binds = ((2, 1), (3, 2))
    output_places = list(range(-len(output_buffers[1]), 0))
    if inplace_binds is not None:
        for bind in inplace_binds:
            output_places[bind[1]] = bind[0]


    # origin run
    conv_out = utils.mod_launch(mod_conv, [*inputs[0], output_buffers[0][0]], expect=expects)

    bn1_out_buffers = tuple(
        [np.full([shape[0], shape[1], 1, 1, shape[4]], 0.0, "float32")] * 2)
    bn1_outs = utils.mod_launch(mod_bn1, [conv_out, *bn1_out_buffers],
                                outputs=list(range(-len(bn1_out_buffers), 0)))

    bn2_out_buffers = tuple(
        [np.full([1, shape[1], 1, 1, shape[4]], 0.0, "float32")] * 4)
    bn2_inplace_binds = ((2, 2), (3, 3))
    output_places_bn2 = list(range(-len(bn2_out_buffers), 0))
    if bn2_inplace_binds is not None:
        for bind in bn2_inplace_binds:
            output_places_bn2[bind[1]] = bind[0]
    bn2_outs = utils.mod_launch(mod_bn2,
                                [bn1_outs[0], bn1_outs[1],
                                 *inputs[1][2:], *bn2_out_buffers],
                                outputs=output_places_bn2, expect=expects)

    bn3_outs = utils.mod_launch(mod_bn3,
                                [conv_out, bn2_outs[0], bn2_outs[1],
                                 *inputs[1][:2], *output_buffers[2]],
                                outputs=list(range(-len(output_buffers[2]), 0)), expect=expects)

    origin_outputs = (conv_out, bn3_outs, bn2_outs[2], bn2_outs[3],
                      bn2_outs[0], bn2_outs[1])

    cmp_res_origin = compare_result(origin_outputs, expects, dtype)

    return inputs, origin_outputs, expects, all(cmp_res_origin)
