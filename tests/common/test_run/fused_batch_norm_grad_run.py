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

from akg.utils import kernel_exec as utils
import numpy as np
from akg.ops.nn import fused_batch_norm_grad
from akg.ops.nn import fused_batch_norm_grad_split
from functools import reduce
from tests.common.base import get_rtol_atol
from tests.common.gen_random import random_gaussian

def benchmark(dy, data, mean, var, gamma, eps, data_format, axis):
    shape = data.shape
    ori_dtype = data.dtype
    is_special5D = (data_format == "NC1HWC0")

    # use float64 for higher precision
    dtype = "float64"
    dy = dy.astype(dtype)
    data = data.astype(dtype)
    mean = mean.astype(dtype)
    var = var.astype(dtype)
    gamma = gamma.astype(dtype)

    if is_special5D:
        axes = (0, 2, 3)
        keepdims = True
        mid_shape = [1, shape[1], 1, 1, shape[4]]
    else:
        axis = axis if axis >= 0 else axis + len(shape)
        axes = tuple([i for i in range(len(shape)) if i != axis])
        keepdims = False
        mid_shape = [1] * len(shape)
        mid_shape[axis] = shape[axis]

    m = reduce(lambda i, j: i * j, [shape[i] for i in axes])
    eps = np.array([eps], dtype=dtype).reshape([1] * len(shape))
    m_rec = np.array([1.0 / m], dtype=dtype).reshape([1] * len(shape))
    one = np.array([1.0], dtype=dtype).reshape([1] * len(shape))

    rsqvar = (one / np.sqrt(var.reshape(mid_shape) + eps))
    norm = (data - mean.reshape(mid_shape)) * rsqvar

    dgamma = np.sum(dy * norm, axis=axes, keepdims=keepdims)
    dbeta = np.sum(dy, axis=axes, keepdims=keepdims)
    dx = gamma.reshape(mid_shape) * rsqvar * \
        (dy - m_rec * dbeta.reshape(mid_shape) - m_rec * norm * dgamma.reshape(mid_shape))

    dx = dx.astype(ori_dtype)
    dgamma = dgamma.astype("float32")
    dbeta = dbeta.astype("float32")

    return [dx, dgamma, dbeta]

def fused_bn_grad_5D_run_1(shape, dtype, kernel_name, attrs):
    """ test bnGrad_1 """
    def get_expect(dy, data, mean):
        if dy.dtype == "float16":
            dy = dy.astype("float32")
            data = data.astype("float32")
        data_minus_mean = data - mean
        dgamma_red_hw = np.sum(dy * data_minus_mean, axis=(2,3), keepdims=True)
        dbeta_red_hw = np.sum(dy, axis=(2,3), keepdims=True)
        return [dgamma_red_hw, dbeta_red_hw, data_minus_mean]

    shape_nc1c0 = (shape[0], shape[1], 1, 1, shape[4])
    shape_c1c0 = (1, shape[1], 1, 1, shape[4])

    bng1_shapes = [shape, shape, shape_c1c0]
    bng1_dtypes = [dtype, dtype, "float32"]
    
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(fused_batch_norm_grad_split.fused_bn_grad_1,
                            bng1_shapes, bng1_dtypes,
                            kernel_name=kernel_name + "_step1", attrs=attrs, tuning=t)     
        if t:
            inputs = [np.random.rand(*s).astype(t) for (s, t) in zip(bng1_shapes, bng1_dtypes)]
            inputs[2] = np.mean(inputs[1], axis=(0, 2, 3), keepdims=True).astype(bng1_dtypes[2])
            out_shapes = [shape_nc1c0, shape_nc1c0, shape]
            outputs = [np.full(s, np.nan, "float32") for s in out_shapes] 
            expects = get_expect(*inputs)
            return mod, expects, {"args": (*inputs, *outputs), 'outputs': tuple(range(-len(outputs), 0)),
                                 'tuning': False}
        else:
            return mod        
   
    mod = utils.op_build_test(fused_batch_norm_grad_split.fused_bn_grad_1,
                        bng1_shapes, bng1_dtypes,
                        kernel_name=kernel_name + "_step1", attrs=attrs)
    # np.random.seed(0)
    inputs = [np.random.rand(*s).astype(t) for (s, t) in zip(bng1_shapes, bng1_dtypes)]
    inputs[2] = np.mean(inputs[1], axis=(0, 2, 3), keepdims=True).astype(bng1_dtypes[2])
    out_shapes = [shape_nc1c0, shape_nc1c0, shape]
    outputs = [np.full(s, np.nan, "float32") for s in out_shapes]
    outputs = list(utils.mod_launch(mod, (*inputs, *outputs), outputs=tuple(range(-len(outputs), 0)),
                                    expect=get_expect(*inputs)))
    expects = get_expect(*inputs)
    rtol, atol = get_rtol_atol("fused_batch_norm_grad", dtype)
    results = list(map(lambda x, y: np.allclose(x, y, rtol=rtol, atol=atol), outputs, expects))
    print("results", results)
    return inputs, outputs, expects, all(results)

def fused_bn_grad_5D_run_2(shape, dtype, eps, kernel_name, attrs):
    """ test bnGrad_2 """
    def get_expect(dgamma_red_hw, dbeta_red_hw, var, gamma, eps, data_shape):
        m = data_shape[0] * data_shape[2] * data_shape[3]
        neg_m_rec = -1.0 / m
        eps = np.array([eps], dtype=var.dtype).reshape([1] * 5)
        neg_m_rec = np.array([neg_m_rec], dtype=var.dtype).reshape([1] * 5)
        s = (1.0 / np.sqrt(var + eps)).astype(var.dtype)
        dgamma = s * np.sum(dgamma_red_hw, axis=0, keepdims=True)
        dbeta = np.sum(dbeta_red_hw, axis=0, keepdims=True)
        rs = gamma * s
        dgamma_dx = neg_m_rec * rs * s * dgamma
        dbeta_dx = neg_m_rec * rs * dbeta
        return [dgamma, dbeta, rs, dgamma_dx, dbeta_dx]

    shape_nc1c0 = (shape[0], shape[1], 1, 1, shape[4])
    shape_c1c0 = (1, shape[1], 1, 1, shape[4])
    bng2_shapes = [shape_nc1c0, shape_nc1c0, shape_c1c0, shape_c1c0]
    bng2_dtypes = ["float32"] * len(bng2_shapes)
    bng2_opattrs = [eps, shape]
    # np.random.seed(0)
    inputs = [np.random.rand(*s).astype(t) for (s, t) in zip(bng2_shapes, bng2_dtypes)]
    out_shapes = [shape_c1c0, shape_c1c0, shape_c1c0, shape_c1c0, shape_c1c0]
    
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(fused_batch_norm_grad_split.fused_bn_grad_2,
                        bng2_shapes, bng2_dtypes, bng2_opattrs,
                        kernel_name=kernel_name + "_step2", attrs=attrs, tuning=t)    
        if t:
            outputs = [np.full(s, np.nan, "float32") for s in out_shapes]
            expects = get_expect(*inputs, *bng2_opattrs)
            return mod, expects, {"args": (*inputs, *outputs), 'outputs': tuple(range(-len(outputs), 0)),
                                 'tuning': False}
        else:
            return mod     
    mod = utils.op_build_test(fused_batch_norm_grad_split.fused_bn_grad_2,
                    bng2_shapes, bng2_dtypes, bng2_opattrs,
                    kernel_name=kernel_name + "_step2", attrs=attrs)
    outputs = [np.full(s, np.nan, "float32") for s in out_shapes]
    outputs = list(utils.mod_launch(mod, (*inputs, *outputs), outputs=tuple(range(-len(outputs), 0)),
                                    expect=get_expect(*inputs, *bng2_opattrs)))
    expects = get_expect(*inputs, *bng2_opattrs)
    rtol, atol = get_rtol_atol("fused_batch_norm_grad", dtype)
    results = list(map(lambda x, y: np.allclose(x, y, rtol=rtol, atol=atol), outputs, expects))
    print("results", results)
    return inputs, outputs, expects, all(results)

def fused_bn_grad_5D_run_3(shape, dtype, kernel_name, attrs):
    """ test bnGrad_3 """
    def get_expect(dy, rs, dgamma_dx, dbeta_dx, data_minus_mean):
        dx = rs * dy + dbeta_dx + data_minus_mean * dgamma_dx
        return [dx]
    shape_c1c0 = (1, shape[1], 1, 1, shape[4])
    bng3_shapes = [shape, shape_c1c0, shape_c1c0, shape_c1c0, shape]
    bng3_dtypes = [dtype, "float32", "float32", "float32", "float32"]
    
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(fused_batch_norm_grad_split.fused_bn_grad_3,
                        bng3_shapes, bng3_dtypes,
                        kernel_name=kernel_name + "_step3", attrs=attrs, tuning=t)  
        if t:
            inputs = [np.random.rand(*s).astype(t) for (s, t) in zip(bng3_shapes, bng3_dtypes)]
            outputs = np.full(shape, np.nan, dtype)
            expects = get_expect(*inputs)
            return mod, expects, (*inputs, *outputs)
        else:
            return mod   
    # np.random.seed(0)
    inputs = [np.random.rand(*s).astype(t) for (s, t) in zip(bng3_shapes, bng3_dtypes)]
    outputs = np.full(shape, np.nan, dtype)
    mod = utils.op_build_test(fused_batch_norm_grad_split.fused_bn_grad_3,
                    bng3_shapes, bng3_dtypes,
                    kernel_name=kernel_name + "_step3", attrs=attrs)
    outputs = [ utils.mod_launch(mod, (*inputs, outputs), expect=get_expect(*inputs)) ]
    expects = get_expect(*inputs)
    rtol, atol = get_rtol_atol("fused_batch_norm_grad", dtype)
    results = list(map(lambda x, y: np.allclose(x, y, rtol=rtol, atol=atol), outputs, expects))
    print("results", results)
    return inputs, outputs, expects, all(results)

def fused_bn_grad_5D_all_run(shape, dtype, eps, kernel_name, attrs):
    shape_nc1c0 = (shape[0], shape[1], 1, 1, shape[4])
    shape_c1c0 = (1, shape[1], 1, 1, shape[4])

    in_shapes = [shape, shape, shape_c1c0, shape_c1c0, shape_c1c0]
    in_dtypes = [dtype, dtype, "float32", "float32", "float32"]
    # np.random.seed(3)
    inputs = [np.random.rand(*s).astype(t) for (s, t) in zip(in_shapes, in_dtypes)]
    inputs[2] = np.mean(inputs[1].astype("float32"), axis=(0,2,3), keepdims=True)
    inputs[3] = np.var(inputs[1].astype("float32"), axis=(0,2,3), keepdims=True)
    outputs = [None] * 3

    # step 1
    bng1_shapes = [shape, shape, shape_c1c0]
    bng1_dtypes = [dtype, dtype, "float32"]
    mod = utils.op_build_test(fused_batch_norm_grad_split.fused_bn_grad_1,
                        bng1_shapes, bng1_dtypes,
                        kernel_name=kernel_name + "_step1", attrs=attrs.copy())
    bng1_inputs = inputs[:3]
    bng1_outshapes = [shape_nc1c0, shape_nc1c0, shape]
    bng1_outputs = [np.full(s, np.nan, "float32") for s in bng1_outshapes]
    bng1_outputs = list(utils.mod_launch(mod, (*bng1_inputs, *bng1_outputs),
                        outputs=tuple(range(-len(bng1_outputs), 0)), expect=benchmark(*inputs, eps, "NC1HWC0", None)))

    # step 2
    bng2_shapes = [shape_nc1c0, shape_nc1c0, shape_c1c0, shape_c1c0]
    bng2_dtypes = ["float32"] * len(bng2_shapes)
    bng2_opattrs = [eps, shape]
    bng2_inputs = bng1_outputs[:2] + inputs[3:]
    bng2_outshapes = [shape_c1c0, shape_c1c0, shape_c1c0, shape_c1c0, shape_c1c0]
    bng2_outputs = [np.full(s, np.nan, "float32") for s in bng2_outshapes]
    mod = utils.op_build_test(fused_batch_norm_grad_split.fused_bn_grad_2,
                    bng2_shapes, bng2_dtypes, bng2_opattrs,
                    kernel_name=kernel_name + "_step2", attrs=attrs.copy())
    bng2_outputs = list(utils.mod_launch(mod, (*bng2_inputs, *bng2_outputs),
                        outputs=tuple(range(-len(bng2_outputs), 0)), expect=benchmark(*inputs, eps, "NC1HWC0", None)))
    outputs[1:] = bng2_outputs[:2]

    # step 3
    bng3_shapes = [shape, shape_c1c0, shape_c1c0, shape_c1c0, shape]
    bng3_dtypes = [dtype, "float32", "float32", "float32", "float32"]
    bng3_inputs = [inputs[0]] + bng2_outputs[2:] + [bng1_outputs[2]]
    bng3_output = np.full(shape, np.nan, dtype)
    mod = utils.op_build_test(fused_batch_norm_grad_split.fused_bn_grad_3,
                    bng3_shapes, bng3_dtypes,
                    kernel_name=kernel_name + "_step3", attrs=attrs.copy())
    bng3_output = utils.mod_launch(mod, (*bng3_inputs, bng3_output), expect=benchmark(*inputs, eps, "NC1HWC0", None))
    outputs[0] = bng3_output

    expects = benchmark(*inputs, eps, "NC1HWC0", None)
    rtol, atol = get_rtol_atol("fused_batch_norm_grad", dtype)
    results = list(map(lambda x, y: np.allclose(x, y, rtol=rtol, atol=atol), outputs, expects))
    print("results", results)
    return inputs, outputs, expects, all(results)


def fused_batch_norm_grad_run(shape, dtype, eps, data_format, axis, kernel_name, attrs):
    is_special5D = (data_format == "NC1HWC0")

    if is_special5D:
        axes = (0, 2, 3)
        param_shape = [1, shape[1], 1, 1, shape[4]]
    else:
        tmp_axis = axis if axis >= 0 else len(shape) + axis
        axes = tuple([i for i in range(len(shape)) if i != tmp_axis])
        param_shape = [shape[axis]]

    shapes = [shape, shape, param_shape, param_shape, param_shape]
    dtypes = [dtype, dtype, dtype, dtype, dtype]
    op_attrs = [eps, data_format, axis]

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(fused_batch_norm_grad.fused_batch_norm_grad, shapes, dtypes, op_attrs,
                                  kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            data, dy, expects, gamma, mean, outputs, var = gen_data(axes, dtype, eps, data_format,
                                                                    param_shape, shape, axis)
            return mod, expects, {"args": (dy, data, mean, var, gamma, *outputs),
                                  'outputs': tuple(range(-len(outputs), 0)), 'tuning': False}
        else:
            return mod
    else:
        mod = utils.op_build_test(fused_batch_norm_grad.fused_batch_norm_grad, shapes, dtypes, op_attrs,
                                  kernel_name=kernel_name, attrs=attrs)

        data, dy, expects, gamma, mean, outputs, var = gen_data(axes, dtype, eps, data_format, param_shape,
                                                                shape, axis)
        outputs = utils.mod_launch(mod, (dy, data, mean, var, gamma, *outputs), outputs=tuple(range(-len(outputs), 0)),
                                   expect=expects)
        outputs = [outputs] if len(expects) == 1 else list(outputs)

        rtol, atol = get_rtol_atol("fused_batch_norm_grad", dtype)
        results = list(map(lambda x, y: np.allclose(x, y, rtol=rtol, atol=atol), outputs, expects))
        print("results", results)
        return (dy, data, gamma), outputs, expects, all(results)


def gen_data(axes, dtype, eps, data_format, param_shape, shape, axis):
    dy = random_gaussian(shape, miu=1, sigma=0.3).astype(dtype)
    data = random_gaussian(shape, miu=1, sigma=0.3).astype(dtype)
    gamma = random_gaussian(param_shape, miu=1, sigma=0.3).astype(dtype)
    mean = np.mean(data, axis=axes, keepdims=True).astype(dtype)
    var = np.var(data, axis=axes, keepdims=True).astype(dtype)
    expects = benchmark(dy, data, mean, var, gamma, eps, data_format, axis)
    outputs = [np.full(e.shape, np.nan, dtype) for e in expects]
    return data, dy, expects, gamma, mean, outputs, var
