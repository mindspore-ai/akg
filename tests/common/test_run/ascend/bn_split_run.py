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

"""run function: splited bn"""
import time

import numpy as np
from tests.common.tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from akg.ops.nn.ascend import FusedBn1, FusedBn2, FusedBn3
from tests.common.base import get_rtol_atol
from tests.common.gen_random import random_gaussian

def bn_benchmark(data, gamma, beta, running_mean, running_var,
                 momentum, eps, part_num: int = 0):
    """Benchmark function for bn split."""
    mean = np.mean(data.astype(np.float64), axis=(0, 2, 3),
                   keepdims=True).astype(np.float32)
    if part_num == 1:
        var_part = np.mean(np.power(data, 2).astype(np.float64),
                           axis=(0, 2, 3), keepdims=True).astype(np.float32)
        return (mean, var_part)

    var = np.var(data.astype(np.float64), axis=(0, 2, 3),
                 keepdims=True).astype(np.float32)
    mean_new = momentum * running_mean + (1 - momentum) * mean
    var_new = momentum * running_var + (1 - momentum) * var

    if part_num == 2:
        return (var, mean_new, var_new)

    rsd = (1.0 / np.sqrt(var + eps)).astype("float32")
    hat_gamma = gamma * rsd
    hat_beta = beta - gamma * mean * rsd

    data_cast = data.astype("float32")

    res = (hat_gamma * data_cast + hat_beta).astype(data.dtype)
    if part_num == 3:
        return (res,)

    if part_num == 0:
        # Whole BN outputs
        return (res, mean_new, var_new, mean, var)

    return None

def get_compile_param(shape, dtype, part_num: int = 1):
    """get parameters for compiling module"""
    assert 0 <= part_num <= 3, \
        "parameter part_num({}) is invalid".format(part_num)

    mid_shape = (1, shape[1], 1, 1, shape[4])
    mid_dtype = "float32"

    if part_num == 1:
        in_shapes = [shape]
        in_dtypes = [dtype]
    elif part_num == 2:
        in_shapes = [mid_shape] * 4
        in_dtypes = [mid_dtype] * 4
    elif part_num == 3:
        in_shapes = [shape] + [mid_shape] * 4
        in_dtypes = [dtype] + [mid_dtype] * 4
    else:
        in_shapes = []
        in_dtypes = []

    return in_shapes, in_dtypes

def malloc_out_buffer(expects, full_value=0):
    """malloc buffer by expects for launch"""
    return tuple([np.full(e.shape, full_value, e.dtype) for e in expects])

def gen_data(shape, dtype, momentum, eps, part_num=0):
    """Generate datas.

    Generate input datas, calculate expect results,
    and generate output_buffers for splited fused batch norm.

    Args:
        shape: Shape of data that will be normalized.
        dtype: Data's type.
        momentum: Momentum for moving average.
        eps: A small value for avoiding divide zero.

    Returns:
        inputs: A tuple contain all generated input data.
        output_buffers: A tuple contain all generated output buffer.
        expects: A tuple contain expect results.
    """

    mid_shape = (1, shape[1], 1, 1, shape[4])
    mid_dtype = "float32"

    seed_tmp = int(time.time())
    data = random_gaussian(shape, miu=1, sigma=0.3,
                           seed=seed_tmp).astype(dtype)

    inputs1 = (data,)
    expects1 = bn_benchmark(data, None, None, None, None, momentum, eps, 1)
    out_buffer1 = malloc_out_buffer(expects1)

    if part_num == 1:
        return inputs1, out_buffer1, expects1

    running_mean = random_gaussian(mid_shape, miu=1, sigma=0.3,
                                         seed=seed_tmp + 3).astype(mid_dtype)
    running_var = abs(random_gaussian(mid_shape, miu=1, sigma=0.3,
                                            seed=seed_tmp + 4)).astype(mid_dtype)

    inputs2 = (*expects1, running_mean, running_var)
    expects2 = bn_benchmark(data, None, None, running_mean, running_var,
                            momentum, eps, 2)
    out_buffer2 = malloc_out_buffer(expects2)
    if part_num == 2:
        return inputs2, out_buffer2, expects2

    gamma = random_gaussian(mid_shape, miu=1, sigma=0.3,
                            seed=seed_tmp + 1).astype(mid_dtype)
    beta = random_gaussian(mid_shape, miu=1, sigma=0.3,
                           seed=seed_tmp + 2).astype(mid_dtype)

    inputs3 = (data, expects1[0], expects2[0], gamma, beta)
    expects3 = bn_benchmark(data, gamma, beta, running_mean, running_var,
                            momentum, eps, 3)
    out_buffer3 = malloc_out_buffer(expects3)

    if part_num == 3:
        return inputs3, out_buffer3, expects3

    # Whole BN inputs
    inputs = [inputs1, inputs2, inputs3]
    inputs.append((data, gamma, beta, running_mean, running_var))
    out_buffers = [out_buffer1, out_buffer2, out_buffer3]

    expects = (expects3[0], expects2[1], expects2[2], expects1[0], expects2[0])
    print("INFO data seed: ", seed_tmp)

    return inputs, out_buffers, expects

def bn_1_run(shape, dtype, momentum, eps, kernel_name, attrs):
    """Test run function for first part of splited bn"""
    in_shapes, in_dtypes = get_compile_param(shape, dtype, 1)

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)

        mod = utils.op_build_test(FusedBn1,
                                  in_shapes, in_dtypes,
                                  kernel_name=kernel_name,
                                  attrs=attrs, tuning=t)
        if t:
            inputs, output_buffers, expects = gen_data(shape, dtype,
                                                       momentum, eps, 1)
            output_places = list(range(-len(output_buffers), 0))
            return mod, expects, {
                "args": (*inputs, *output_buffers),
                'outputs': output_places,
                'tuning': False}
        return mod

    mod_1 = utils.op_build_test(FusedBn1,
                                in_shapes, in_dtypes,
                                kernel_name="fusedbn1_"+kernel_name,
                                attrs=attrs)

    inputs, output_buffers, expects = gen_data(shape, dtype, momentum, eps, 1)
    output_places1 = list(range(-len(output_buffers), 0))
    res_1 = utils.mod_launch(mod_1, [*inputs, *output_buffers],
                             outputs=output_places1, expect=expects)

    rtol, atol = get_rtol_atol("bn_split", dtype)
    cmp_res = list(map(lambda x, y:
                       compare_tensor(x, y, rtol=rtol, atol=atol),
                       res_1, expects))

    return inputs, res_1, expects, all(cmp_res)

def bn_2_run(shape, dtype, momentum, eps, kernel_name, attrs):
    """Test run function for second part of splited bn"""
    in_shapes, in_dtypes = get_compile_param(shape, dtype, 2)

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(FusedBn2,
                                  in_shapes, in_dtypes,
                                  op_attrs=[momentum],
                                  kernel_name=kernel_name,
                                  attrs=attrs, tuning=t)
        if t:
            inputs, output_buffers, expects = gen_data(shape, dtype, momentum, eps, 2)
            inplace_binds = ((2, 1), (3, 2))
            output_places2 = list(range(-len(output_buffers), 0))
            if inplace_binds is not None:
                for bind in inplace_binds:
                    output_places2[bind[1]] = bind[0]
            return mod, expects, {
                "args": (*inputs, *output_buffers),
                'outputs': output_places2,
                'tuning': False}
        return mod

    mod_2 = utils.op_build_test(FusedBn2,
                                in_shapes, in_dtypes,
                                op_attrs=[momentum],
                                kernel_name="fusedbn2_"+kernel_name,
                                attrs=attrs)

    inputs, output_buffers, expects = gen_data(shape, dtype, momentum, eps, 2)
    inplace_binds = ((2, 1), (3, 2))
    output_places2 = list(range(-len(output_buffers), 0))
    if inplace_binds is not None:
        for bind in inplace_binds:
            output_places2[bind[1]] = bind[0]
    res_2 = utils.mod_launch(mod_2, [*inputs, *output_buffers],
                             outputs=output_places2, expect=expects)

    rtol, atol = get_rtol_atol("bn_split", dtype)
    cmp_res = list(map(lambda x, y:
                       compare_tensor(x, y, rtol=rtol, atol=atol),
                       res_2, expects))
    return inputs, res_2, expects, all(cmp_res)

def bn_3_run(shape, dtype, momentum, eps, kernel_name, attrs):
    """Test run function for third part of splited bn"""
    in_shapes, in_dtypes = get_compile_param(shape, dtype, 3)

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(FusedBn3,
                                  in_shapes, in_dtypes,
                                  op_attrs=[eps],
                                  kernel_name=kernel_name,
                                  attrs=attrs, tuning=t)
        if t:
            inputs, output_buffers, expects = gen_data(shape, dtype,
                                                       momentum, eps, 3)
            output_places3 = list(range(-len(output_buffers), 0))
            return mod, expects[0], (*inputs, *output_buffers)

        return mod

    mod_3 = utils.op_build_test(FusedBn3,
                                in_shapes, in_dtypes,
                                op_attrs=[eps],
                                kernel_name="fusedbn3_"+kernel_name,
                                attrs=attrs)

    inputs, output_buffers, expects = gen_data(shape, dtype, momentum, eps, 3)
    output_places3 = list(range(-len(output_buffers), 0))
    res_3 = utils.mod_launch(mod_3, [*inputs, *output_buffers],
                             outputs=output_places3, expect=expects)

    if not isinstance(res_3, tuple):
        res_3 = (res_3,)

    rtol, atol = get_rtol_atol("bn_split", dtype)
    cmp_res = list(map(lambda x, y:
                       compare_tensor(x, y, rtol=rtol, atol=atol),
                       res_3, expects))
    return inputs, res_3, expects, all(cmp_res)

def bn_split_run(shape, dtype, momentum, eps, kernel_name, attrs):
    """Test run function for whole splited bn"""
    in_shapes1, in_dtypes1 = get_compile_param(shape, dtype, 1)
    mod_1 = utils.op_build_test(FusedBn1,
                                in_shapes1, in_dtypes1,
                                kernel_name="fused_bn1_"+kernel_name,
                                attrs=attrs.copy())


    in_shapes2, in_dtypes2 = get_compile_param(shape, dtype, 2)
    mod_2 = utils.op_build_test(FusedBn2,
                                in_shapes2, in_dtypes2,
                                op_attrs=[momentum],
                                kernel_name="fused_bn2_"+kernel_name,
                                attrs=attrs.copy())

    in_shapes3, in_dtypes3 = get_compile_param(shape, dtype, 3)
    mod_3 = utils.op_build_test(FusedBn3,
                                in_shapes3, in_dtypes3,
                                op_attrs=[eps],
                                kernel_name="fused_bn3_"+kernel_name,
                                attrs=attrs.copy())

    inputs, output_buffers, expects = gen_data(shape, dtype, momentum, eps, 0)
    output_places1 = list(range(-len(output_buffers[0]), 0))
    res_1_tmp = utils.mod_launch(mod_1, [inputs[-1][0], *output_buffers[0]],
                                 outputs=output_places1, expect=expects)

    inplace_binds = ((2, 1), (3, 2))
    output_places2 = list(range(-len(output_buffers[1]), 0))
    if inplace_binds is not None:
        for bind in inplace_binds:
            output_places2[bind[1]] = bind[0]
    res_2_tmp = utils.mod_launch(mod_2,
                                 [res_1_tmp[0], res_1_tmp[1],
                                  inputs[-1][3], inputs[-1][4],
                                  *output_buffers[1]],
                                 outputs=output_places2)

    output_places3 = list(range(-len(output_buffers[2]), 0))
    res_3_tmp = utils.mod_launch(mod_3,
                                 [inputs[-1][0], res_1_tmp[0], res_2_tmp[0],
                                  inputs[-1][1], inputs[-1][2],
                                  *output_buffers[2]],
                                 outputs=output_places3, expect=expects)

    results = (res_3_tmp, res_2_tmp[1], res_2_tmp[2],
               res_1_tmp[0], res_2_tmp[0])

    rtol, atol = get_rtol_atol("bn_split", dtype)
    cmp_res = list(map(lambda x, y:
                       compare_tensor(x, y, rtol=rtol, atol=atol),
                       results, expects))

    return inputs[-1], results, expects, all(cmp_res)
