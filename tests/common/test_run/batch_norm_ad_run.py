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

import numpy as np
from tests.common.tensorio import compare_tensor
from tests.common.base import get_rtol_atol
from akg.utils import kernel_exec as utils
from akg.ops.nn.batch_norm_ad import batch_norm_ad
from . import fused_batch_norm_grad_run
from tests.common.gen_random import random_gaussian

def batch_norm_ad_run(shape, dtype, eps, kernel_name, attrs):
    support_list = {"float16": np.float16, "float32": np.float32}
    if len(shape) == 5:
        data_format = "NC1HWC0"
        (n, c, h, w, c0) = shape
        axis = None
        param_shape = (1, c, 1, 1, c0)
        mean_shape = (1, c, 1, 1, c0)
    else:
        data_format = "NCHW"
        (n, c, h, w) = shape
        c0 = None
        axis = 1
        param_shape = (c,)
        mean_shape = (c,)

    if attrs is None:
        attrs = {"enable_bk_optimize": False}
    else:
        attrs['enable_bk_optimize'] = False

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)

        mod = utils.op_build_test(batch_norm_ad, [shape, shape, mean_shape, mean_shape, param_shape],
                                  [dtype, dtype, dtype, dtype, dtype], kernel_name=kernel_name,
                                  op_attrs=[data_format, axis, eps], attrs=attrs, tuning=t)

        if t:
            np_data, head, expects, np_gamma, np_beta, np_mean, outputs, np_var = gen_data(dtype, eps, data_format,
                                                                                           param_shape, shape, axis)

            (expect_data, expect_gamma, expect_beta) = expects
            (output, output_gamma, output_beta) = outputs
            return mod, (expect_data, expect_gamma, expect_beta), {
                "args": (head, np_data, np_mean, np_var, np_gamma, output, output_gamma, output_beta),
                "outputs": tuple(range(-len(outputs), 0)), 'tuning': False}
        else:
            return mod
    else:
        mod = utils.op_build_test(batch_norm_ad, [shape, shape, mean_shape, mean_shape, param_shape],
                                  [dtype, dtype, dtype, dtype, dtype], kernel_name="batch_norm_ad",
                                  op_attrs=[data_format, axis, eps], attrs=attrs)

        np_data, head, expects, np_gamma, np_beta, np_mean, outputs, np_var = gen_data(dtype, eps, data_format,
                                                                                       param_shape, shape, axis)
        expect_data, expect_gamma, expect_beta = [o.astype(dtype) for o in expects]

        (output, output_gamma, output_beta) = outputs
        outputs = [o.astype(dtype) for o in outputs]

        outputs = utils.mod_launch(mod, [head, np_data, np_mean, np_var, np_gamma, *outputs],
                                   outputs=tuple(range(-len(outputs), 0)), expect=expects)

        outputs = [outputs] if len(expects) == 1 else list(outputs)
        output, output_gamma, output_beta = outputs

        rtol, atol = get_rtol_atol("batch_norm_ad", dtype)

        assert_res = True
        assert_res &= compare_tensor(output, expect_data, rtol=rtol, atol=atol,
                                     equal_nan=True)
        assert_res &= compare_tensor(output_gamma, expect_gamma, rtol=rtol, atol=atol,
                                     equal_nan=True)
        assert_res &= compare_tensor(output_beta, expect_beta, rtol=rtol, atol=atol,
                                     equal_nan=True)
        return [head, np_data, np_mean, np_var, np_gamma, np_beta], (output, output_gamma, output_beta), (
        expect_data, expect_gamma, expect_beta), assert_res


def gen_data(dtype, eps, data_format, param_shape, shape, axis):
    axes = (0, 2, 3)
    if data_format == "NC1HWC0":
        keepdims = True
    else:
        keepdims = False

    miu = 0.5
    sigma = 0.03
    dy = random_gaussian(shape, miu=miu, sigma=sigma).astype(dtype)
    data = random_gaussian(shape, miu=miu, sigma=sigma).astype(dtype)
    gamma = random_gaussian(param_shape, miu=miu, sigma=sigma).astype(dtype)
    beta = random_gaussian(param_shape, miu=miu, sigma=sigma).astype(dtype)
    mean = np.mean(data, axis=axes, keepdims=keepdims).astype(dtype)
    var = np.var(data, axis=axes, keepdims=keepdims).astype(dtype)
    expects = fused_batch_norm_grad_run.benchmark(dy, data, mean, var, gamma, eps, data_format, axis)
    outputs = [np.full(e.shape, np.nan, e.dtype) for e in expects]
    return data, dy, expects, gamma, beta, mean, outputs, var


def get_input_data(c, c0, dtype, eps, shape, support_list):
    if c0 is None:
        chan_shape = (1, c, 1, 1)
    else:
        chan_shape = (1, c, 1, 1, c0)

    axes = (0, 2, 3)

    miu = 0.5
    sigma = 0.03
    np_data = random_gaussian(shape, miu=miu, sigma=sigma).astype(support_list[dtype])
    np_mean = np.mean(np_data, axis=axes, keepdims=True).astype(dtype)
    np_var = np.var(np_data, axis=axes, keepdims=True).astype(dtype)

    np_gamma = random_gaussian(chan_shape, miu=miu, sigma=sigma).astype(support_list[dtype])
    np_beta = random_gaussian(chan_shape, miu=miu, sigma=sigma).astype(support_list[dtype])
    np_var = np.abs(np_var)

    mean_bc = np.broadcast_to(np_mean, shape)
    var_bc = np.broadcast_to(np_var, shape)
    gamma_bc = np.broadcast_to(np_gamma, shape)
    beta_bc = np.broadcast_to(np_beta, shape)
    rsqvar2 = (1.0 / np.sqrt(var_bc + eps)).astype(var_bc.dtype)

    rsqvar = np.exp(np.log(var_bc + eps, dtype="float32") * -0.5, dtype="float32")
    normalize_data = (np_data - mean_bc) * rsqvar
    y = gamma_bc * normalize_data + beta_bc

    return gamma_bc, np_beta, np_data, np_gamma, np_mean, np_var, var_bc, normalize_data, y
