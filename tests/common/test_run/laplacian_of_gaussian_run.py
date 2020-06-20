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
from tensorio import compare_tensor
import numpy as np
from test_op.laplacian_of_gaussian import laplacian_of_gaussian_ad
from base import get_rtol_atol

def laplacian_of_gaussian_benchmark(head, x, sig=1.0, mean=0.0):
    """mannual laplacian of gaussian filter."""
    if x.dtype == 'float32':
        sig_cast = np.float32(sig)
        mean_cast = np.float32(mean)
    elif x.dtype == 'float16':
        sig_cast = np.float16(sig)
        mean_cast = np.float16(mean)

    if (len(x.shape) == 1):
        x_hat = x * x
        sig_square = sig_cast * sig_cast
        return head * np.exp(-x_hat / (2.0 * sig_square)) * (x_hat/sig_square-1) / (6.283 * (np.power(sig_cast, 4)))
    elif (len(x.shape) == 2):
        hadamard = np.multiply(x, x)
        sig_square = sig_cast * sig_cast
        temp = np.sum(-hadamard / (2 * sig_square), axis=(1), keepdims=True)
        exp_t = np.exp(temp)
        exp_bc = np.broadcast_to(exp_t*head, x.shape)
        derivative_1st = np.multiply(exp_bc, -x)
        sum_reduce = np.sum(-x, axis=(1), keepdims=True)
        res1 = np.multiply(derivative_1st, sum_reduce/np.power(sig_cast, 2))
        res  = res1 - exp_bc
        return res / (6.283 * (np.power(sig_cast, 4)))
    else:
        raise RuntimeError("Do not support {0} dim laplacian of gaussian.".format(len(x.shape)))

def gen_data(dtype, shape):
    input_np = np.random.uniform(low=0.0, high=1.0, size=shape).astype(dtype)
    if len(shape) == 2:
        head_shape = [shape[0], 1]
    elif len(shape) == 1:
        head_shape = [shape[0]]
    else:
        raise RuntimeError("Do not support {0} dim laplacian of gaussian.".format(len(shape)))

    head_np = np.random.uniform(low=0.0, high=1.0, size=head_shape).astype(dtype)
    expect = laplacian_of_gaussian_benchmark(head_np, input_np)
    return expect, head_np, input_np


def laplacian_of_gaussian_ad_run(shape, dtype, attrs):
    expect, head_np, input_np = gen_data(dtype, shape)
    mod = utils.op_build_test(laplacian_of_gaussian_ad, [head_np.shape, shape], [dtype, dtype], kernel_name='mulexp', attrs=attrs)
    output = np.full(expect.shape, np.nan, dtype)
    output = utils.mod_launch(mod, (head_np, input_np, output), expect=expect)
    rtol, atol = get_rtol_atol("laplacian_of_gaussian", dtype)
    return (head_np, input_np), output, expect, compare_tensor(output, expect, rtol=rtol, atol = atol)
