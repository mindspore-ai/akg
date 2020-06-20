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
from tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from test_op import gelu_ad
from base import get_rtol_atol
from gen_random import random_gaussian

def gelu_grad_data(shape, dtype):
    dy = random_gaussian(shape, miu=1, sigma=0.3).astype(dtype)
    x = random_gaussian(shape, miu=10, sigma=0.3).astype(dtype)
    x_1 = 0.7978845 * x + 0.035677 * np.power(x, 3)
    cdf = 0.5 * (1.0 + np.tanh(x_1))
    y = x * cdf
    tanh_x_1 = np.tanh(x_1)
    res_grad = cdf + x * 0.5 * (1 - np.power(tanh_x_1, 2)) * (0.7978845 + 0.1070322 * np.power(x, 2))
    bench_mark = dy * res_grad

    input_np = x
    head_np = dy
    expect = bench_mark
    output = np.full(expect.shape, np.nan, dtype)

    return input_np, head_np, output, expect


def gelu_ad_run(shape, dtype, attrs):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        if dtype == 'float16' and not utils.product_is_mini():
            mod = utils.op_build_test(gelu_ad.gelu_ad_custom, [shape, shape], [dtype, dtype],
                          kernel_name=kernel_name, attrs=attrs, tuning=t)
        else:
            mod = utils.op_build_test(gelu_ad.gelu_ad, [shape, shape], [dtype, dtype],
                          kernel_name=kernel_name, attrs=attrs, tuning=t)



        if t:
            input_np, head_np, output, expect = gelu_grad_data(shape, dtype)
            return mod, expect, (head_np, input_np, output)
        else:
            return mod
    else:
        if dtype == 'float16' and not utils.product_is_mini():
            mod = utils.op_build_test(gelu_ad.gelu_ad_custom, [shape, shape], [dtype, dtype],
                                      kernel_name="gelu_ad", attrs=attrs)
        else:
            mod = utils.op_build_test(gelu_ad.gelu_ad, [shape, shape], [dtype, dtype],
                          kernel_name="gelu_ad", attrs=attrs)

        input_np, head_np, output, expect = gelu_grad_data(shape, dtype)
        output = utils.mod_launch(mod, (head_np, input_np, output), expect=expect)
        rtol, atol = get_rtol_atol("gelu_ad", dtype)
        return (input_np, head_np), output, expect, compare_tensor(output, expect, rtol=rtol, atol=atol)
