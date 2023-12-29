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

import numpy as np
from akg.utils import kernel_exec as utils
from tests.common.test_op.ascend.reduce_logsumexp_ad import reduce_logsumexp_ad
from tests.common.tensorio import compare_tensor
from tests.common.base import get_rtol_atol
from tests.common.gen_random import random_gaussian
from akg.utils.dsl_create import get_reduce_out_shape


def reduce_logsumexp_ad_benchmark(input_np, axis, keepdims):
    exp_input = np.exp(input_np)
    sumexp_input = np.sum(exp_input, axis=axis, keepdims=keepdims)
    log_grad = np.reciprocal(sumexp_input)
    reduce_logsumexp_grad = exp_input * np.broadcast_to(log_grad, exp_input.shape)
    return reduce_logsumexp_grad


def reduce_logsumexp_ad_run(shape, dtype, axis, keepdims, kernel_name, attrs):
    op_attrs = [axis, keepdims]
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(reduce_logsumexp_ad, [shape, shape], [dtype, dtype], op_attrs=op_attrs, kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            expect, head_np, input_np, output = gen_data(dtype, shape, axis, keepdims)
            return mod, expect, (head_np, input_np, output)
        else:
            return mod

    else:
        expect, head_np, input_np, output = gen_data(dtype, shape, axis, keepdims)
        mod = utils.op_build_test(reduce_logsumexp_ad, [head_np.shape, shape], [dtype, dtype], op_attrs=op_attrs, kernel_name="reduce_logsumexp", attrs=attrs)
        output = utils.mod_launch(mod, [head_np, input_np, output], expect=expect)
        rtol, atol = get_rtol_atol("reduce_logsumexp", dtype)
        return (head_np, input_np), output, expect, compare_tensor(output, expect, rtol=rtol, atol=atol, equal_nan=True)


def gen_data(dtype, shape, axis, keepdims):
    input_np = random_gaussian(shape, miu=1, sigma=0.5).astype(dtype)
    out_shape = get_reduce_out_shape(shape, axis=axis, keepdims=keepdims)
    head_np = random_gaussian(out_shape, miu=1, sigma=0.5).astype(dtype)

    reduce_logsumexp_grad = reduce_logsumexp_ad_benchmark(input_np, axis, keepdims)
    expect = reduce_logsumexp_grad * head_np
    output = np.full(expect.shape, np.nan, dtype)
    return expect, head_np, input_np, output