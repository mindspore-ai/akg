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
from akg.utils import kernel_exec as utils
from tests.common.test_op import reduce_logsumexp
from tests.common.tensorio import compare_tensor
from tests.common.base import get_rtol_atol
from tests.common.gen_random import random_gaussian
from akg.utils.dsl_create import get_reduce_out_shape

def reduce_logsumexp_run(shape, dtype, axis=None, keepdims=False, kernel_name="reduce_logsumexp", attrs=None):
    op_attrs = [axis, keepdims]
    
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(reduce_logsumexp.reduce_logsumexp, [shape], [dtype],
                                  op_attrs=op_attrs, kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            expect, input, output = gen_data(dtype, shape, axis, keepdims)
            return mod, expect, (input, output)
        else:
            return mod

    else:
        expect, input, output = gen_data(dtype, shape, axis, keepdims)
        mod = utils.op_build_test(reduce_logsumexp.reduce_logsumexp, [shape], [dtype],
                                  op_attrs=op_attrs, kernel_name=kernel_name, attrs=attrs)
        output = utils.mod_launch(mod, (input, output), expect=expect)
        rtol, atol = get_rtol_atol("reduce_logsumexp", dtype)
        return input, output, expect, compare_tensor(output, expect, rtol=rtol, atol=atol, equal_nan=True)


def gen_data(dtype, shape, axis, keepdims):
    input_np = random_gaussian(shape, miu=1, sigma=0.5).astype(dtype)
    exp_input = np.exp(input_np)
    sumexp_input = np.sum(exp_input, axis=axis, keepdims=keepdims)
    logsumexp_input = np.log(sumexp_input)
    out_shape = get_reduce_out_shape(shape, axis=axis, keepdims=keepdims)
    output = np.full(out_shape, np.nan, dtype)
    return logsumexp_input, input_np, output