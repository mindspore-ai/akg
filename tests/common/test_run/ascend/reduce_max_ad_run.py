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

from akg.utils import kernel_exec as utils
import numpy as np
from tests.common.gen_random import random_gaussian
from tests.common.test_op.ascend import reduce_max_ad
from tests.common.test_op.ascend.reduce_max_ad import reduce_max_ad
from tests.common.test_op.ascend.reduce_max_ad import reduce_max_ad_optimized
from tests.common.test_op.ascend.reduce_max_ad import reduce_max_ad_optimized_manual_schedule
from tests.common.tensorio import compare_tensor


def reduce_max_ad_run(shape, axis, keepdims, dtype, optimized, kernel_name="reduce_max", attrs_op={}, polyhedral=True, attrs={}):
    attrs.update(attrs_op)
    op_attrs = [axis, keepdims]

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        head_np, input = gen_input_data(axis, dtype, keepdims, shape)
        if polyhedral:
            if optimized:
                mod = utils.op_build_test(reduce_max_ad_optimized, [head_np.shape, shape], [dtype, dtype],
                                          op_attrs=op_attrs, kernel_name=kernel_name, attrs=attrs,
                                          polyhedral=polyhedral, tuning=t)
            else:
                mod = utils.op_build_test(reduce_max_ad, [head_np.shape, shape], [dtype, dtype], op_attrs=op_attrs,
                                          kernel_name=kernel_name, attrs=attrs, polyhedral=polyhedral, tuning=t)
        else:
            mod = reduce_max_ad_optimized_manual_schedule(shape, dtype, axis, keepdims, polyhedral=polyhedral, attrs=attrs,)
        if t:
            expect = (input == np.amax(input, axis=axis, keepdims=True)) * input
            output = np.full(expect.shape, np.nan, dtype)
            return mod, expect, (head_np, input, output)
        else:
            return mod
    else:
        head_np, input = gen_input_data(axis, dtype, keepdims, shape)
        if polyhedral:
            if optimized:
                mod = utils.op_build_test(reduce_max_ad_optimized, [head_np.shape, shape], [dtype, dtype],
                                          op_attrs=op_attrs, kernel_name='reduce_max_ad_opt', attrs=attrs,
                                          polyhedral=polyhedral)
            else:
                mod = utils.op_build_test(reduce_max_ad, [head_np.shape, shape], [dtype, dtype], op_attrs=op_attrs,
                                          kernel_name='reduce_max_ad', attrs=attrs, polyhedral=polyhedral)
        else:
            mod = reduce_max_ad_optimized_manual_schedule(shape, dtype, axis, keepdims, polyhedral=polyhedral, attrs=attrs)
        expect = (input == np.amax(input, axis=axis, keepdims=True)) * input
        output = np.full(expect.shape, np.nan, dtype)
        output = utils.mod_launch(mod, (head_np, input, output), expect=expect)

        return (input, head_np), output, expect, compare_tensor(output, expect, atol=0.1)


def gen_input_data(axis, dtype, keepdims, shape):
    input = random_gaussian(shape, miu=0, sigma=0.5).astype(dtype.lower())
    forward = np.amax(input, axis=axis, keepdims=keepdims)
    head_np = forward
    return head_np, input
