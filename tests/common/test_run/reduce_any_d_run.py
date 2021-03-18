# Copyright 2020 Huawei Technologies Co., Ltd
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

"""reduce_any_d_run"""
import numpy as np

from tests.common.tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from akg.utils.dsl_create import get_reduce_out_shape
from tests.common.test_op import reduce_any_d
from tests.common.base import get_rtol_atol

def reduce_any_d_run(shape, dtype, axis, keepdims, attrs=None):
    """reduce_any_d_run implementation"""
    mod = utils.op_build_test(reduce_any_d.reduce_any_d, [shape], [dtype],
                              kernel_name='reduce_any_d', op_attrs=[axis, keepdims], attrs=attrs)
    args, exp_output, x = gen_data(dtype, shape, axis, keepdims)

    acu_output = utils.mod_launch(mod, args, expect=exp_output)

    rtol, atol = get_rtol_atol("reduce_any_d", dtype)
    testcase_result = compare_tensor(acu_output, exp_output, rtol=rtol, atol=atol, equal_nan=True)

    return x, acu_output, exp_output, testcase_result



def gen_data(dtype, shape, axis, keepdims):
    # generate data for test
    x = np.abs(np.random.uniform(low=-127, high=128, size=tuple(shape)).astype(dtype))
    exp_output = np.amax(x, axis=axis, keepdims=keepdims)
    out_shape = get_reduce_out_shape(shape, axis=axis, keepdims=keepdims)
    # inputs and output to hold the data
    output = np.full(out_shape, np.nan, dtype)
    args = [x, output]
    return args, exp_output, x
