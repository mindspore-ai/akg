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

"""axpy_run"""
import numpy as np
from tests.common.tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from akg.ops.math.ascend import Axpy
from tests.common.gen_random import random_gaussian

def axpy_run(shape1, shape2, alpha, dtype, attrs):
    """axpy_run implementation"""
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(Axpy, [shape1, shape2], [dtype, dtype], kernel_name=kernel_name,
                                  op_attrs=[alpha], attrs=attrs, tuning=t)
        if t:
            args, exp_output, inputs1, inputs2 = gen_data(alpha, dtype, shape1, shape2)
            return mod, exp_output, args
        else:
            return mod
    else:
        mod = utils.op_build_test(Axpy, [shape1, shape2], [dtype, dtype], kernel_name='axpy', op_attrs=[alpha],
                                  attrs=attrs)
        args, exp_output, inputs1, inputs2 = gen_data(alpha, dtype, shape1, shape2)
        # result_tvm
        acu_output = utils.mod_launch(mod, args, expect=exp_output)
        # compare result
        testcase_result = compare_tensor(acu_output, exp_output, rtol=5e-03, atol=5e-03, equal_nan=True)

        return [inputs1, inputs2], acu_output, exp_output, testcase_result
        # return [inputs1,inputs2],acu_output,exp_output,False


def gen_data(alpha, dtype, shape1, shape2):
    # result_numpy
    inputs1 = random_gaussian(shape1, miu=1, sigma=10.0).astype(dtype)
    inputs2 = random_gaussian(shape2, miu=1, sigma=10.0).astype(dtype)
    exp_output = np.add(np.dot(inputs1, alpha), inputs2)
    # inputs and output to hold the data
    output = np.full(shape1, np.nan, dtype)
    args = [inputs1, inputs2, output]
    return args, exp_output, inputs1, inputs2
