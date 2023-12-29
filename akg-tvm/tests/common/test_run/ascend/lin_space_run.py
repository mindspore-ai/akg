# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

"""lin_space_run"""
import numpy as np
from tests.common.tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from tests.common.test_op.ascend import lin_space
from tests.common.base import get_rtol_atol
from tests.common.gen_random import random_gaussian

def lin_space_run(shape_assist, shape_scalar,  dtype_assit, dtype_num, attrs):
    """lin_space_run implementation"""
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(lin_space.lin_space, [shape_assist, shape_scalar, shape_scalar, shape_scalar],
                                  [dtype_assit, dtype_assit, dtype_assit, dtype_num], kernel_name=kernel_name,
                                  op_attrs=[], attrs=attrs, tuning=t)
        if t:
            args, exp_output, input_assist, input_start, input_stop, input_num = \
                gen_data(shape_assist, shape_scalar, dtype_assit, dtype_num)
            return mod, exp_output, args
        else:
            return mod
    else:
        mod = utils.op_build_test(lin_space.lin_space, [shape_assist, shape_scalar, shape_scalar, shape_scalar],
                                  [dtype_assit, dtype_assit, dtype_assit, dtype_num], kernel_name='lin_space',
                                  op_attrs=[], attrs=attrs)
        args, exp_output, input_assist, input_start, input_stop, input_num = \
            gen_data(shape_assist, shape_scalar, dtype_assit, dtype_num)
        acu_output = utils.mod_launch(mod, args, expect=exp_output)
        # compare result
        rtol, atol = get_rtol_atol("lin_space", dtype_assit)
        testcase_result = compare_tensor(acu_output, exp_output, rtol=rtol, atol=atol, equal_nan=True)

        return [input_assist, input_start, input_stop, input_num], acu_output, exp_output, testcase_result



def gen_data(shape_assist, shape_scalar, dtype_assit, dtype_num):
    # generate data
    input_assist = np.linspace(0.0, (shape_assist[0] - 1) * 1.0, shape_assist[0]).astype(dtype_assit)
    input_start = random_gaussian(shape_scalar, miu=10, sigma=0.3).astype(dtype_assit)
    input_stop = random_gaussian(shape_scalar, miu=10, sigma=0.3).astype(dtype_assit)
    input_num = np.array((shape_assist[0], ), dtype=dtype_num)
    exp_output = np.linspace(input_start, input_stop, input_num[0]).transpose()
    # inputs and output to hold the data
    output = np.full(shape_assist, np.nan, dtype_assit)
    args = [input_assist, input_start, input_stop, input_num, output]
    return args, exp_output, input_assist, input_start, input_stop, input_num
