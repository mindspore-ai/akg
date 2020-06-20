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
from test_op import where
from tensorio import compare_tensor


def where_run(shape_con, shape, dtype, attrs):
    condition1 = np.random.uniform(low=-1, high=1, size=shape_con).astype("float16")
    x = np.random.uniform(size=shape).astype("float16")
    shape_con = condition1.shape
    shape = x.shape

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(where.where, [shape_con, shape_con, shape, shape], [dtype] * 4, kernel_name=kernel_name,
                                  attrs=attrs, tuning=t)
        if t:
            args, exp_output, inputs = gen_data(condition1, dtype, shape, shape_con, x)
            return mod, exp_output, args
        else:
            return mod
    else:
        args, exp_output, inputs = gen_data(condition1, dtype, shape, shape_con, x)
        # compute acu_output
        mod = utils.op_build_test(where.where, [shape_con, shape_con, shape, shape], [dtype] * 4, kernel_name='where',
                                  attrs=attrs)
        acu_output = utils.mod_launch(mod, args, expect=exp_output)

        # compare result
        TestCase_Result = compare_tensor(acu_output, exp_output, rtol=5e-03, equal_nan=True)
        return inputs, acu_output, exp_output, TestCase_Result


def gen_data(condition1, dtype, shape, shape_con, x):
    condition2 = np.zeros(shape_con).astype("float16")
    y = np.random.uniform(size=shape).astype("float16")
    # Result_Numpy
    shape_con_len = shape_con[0]
    shape_len = 1
    for len in shape:
        shape_len = shape_len * len
    if shape == shape_con:
        tmp_condition1 = condition1.reshape(shape_len)
    else:
        tmp_condition1 = condition1.repeat(shape_len // shape_con_len)
    if shape == shape_con:
        tmp_condition2 = condition2.reshape(shape_len)
    else:
        tmp_condition2 = condition2.repeat(shape_len // shape_con_len)
    tmp_x = x.reshape(shape_len)
    tmp_y = y.reshape(shape_len)
    tmp_np_out = np.array(
        [xv if c1 >= c2 else yv for (c1, c2, xv, yv) in zip(tmp_condition1, tmp_condition2, tmp_x, tmp_y)])
    exp_output = tmp_np_out.reshape(shape)
    # inputs and output to hold the data
    output = np.full(exp_output.shape, np.nan, dtype)
    inputs = [condition1, condition2, x, y]
    args = [condition1, condition2, x, y, output]
    return args, exp_output, inputs
