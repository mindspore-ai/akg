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

"""clip_run"""
import numpy as np
from akg.utils import kernel_exec as utils
from tests.common.test_op import clip
from tests.common.tensorio import compare_tensor
from tests.common.gen_random import random_gaussian

def clip_run(shape, min_val, max_val, dtype, attrs):
    """clip_run"""
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(clip.clip, [shape], [dtype], kernel_name=kernel_name,
                                  op_attrs=[min_val, max_val], attrs=attrs, tuning=t)
        if t:
            exp_output, inputs, output = gen_data(dtype, max_val, min_val, shape)
            return mod, exp_output, (inputs, output)
        return mod
    else:
        # op_attrs=[shape, dtype]
        mod = utils.op_build_test(clip.clip, [shape], [dtype], kernel_name='clip',
                                  op_attrs=[min_val, max_val], attrs=attrs)
        exp_output, inputs, output = gen_data(dtype, max_val, min_val, shape)
        # result_tvm
        acu_output = utils.mod_launch(mod, (inputs, output), expect=exp_output)
        # compare result
        compare_result = compare_tensor(acu_output, exp_output, rtol=5e-03, equal_nan=True)

        return inputs, acu_output, exp_output, compare_result


def gen_data(dtype, max_val, min_val, shape):
    # Result_Numpy
    inputs = random_gaussian(shape, miu=1, sigma=10.0).astype(dtype)
    exp_output = np.clip(inputs, min_val, max_val)
    # inputs and output to hold the data
    output = np.full(shape, np.nan, dtype)
    return exp_output, inputs, output
