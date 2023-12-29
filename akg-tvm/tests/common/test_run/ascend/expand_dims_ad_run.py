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
from tests.common.test_op.ascend.expand_dims_ad import expand_dims_ad
from tests.common.tensorio import compare_tensor


def expand_dims_ad_run(shape, axis, dtype, attrs):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        input_np = np.random.uniform(low=-1.0, high=1.0, size=shape).astype(dtype)
        forward_output = np.expand_dims(input_np, axis=axis)
        mod = utils.op_build_test(expand_dims_ad, [forward_output.shape, shape], [dtype, dtype],
                                  kernel_name=kernel_name, op_attrs=[axis], attrs=attrs, tuning=t)
        if t:
            expect, head_np, output = gen_data(dtype, forward_output, shape)
            return mod, expect, (head_np, input_np, output)
        else:
            return mod
    else:
        input_np = np.random.uniform(low=-1.0, high=1.0, size=shape).astype(dtype)
        forward_output = np.expand_dims(input_np, axis=axis)
        mod = utils.op_build_test(expand_dims_ad, [forward_output.shape, shape], [dtype, dtype],
                                  kernel_name='expand_dims_ad', op_attrs=[axis], attrs=attrs)
        expect, head_np, output = gen_data(dtype, forward_output, shape)
        output = utils.mod_launch(mod, (head_np, input_np, output), expect=expect)
        return (head_np, input_np, axis), output, expect, compare_tensor(output, expect, atol=0.1)


def gen_data(dtype, forward_output, shape):
    head_np = np.random.uniform(low=-5.0, high=5.0, size=forward_output.shape).astype(dtype)
    expect = np.copy(head_np)
    expect = np.reshape(expect, shape)
    output = np.full(expect.shape, np.nan, dtype)
    return expect, head_np, output
