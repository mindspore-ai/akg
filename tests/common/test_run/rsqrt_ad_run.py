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

from tensorio import compare_tensor
from akg.utils import kernel_exec as utils
import numpy as np
from test_op.rsqrt_ad import rsqrt_ad


def rsqrt_ad_run(shape, dtype, attrs):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(rsqrt_ad, [shape, shape], [dtype, dtype], kernel_name=kernel_name, attrs=attrs,
                                  tuning=t)
        if t:
            expect, head_np, input_np, output = gen_data(dtype, shape)
            return mod, expect, (head_np, input_np, output)
        else:
            return mod
    else:
        expect, head_np, input_np, output = gen_data(dtype, shape)
        mod = utils.op_build_test(rsqrt_ad, [shape, shape], [dtype, dtype], kernel_name='rsqrt_ad', attrs=attrs)
        output = utils.mod_launch(mod, (head_np, input_np, output), expect=expect)
        return (head_np, input_np), output, expect, compare_tensor(output, expect, rtol=5e-02, atol=0.1)


def gen_data(dtype, shape):
    input_np = np.random.uniform(low=0.05, high=1.0, size=shape).astype(dtype)
    head_np = np.random.uniform(low=0.05, high=1.0, size=shape).astype(dtype)
    expect = ((-0.5) * np.power(input_np, -1.5)) * head_np
    output = np.full(expect.shape, np.nan, dtype)
    return expect, head_np, input_np, output
