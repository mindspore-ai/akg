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
from tests.common.test_op.ascend import relu6_grad
from tests.common.tensorio import compare_tensor


def relu6_grad_run(shape, dtype, attrs):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(relu6_grad.relu6_grad, [shape, shape], [dtype, dtype], kernel_name=kernel_name,
                                  attrs=attrs, tuning=t)
        if t:
            dy, expect, input_np, output = gen_data(dtype, shape)
            return mod, expect, (dy, input_np, output)
        else:
            return mod
    else:
        dy, expect, input_np, output = gen_data(dtype, shape)
        mod = utils.op_build_test(relu6_grad.relu6_grad, [shape, shape], [dtype, dtype], kernel_name='relu6_grad',
                                  attrs=attrs)
        output = utils.mod_launch(mod, (dy, input_np, output), expect=expect)
        return (dy, input_np), output, expect, compare_tensor(output, expect, atol=0.1)


def gen_data(dtype, shape):
    input_np = np.random.uniform(low=-1.0, high=10.0, size=shape).astype(dtype)
    dy = np.random.uniform(low=-1.0, high=1.0, size=shape).astype(dtype)
    expect = dy * (input_np > 0) * (input_np < 6)
    output = np.full(expect.shape, np.nan, dtype)
    return dy, expect, input_np, output
