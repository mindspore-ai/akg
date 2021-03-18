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
from tests.common.tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from tests.common.test_op import l1_loss_grad
from tests.common.gen_random import random_gaussian


def l1_loss_grad_run(shape, dtype, kernel_name="l1_loss_grad", attrs=None):
    if not utils.product_is_mini():
        attrs['enable_align_fix'] = True
        attrs['enable_multicore'] = True

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(l1_loss_grad.l1_loss_grad, [shape, shape, shape], [dtype, dtype, dtype],
                                  kernel_name=kernel_name, attrs=attrs, dump_code=True, tuning=t)
        if t:
            dloss, expect, output, prediction, target = gen_data(dtype, shape)
            return mod, expect, (dloss, prediction, target, output)
        else:
            return mod
    else:
        mod = utils.op_build_test(l1_loss_grad.l1_loss_grad, [shape, shape, shape], [dtype, dtype, dtype],
                                  kernel_name=kernel_name, attrs=attrs, dump_code=True)
        dloss, expect, output, prediction, target = gen_data(dtype, shape)
        output = utils.mod_launch(mod, (dloss, prediction, target, output), expect=expect)
        return (dloss, prediction, target), output, expect, compare_tensor(output, expect, rtol=0.001, atol=0.001)


def gen_data(dtype, shape):
    target = random_gaussian(shape, miu=0, sigma=5).astype(dtype)
    diff = random_gaussian(shape, miu=0, sigma=1).astype(dtype)
    prediction = np.add(target, diff)
    dloss = random_gaussian(shape, miu=0, sigma=2).astype(dtype)
    # sigma is a constant parameter
    sigma = 1.0
    diff = np.subtract(prediction, target)
    second_branch = np.where(0 <= diff, 1, -1)
    expect = np.multiply(second_branch, dloss)
    expect = expect.astype(np.float16)
    output = np.full(expect.shape, np.nan, dtype)
    return dloss, expect, output, prediction, target
