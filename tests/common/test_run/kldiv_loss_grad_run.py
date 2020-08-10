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
from tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from test_op import kldiv_loss_grad
from gen_random import random_gaussian


def kldiv_loss_grad_run(shape, dtype, kernel_name="kldiv_loss_grad", attrs=None):
    if not utils.product_is_mini():
        attrs['enable_align_fix'] = True
        attrs['enable_multicore'] = True

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(kldiv_loss_grad.kldiv_loss_grad, [shape, shape, shape], [dtype, dtype, dtype],
                                  kernel_name=kernel_name, attrs=attrs, dump_code=True, tuning=t)
        if t:
            cur_deriv, output, pre_deriv, prediction, target = gen_data(attrs, dtype, shape)
            return mod, cur_deriv, (pre_deriv, prediction, target, output)
        else:
            return mod
    else:
        mod = utils.op_build_test(kldiv_loss_grad.kldiv_loss_grad, [shape, shape, shape], [dtype, dtype, dtype],
                                  kernel_name=kernel_name, attrs=attrs, dump_code=True)
        cur_deriv, output, pre_deriv, prediction, target = gen_data(attrs, dtype, shape)
        output = utils.mod_launch(mod, (pre_deriv, prediction, target, output), expect=cur_deriv)
        return (pre_deriv, prediction, target), output, cur_deriv, compare_tensor(output, cur_deriv, rtol=0.005,
                                                                                  atol=0.001)


def gen_data(attrs, dtype, shape):
    # support_list = {"float16": np.float16, "float32": np.float32}
    target = random_gaussian(shape, miu=0, sigma=1).astype(dtype)
    target = np.abs(target)
    prediction = random_gaussian(shape, miu=0, sigma=1).astype(dtype)
    prediction = np.abs(prediction)
    # off_set = np.full(prediction.shape, 0.05, dtype)
    off_set = np.full(prediction.shape, 2, dtype)
    prediction = np.add(prediction, off_set)
    target = np.add(target, off_set)
    pre_deriv = random_gaussian(shape, miu=0, sigma=1).astype(dtype)
    cur_deriv = np.divide(target, prediction)
    cur_deriv = np.multiply(cur_deriv, pre_deriv)
    output = np.full(prediction.shape, np.nan, dtype)
    return cur_deriv, output, pre_deriv, prediction, target
