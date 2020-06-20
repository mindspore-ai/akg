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
from test_op.softmaxcrossentropywithlogits import softmaxcrossentropywithlogits
from gen_random import random_gaussian

def GenData(shape, dtype):
    """ Generate data for testing the op """
    class_num = shape[1]
    labels_int = np.random.randint(low=0, high=shape[1] - 1, size=shape[0])
    labels = np.eye(class_num)[labels_int].astype(dtype)

    logits = random_gaussian(shape, miu=1, sigma=0.1).astype(dtype)
    testsub = logits - np.max(logits, axis=-1, keepdims=True)
    input_exp = np.exp(testsub)
    softmax = input_exp / np.sum(input_exp, axis=-1, keepdims=True)
    loss_all = labels * np.log(softmax) * -1
    loss = np.sum(loss_all, axis=-1)
    lossNew = np.expand_dims(loss, axis=1)
    grad = (softmax - labels) * lossNew
    return labels, logits, loss, grad


def softmaxcrossentropywithlogits_run(shape, axis, dtype, kernel_name, attrs):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(softmaxcrossentropywithlogits, [shape, shape], [dtype, dtype], op_attrs=[axis],
                                  kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            labels, logits, expect_loss, expect_grad = GenData(shape, dtype)
            res_loss = np.full(shape[0], 0, dtype)
            res_grad = np.full(shape, 0, dtype)
            return mod, (expect_loss, expect_grad), {"args": (labels, logits, res_loss, res_grad), 'outputs': (-2, -1),
                                                     'tuning': False}
        else:
            return mod
    else:
        labels, logits, expect_loss, expect_grad = GenData(shape, dtype)
        # get the result from mod
        res_loss = np.full(shape[0], 0, dtype)
        res_grad = np.full(shape, 0, dtype)
        mod = utils.op_build_test(softmaxcrossentropywithlogits, [shape, shape], [dtype, dtype], op_attrs=[axis],
                                  kernel_name=kernel_name, attrs=attrs)
        res_loss, res_grad = utils.mod_launch(mod, (labels, logits, res_loss, res_grad), (-2, -1), expect=expect_loss)
        cpr_loss = compare_tensor(res_loss, expect_loss, rtol=5e-02, equal_nan=True)
        cpr_grad = compare_tensor(res_grad, expect_grad, rtol=5e-02, equal_nan=True)
        return logits, res_loss, expect_loss, (cpr_loss and cpr_grad)
