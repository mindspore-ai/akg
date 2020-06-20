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
from test_op.crossentropyloss_ad import crossentropyloss_ad


def GenData(shape, dtype):
    """Generate data for testing the op."""
    class_num = shape[1]
    labels_int = np.random.randint(low=0, high=shape[1] - 1, size=shape[0])
    labels = np.eye(class_num)[labels_int].astype(dtype)
    logits = np.random.random(shape).astype(dtype)
    logits = np.where(logits < 0.001, 0.001, logits)
    head_np = np.random.uniform(low=0, high=1.0, size=shape[0]).astype(dtype)
    head_np = head_np.reshape([shape[0], 1])
    loss_ad_temp = labels / logits
    loss_ad_temp = loss_ad_temp * head_np
    loss_ad = loss_ad_temp * -1
    return labels, logits, head_np, loss_ad


def crossentropyloss_ad_run(shape, dtype, kernel_name, attrs):
    if (len(shape) != 2):
        raise RuntimeError("shape must be 2-d")
    shape_head = [shape[0], 1]

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(crossentropyloss_ad, [shape_head, shape, shape], [dtype, dtype, dtype],
                                  kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            labels, logits, head_np, expect_loss_ad = GenData(shape, dtype)
            res_loss_ad = np.full(shape, 0, dtype)
            return mod, expect_loss_ad, (head_np, labels, logits, res_loss_ad)
        else:
            return mod
    else:
        labels, logits, head_np, expect_loss_ad = GenData(shape, dtype)
        res_loss_ad = np.full(shape, 0, dtype)
        mod = utils.op_build_test(crossentropyloss_ad, [shape_head, shape, shape], [dtype, dtype, dtype],
                                  kernel_name=kernel_name, attrs=attrs)
        res_loss_ad = utils.mod_launch(mod, (head_np, labels, logits, res_loss_ad), expect=expect_loss_ad)
        cpr_loss_ad = compare_tensor(res_loss_ad, expect_loss_ad, rtol=5e-03, atol=5e-03, equal_nan=True)

        return logits, res_loss_ad, expect_loss_ad, cpr_loss_ad
