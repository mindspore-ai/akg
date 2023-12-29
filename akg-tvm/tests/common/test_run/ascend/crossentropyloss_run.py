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
from tests.common.tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from tests.common.test_op.ascend.crossentropyloss import crossentropyloss


def GenData(shape, dtype):
    """ Generate data for testing the op """
    class_num = shape[1]
    labels_int = np.random.randint(low=0, high=shape[1] - 1, size=shape[0])
    labels = np.eye(class_num)[labels_int].astype(dtype)
    logits = np.random.random(shape).astype(dtype)
    logits = logits / 10 + 1e-05
    loss_all = labels * np.log(logits) * -1
    loss = np.sum(loss_all, axis=-1)
    return labels, logits, loss


def crossentropyloss_run(shape, axis, dtype, kernel_name, attrs):
    labels, logits, expect_loss = GenData(shape, dtype)
    res_loss = np.full(shape[0], 0, dtype)

    mod = utils.op_build_test(crossentropyloss, [shape, shape], [dtype, dtype], op_attrs=[axis],
                              kernel_name=kernel_name, attrs=attrs)
    res_loss = utils.mod_launch(mod, (labels, logits, res_loss), expect=expect_loss)
    cpr_loss = compare_tensor(res_loss, expect_loss, rtol=1e-03, atol=1e-03, equal_nan=True)
    return logits, res_loss, expect_loss, cpr_loss
