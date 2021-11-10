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
from tests.common.test_op.ascend import triplet_loss


def triplet_loss_run(shape, dtype, margin=12.0, kernel_name="triplet_loss", attrs={}):
    op_attrs = [margin]

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(triplet_loss.triplet_loss_naive, [shape, shape, shape], [dtype, dtype, dtype],
                                  op_attrs=op_attrs, kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            anchor, expect, neg, output, pos = gen_data(dtype, margin, shape)
            return mod, expect, (anchor, pos, neg, output)
        else:
            return mod
    else:
        anchor, expect, neg, output, pos = gen_data(dtype, margin, shape)
        mod = utils.op_build_test(triplet_loss.triplet_loss_naive, [shape, shape, shape], [dtype, dtype, dtype],
                                  op_attrs=op_attrs, kernel_name=kernel_name, attrs=attrs)
        output = utils.mod_launch(mod, (anchor, pos, neg, output), expect=expect)

        return anchor, output, expect, compare_tensor(output, expect, rtol=5e-03, equal_nan=True)


def gen_data(dtype, margin, shape):
    # create a non-zero loss situation
    anchor = np.arange(np.prod(shape)).reshape(shape).astype(dtype)
    pos = anchor + 1.0
    neg = anchor + 2.0
    d_pos = np.sum((anchor - pos) * (anchor - pos), -1)
    d_neg = np.sum((anchor - neg) * (anchor - neg), -1)
    loss = margin + d_pos - d_neg  # margin + 1.0 * shape[-1] - 4.0 * shape[-1]
    np.maximum(loss, 0, loss)  # perform relu
    expect = loss
    output = np.full(expect.shape, np.nan, dtype)
    return anchor, expect, neg, output, pos
