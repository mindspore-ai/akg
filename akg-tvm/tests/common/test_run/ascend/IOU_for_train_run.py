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

"""
iou run define
"""

import numpy as np
from akg.utils import kernel_exec as utils
from tests.common.tensorio import compare_tensor
from tests.common.test_op.ascend import IOU_for_train


def gen_data(m, n):
    input_tensor = np.array([[11, 10, 30, 30],
                             [50, 10, 80, 20],
                             [40, 15, 85, 25],
                             [82, 50, 90, 90],
                             ]).astype("float16")
    input_tensor = np.pad(input_tensor, ((0, 0), (0, 4)), 'constant', constant_values=(0, 0))
    input_tensor = np.tile(input_tensor, (m, n // 4, 1))
    np.random.seed(0)
    noise = np.random.randint(-5, 5, size=input_tensor.shape).astype("float16")
    return input_tensor + noise


def np_iou(A, B):
    offset = 1
    A = A[:, :, 0:4]
    B = B[:, :, 0:4]
    bs, anchor_box_num, _ = A.shape
    _, ground_truth_box_num, _ = B.shape
    iou = np.zeros([bs, anchor_box_num, ground_truth_box_num])
    for i in range(bs):
        x11, y11, x12, y12 = np.split(A[i], 4, axis=1)
        x21, y21, x22, y22 = np.split(B[i], 4, axis=1)
        xA = np.maximum(x11, np.transpose(x21))
        yA = np.maximum(y11, np.transpose(y21))
        xB = np.minimum(x12, np.transpose(x22))
        yB = np.minimum(y12, np.transpose(y22))
        interArea = np.maximum((xB - xA) + offset, 0) * np.maximum((yB - yA) + offset, 0)
        boxAArea = (x12 - x11 + offset) * (y12 - y11 + offset)
        boxBArea = (x22 - x21 + offset) * (y22 - y21 + offset)
        union = (boxAArea + np.transpose(boxBArea) - interArea)
        iou[i] = interArea / union

    return iou


def iou_for_train_run(shape_tensor,
                      shape_tensor1,
                      dtype,
                      kernel_name,
                      attrs):
    # Create op
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(IOU_for_train.iou_for_train, [shape_tensor, shape_tensor1], [dtype, dtype],
                                  kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            anchor, expect, ground_truth, output = gen_output_data(dtype, shape_tensor, shape_tensor1)
            return mod, expect, (anchor, ground_truth, output)
        else:
            return mod
    else:
        mod = utils.op_build_test(IOU_for_train.iou_for_train, [shape_tensor, shape_tensor1], [dtype, dtype],
                                  kernel_name=kernel_name, attrs=attrs)
        anchor, expect, ground_truth, output = gen_output_data(dtype, shape_tensor, shape_tensor1)
        output = utils.mod_launch(mod, (anchor, ground_truth, output), expect=expect)

        source_code = mod.imported_modules[0].get_source()
        utils.create_code(kernel_name, "./", source_code)
        return input, output, expect, compare_tensor(output, expect, rtol=5e-03, equal_nan=True)


def gen_output_data(dtype, shape_tensor, shape_tensor1):
    anchor = gen_data(shape_tensor[0], shape_tensor[1])
    ground_truth = gen_data(shape_tensor1[0], shape_tensor1[1])
    out_shape = (shape_tensor[0], shape_tensor[1], shape_tensor1[1])
    expect = np_iou(anchor, ground_truth)
    output = np.full(out_shape, np.nan, dtype)
    return anchor, expect, ground_truth, output
