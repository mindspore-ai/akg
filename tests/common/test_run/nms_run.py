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

"""
iou run define
"""

import numpy as np
from akg.utils import kernel_exec as utils
from test_op import nms



def gen_data(bs, n, thres, dtype):
    input_tensor = np.array([[10, 10, 15, 30],
                             [9, 10, 13, 30],
                             [30, 15, 45, 25],
                             [30, 50, 90, 55],
                             ]).astype("float16")
    input_tensor = np.pad(input_tensor, ((0, 0), (0, 4)), 'constant', constant_values=(0, 0))
    input_tensor = np.tile(input_tensor, (bs, n // 4, 1))
    np.random.seed(0)
    # noise = np.random.randint(-2,2,size=input_tensor.shape).astype("float16")
    # input_tensor = input_tensor + noise
    input_tensor[0, n // 2 + 1, :] = np.array([0, 0, 100, 100, 0, 0, 0, 0])
    input_tensor[0, n // 2 + 3, :] = np.array([8, 10, 11, 30, 0, 0, 0, 0])

    expect = np_nms(input_tensor, thres)
    out_shape = expect.shape
    output = np.full(out_shape, 0, dtype)
    return input_tensor, expect, output, out_shape


def np_nms(input, thre):
    offset = 1
    A = input[:, :, 0:4]
    bs, anchor_box_num, _ = A.shape
    iou = np.zeros([bs, anchor_box_num, anchor_box_num])
    for i in range(bs):
        x11, y11, x12, y12 = np.split(A[i], 4, axis=1)
        x21, y21, x22, y22 = np.split(A[i], 4, axis=1)
        xA = np.maximum(x11, np.transpose(x21))
        yA = np.maximum(y11, np.transpose(y21))
        xB = np.minimum(x12, np.transpose(x22))
        yB = np.minimum(y12, np.transpose(y22))
        interArea = np.maximum((xB - xA) + offset, 0) * np.maximum((yB - yA) + offset, 0)
        boxAArea = (x12 - x11 + offset) * (y12 - y11 + offset)
        boxBArea = (x22 - x21 + offset) * (y22 - y21 + offset)
        union = (boxAArea + np.transpose(boxBArea) - interArea)
        iou[i] = interArea / union
    pick_vector = np_rpn_cor(iou > thre)
    return pick_vector


def np_rpn_cor(rpn_matrix):
    bs, box_num, _ = rpn_matrix.shape
    rpn_vector = np.zeros((bs, box_num,), dtype=np.float16)
    for i in range(bs):
        for j in range(1, box_num):
            for k in range(j):
                if rpn_matrix[i, j, k] and rpn_vector[i, k] == 0:
                    rpn_vector[i, j] = 6e-8
    return rpn_vector


def nms_run(shape_tensor, thres, dtype, kernel_name, attrs):
    # Create op
    op_attrs = [thres]
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(nms.nms, [shape_tensor], [dtype], op_attrs=op_attrs,
                                  kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            anchor, expect, output, out_shape = gen_data(shape_tensor[0], shape_tensor[1], thres, dtype)
            return mod, expect, (anchor, output)
        else:
            return mod
    else:
        mod = utils.op_build_test(nms.nms, [shape_tensor], [dtype], op_attrs=op_attrs,
                                  kernel_name=kernel_name, attrs=attrs)
        anchor, expect, output, out_shape = gen_data(shape_tensor[0], shape_tensor[1], thres, dtype)
        output = utils.mod_launch(mod, (anchor, output), expect=expect)
        output = np.frombuffer(output.tobytes(), np.uint16).reshape(out_shape)
        source_code = mod.imported_modules[0].get_source()
        utils.create_code(kernel_name, "./", source_code)
        expect = np.frombuffer(expect.tobytes(), np.uint16).reshape(out_shape)
        return anchor, output, expect, np.all(output == expect)
