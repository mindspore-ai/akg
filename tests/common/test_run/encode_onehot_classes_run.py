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
from akg.utils import kernel_exec as utils
from test_op import encode_onehot_classes
from tensorio import compare_tensor


def onehot_expect(groundtruth_class, anchor_sample, class_num):
    batch_size, anchor_num = anchor_sample.shape
    out_shape = [batch_size, anchor_num, class_num]
    res = np.full(out_shape, 0, groundtruth_class.dtype)
    for b in range(batch_size):
        for a in range(anchor_num):
            res[b, a, groundtruth_class[b, anchor_sample[b, a]]] = 1
    return res


def encode_onehot_classes_run(gt_shape, anchor_shape, class_num, dtype, kernel_name, attrs):
    kernel_name = utils.gen_name_kernel(kernel_name, dtype, gt_shape)
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(encode_onehot_classes.encode_onehot_classes,
                                  input_shapes=[gt_shape, anchor_shape],
                                  input_types=[dtype, dtype],
                                  op_attrs=[class_num], kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            anchor_sample, expect, groundtruth_class, output = gen_data(anchor_shape, class_num, dtype, gt_shape)
            return mod, expect, (groundtruth_class, anchor_sample, output)
        else:
            return mod
    else:
        # Create op
        mod = utils.op_build_test(encode_onehot_classes.encode_onehot_classes,
                                  input_shapes=[gt_shape, anchor_shape],
                                  input_types=[dtype, dtype],
                                  op_attrs=[class_num], kernel_name=kernel_name, attrs=attrs)

        anchor_sample, expect, groundtruth_class, output = gen_data(anchor_shape, class_num, dtype, gt_shape)
        output = utils.mod_launch(mod, (groundtruth_class, anchor_sample, output), expect=expect)
        return (groundtruth_class, anchor_sample), output, expect, compare_tensor(output, expect, atol=5e-01, rtol=5e-03, equal_nan=True)


def gen_data(anchor_shape, class_num, dtype, gt_shape):
    # Generate data for testing the op
    groundtruth_class = np.random.randint(class_num, size=gt_shape, dtype=dtype)
    anchor_sample = np.random.randint(gt_shape[1], size=anchor_shape, dtype=dtype)
    # Generate expected output using numpy implementation of resize bilinear
    expect = onehot_expect(groundtruth_class, anchor_sample, class_num)
    # Predict output
    output = np.full(expect.shape, np.nan, dtype)
    return anchor_sample, expect, groundtruth_class, output
