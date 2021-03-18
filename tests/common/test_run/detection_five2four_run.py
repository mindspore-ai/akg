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

from akg.utils import kernel_exec as utils
from tests.common.tensorio import compare_tensor
import numpy as np
from tests.common.test_op import detection_five2four
from tests.common.gen_random import random_gaussian

def detection_five2four_np(arg_list):
    bs_i, box_num, wi = arg_list
    shape_i = [bs_i, wi * wi * box_num, 4]
    result = random_gaussian(shape_i, miu=1, sigma=0.1).astype("float16")
    input_data = result.reshape(bs_i * wi * wi, box_num * 4)
    if box_num % 4 != 0:
        pad_box = 4 - box_num % 4
        input_data = np.pad(input_data, ((0, 0), (0, pad_box * 4)), 'constant')
        box_num += pad_box
    input_data = input_data.reshape(bs_i, wi, wi, box_num // 4, 16).transpose([0, 3, 1, 2, 4])
    return input_data, result


def detection_five2four_run(input_box, dtype, attrs):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        inputs, expect = detection_five2four_np(input_box)
        mod = utils.op_build_test(detection_five2four.detection_five2four, [inputs.shape], [dtype],
                                  op_attrs=[input_box[1]], kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            out_shape = expect.shape
            output = np.full(out_shape, 0, dtype)
            return mod, expect, [inputs, output]
        else:
            return mod
    else:
        inputs, expect = detection_five2four_np(input_box)
        mod = utils.op_build_test(detection_five2four.detection_five2four, [inputs.shape], [dtype],
                                  op_attrs=[input_box[1]], kernel_name='detection_five2four', attrs=attrs)
        out_shape = expect.shape
        output = np.full(out_shape, 0, dtype)
        output = utils.mod_launch(mod, [inputs, output], expect=expect)
        return inputs, output, expect, compare_tensor(output, expect, rtol=5e-03, equal_nan=True)
