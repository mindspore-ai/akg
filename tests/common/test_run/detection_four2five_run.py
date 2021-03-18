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
from tests.common.test_op import detection_four2five
from tests.common.gen_random import random_gaussian

def detection_four2five_np(bs_i):
    shape_i = [bs_i, 8732 * 4]
    input_data = random_gaussian(shape_i, miu=1, sigma=0.1).astype("float16")
    result0 = input_data[:, 0:16]
    result1 = input_data[:, 16:160]
    result2 = input_data[:, 160:760]
    result3 = input_data[:, 760:3160]
    result4 = input_data[:, 3160:11824]
    result5 = input_data[:, 11824:34928]
    return input_data, [result0, result1, result2, result3, result4, result5]


def detection_four2five_run(args, dtype, attrs):
    bs, slice_idx = args

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        inputs, expects = detection_four2five_np(bs)
        mod = utils.op_build_test(detection_four2five.detection_four2five, [inputs.shape], [dtype],
                                  op_attrs=[slice_idx], kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            out_shape = [i.shape for i in expects]
            expect = expects[slice_idx]
            output = np.full(out_shape[slice_idx], 0, dtype)
            return mod, expect, (inputs, output)
        else:
            return mod
    else:
        inputs, expects = detection_four2five_np(bs)
        out_shape = [i.shape for i in expects]
        expect = expects[slice_idx]
        mod = utils.op_build_test(detection_four2five.detection_four2five, [inputs.shape], [dtype],
                                  op_attrs=[slice_idx], kernel_name='detection_four2five', attrs=attrs)
        output = np.full(out_shape[slice_idx], 0, dtype)
        output = utils.mod_launch(mod, [inputs, output], expect=expect)
        return inputs, output, expect, compare_tensor(output, expect, rtol=5e-03, equal_nan=True)
