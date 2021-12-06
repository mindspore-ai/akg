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
from akg.utils import kernel_exec as utils
from tests.common.test_op.ascend import strided_slice_ad
from tests.common.tensorio import compare_tensor


def strided_slice_python(input, begin, end, strides, grad):
    strides = [] if strides is None else strides
    slices = []
    for i in range(len(input)):
        e = end[i]
        if end[i] == 0:
            e = input[i]
        slices.append(slice(
            begin[i] if i < len(begin) else None,
            e if i < len(end) else None,
            strides[i] if i < len(strides) else None))
    data = np.zeros(input).astype(np.float64)
    data[tuple(slices)] = grad
    return data


def strided_slice_ad_run(input_shape, begin, end, strides, dtype, attrs_op={}, cce_path="./", attrs={}):
    out_shape = [(e - b) // s for b, e, s in zip(begin, end, strides)]
    attrs.update(attrs_op)
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(strided_slice_ad.strided_slice_ad, [out_shape, input_shape], [dtype, dtype],
                                  [begin, end, strides, dtype], kernel_name, attrs, tuning=t)
        if t:
            H_data, expect, input1, output = gen_data(begin, dtype, end, input_shape, out_shape, strides)
            return mod, expect, (H_data, input1, output)
        else:
            return mod
    else:
        mod = utils.op_build_test(strided_slice_ad.strided_slice_ad, [out_shape, input_shape], [dtype, dtype],
                                  [begin, end, strides, dtype], "strided_slice_ad", attrs)
        H_data, expect, input1, output = gen_data(begin, dtype, end, input_shape, out_shape, strides)
        output = utils.mod_launch(mod, (H_data, input1, output), expect=expect)

        return (H_data, input1), output, expect, compare_tensor(output, expect, rtol=0.1, equal_nan=True)


def gen_data(begin, dtype, end, input_shape, out_shape, strides):
    input1 = np.zeros(input_shape)
    H_data = np.full(out_shape, 1., dtype)
    if (dtype == "int32"):
        input1 = input1.astype(np.int32)
    elif (dtype == "float16"):
        input1 = input1.astype(np.float16)
    else:
        input1 = input1.astype(np.float32)
    expect = strided_slice_python(input_shape, begin, end, strides, H_data)
    output = np.full(input_shape, 0, dtype)
    return H_data, expect, input1, output
