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
from akg.ops.array import strided_slice
from base import get_rtol_atol


def strided_slice_execute(shape, begin, end, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask,
                          shrink_axis_mask, dtype, attrs):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = strided_slice_compile(shape, begin, end, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask,
                                    shrink_axis_mask, dtype, attrs, kernel_name=kernel_name, tuning=t)
        if t:
            expect, input, output = gen_data(begin, begin_mask, dtype, ellipsis_mask, end, end_mask, new_axis_mask,
                                             shape, shrink_axis_mask, strides)
            return mod, expect, (input, output)
        else:
            return mod
    else:
        mod = strided_slice_compile(shape, begin, end, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask,
                                    shrink_axis_mask, dtype, attrs)
        expect, input, output = gen_data(begin, begin_mask, dtype, ellipsis_mask, end, end_mask, new_axis_mask,
                                         shape, shrink_axis_mask, strides)
        output = utils.mod_launch(mod, (input, output), expect=expect)
        rtol, atol = get_rtol_atol("strided_slice", dtype)
        return input, output, expect, compare_tensor(output, expect, rtol=rtol, atol=atol, equal_nan=True)


def gen_data(begin, begin_mask, dtype, ellipsis_mask, end, end_mask, new_axis_mask, shape, shrink_axis_mask,
             strides):
    """ Generate data for testing the op """
    input = np.random.uniform(low=-1.0, high=1.0, size=tuple(shape)).astype(dtype)
    # get numpy result
    slices = strided_slice.args_to_slices(begin, end, strides,
                                          begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask)
    expect = input[tuple(slices)]
    out_shape = expect.shape if expect.shape != (0,) else (1,)
    output = np.full(out_shape, np.nan, dtype)
    return expect, input, output


def strided_slice_compile(shape, begin, end, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask,
                          shrink_axis_mask, dtype, attrs, kernel_name='strided_slice', tuning=False):
    return utils.op_build_test(strided_slice.strided_slice, [shape], [dtype],
                               op_attrs=[begin, end, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask,
                                         shrink_axis_mask], kernel_name='strided_slice', attrs=attrs, tuning=tuning)
