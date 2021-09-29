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
from akg.ops.array import strided_slice
from tests.common.test_op import strided_slice_grad
from tests.common.tensorio import compare_tensor
from tests.common.base import get_rtol_atol
from tests.common.gen_random import random_gaussian
def check_grad_shape(input_shape, begin, end, strides,
                     begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask,
                         grad_shape_given, dtype=np.float16):
    slices = strided_slice.args_to_slices(begin, end, strides,
                                          begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask)
    dx = np.zeros(input_shape)
    grad_shape = dx[tuple(slices)].shape
    assert list(grad_shape) == list(grad_shape_given), \
        ("parameters invalid: grad shape should be ", list(grad_shape),
         "but given is", list(grad_shape_given))
    return slices

def strided_slice_python(input_shape, begin, end, strides,
                         begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask,
                         grad, dtype=np.float16):
    slices = strided_slice.args_to_slices(begin, end, strides,
                                          begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask)

    dx = np.zeros(input_shape).astype(dtype)
    dx[tuple(slices)] = grad
    return dx


def strided_slice_grad_execute(input_shape, begin, end, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask,
                               shrink_axis_mask, grad_shape, dtype, attrs=None):
    check_grad_shape(input_shape, begin, end, strides,
                     begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask,
                     grad_shape, dtype)
    attrs["pragma_disable_whole_component"] = False
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = strided_slice_grad_compile(input_shape, begin, end, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask,
                                         shrink_axis_mask, grad_shape, dtype, attrs=None, kernel_name=kernel_name, tuning=t)
        if t:
            expect, grad, output = gen_data(begin, begin_mask, dtype, ellipsis_mask, end, end_mask, grad_shape,
                                            input_shape, new_axis_mask, shrink_axis_mask, strides)
            return mod, expect, (grad, output)
        else:
            return mod
    else:
        mod = strided_slice_grad_compile(input_shape, begin, end, strides, begin_mask, end_mask, ellipsis_mask,
                                         new_axis_mask, shrink_axis_mask, grad_shape, dtype, attrs)
        expect, grad, output = gen_data(begin, begin_mask, dtype, ellipsis_mask, end, end_mask, grad_shape, input_shape,
                                        new_axis_mask, shrink_axis_mask, strides)
        output = utils.mod_launch(mod, (grad, output), expect=expect)

        rtol, atol = get_rtol_atol("strided_slice_grad", dtype)
        return grad, output, expect, compare_tensor(output, expect, rtol=rtol, atol=atol, equal_nan=True)


def gen_data(begin, begin_mask, dtype, ellipsis_mask, end, end_mask, grad_shape, input_shape, new_axis_mask,
             shrink_axis_mask, strides):
    grad = random_gaussian(grad_shape, miu=1, sigma=0.1).astype(dtype)
    expect = strided_slice_python(input_shape, begin, end, strides,
                                  begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask,
                                  grad, dtype)
    # mod = strided_slice_grad.strided_slice_grad(input_shape, begin, end, strides,
    #                                           begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask,
    #                                            grad_shape, dtype, kernel_name="strided_slice_grad", attrs=attrs)
    # source_code = mod.imported_modules[0].get_source()
    # print(source_code)
    # kernel_name = "cce_strided_slice_grad_fp16"
    # utils.create_code(kernel_name, './', source_code)
    out_shape = input_shape
    output = np.full(out_shape, 0, dtype)
    return expect, grad, output


def strided_slice_grad_compile(input_shape, begin, end, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask,
                               shrink_axis_mask, grad_shape, dtype, attrs=None, kernel_name="strided_slice_grad",
                               tuning=False):
    return utils.op_build_test(strided_slice_grad.strided_slice_grad, [grad_shape], [dtype],
                               op_attrs=[input_shape, begin, end, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask,
                                         shrink_axis_mask], kernel_name=kernel_name, attrs=attrs, tuning=tuning)
