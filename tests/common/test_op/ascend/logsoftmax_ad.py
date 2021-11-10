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

"""operator dsl function: logsoftmax_ad"""
import akg.tvm
import akg
from akg.utils.kernel_exec import debug_mode
from tests.common.test_op.ascend import logsoftmax


def logsoftmax_ad(shape, dtype, axis, kernel_name, attrs):
    """Compute the gradient of logsoftmax by autodiff."""
    check_list = ["float16"]
    if not dtype.lower() in check_list:
        raise RuntimeError("logsoftmax test only support %s while dtype is %s" % (",".join(check_list), dtype))
    # check_shape(shape)
    if axis < 0:
        axis = len(shape) + axis
    if axis >= len(shape):
        raise RuntimeError("axis should be less than dimension")
    if axis != len(shape) - 1:
        raise RuntimeError("Only support the last axis currently")

    shape_new = [shape[-2], shape[-1]]
    if len(shape) > 2:
        for i in range(len(shape) - 2):
            shape_new[0] = shape_new[0] * shape[i]
    shape = shape_new

    a_up = akg.tvm.placeholder(shape, dtype=dtype, name="input")
    b_up = logsoftmax.logsoftmax_op(a_up, shape, axis)

    head = akg.tvm.placeholder(b_up.shape, name="head", dtype=dtype)
    _jacs = list(akg.differentiate(b_up, [a_up], head))
    sjac = akg.tvm.create_schedule([_jacs[0].op])
    sjac[_jacs[0].op.input_tensors[1]].compute_inline()
    op_vars = [head, a_up, _jacs[0]]

    with akg.build_config(add_lower_pass=debug_mode(0), dump_pass_ir=True):
        mod = akg.build(sjac, op_vars, "cce", name="test2", attrs=attrs, polyhedral=True)
        return mod
