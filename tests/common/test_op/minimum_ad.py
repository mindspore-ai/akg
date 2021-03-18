# Copyright 2020 Huawei Technologies Co., Ltd
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

"""operator dsl function: minimum_ad"""
import akg
from tests.common.test_op import minimum
from akg.utils import validation_check as vc_util

@vc_util.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, (bool, type(None)),
                          (bool, type(None)))
def minimum_ad(head, data_x, data_y, grad_x=True, grad_y=True):
    """
    Calculating the reversed outputs of the operator minimum by using automatic differentiate.

    Args:
        head (tvm.tensor.Tensor): Input tensor of float32, float16 and int32.
        data_x (tvm.tensor.Tensor): Input tensor of float32, float16 and int32.
        data_y (tvm.tensor.Tensor): Input tensor of float32, float16 and int32.
        grad_x (bool): Default is True, whether to differentiate x.
        grad_y (bool): Default is True, whether to differentiate y.

    Returns:
        tvm.tensor.Tensor, has the same type and shape as grads, if grad_x and grad_y all equal to True, need return
        a list like: [jacs[0], jacs[1]].
    """
    vc_util.elemwise_shape_check(data_x.shape, data_y.shape)
    vc_util.elemwise_shape_check(head.shape, data_x.shape)
    vc_util.elemwise_dtype_check(data_x.dtype, head.dtype,
                                 [vc_util.DtypeForDavinci.ALL_FLOAT, vc_util.DtypeForDavinci.INT32])
    vc_util.elemwise_dtype_check(data_x.dtype, data_y.dtype,
                                 [vc_util.DtypeForDavinci.ALL_FLOAT, vc_util.DtypeForDavinci.INT32])
    if not grad_x and not grad_y:
        raise ValueError("At least one of grad_x and grad_y is True.")
    op = minimum.minimum(data_x, data_y)
    jacs = list(akg.differentiate(op, [data_x, data_y], head))
    if grad_x and grad_y:
        return jacs[0], jacs[1]
    if grad_x:
        return jacs[0]
    return jacs[1]
