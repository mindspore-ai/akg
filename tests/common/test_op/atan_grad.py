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

"""operator dsl function: atan_grad"""

import akg
from akg import tvm
from akg.utils import validation_check as vc_util, dsl_create as dc


@vc_util.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor)
def atan_grad(head, input_x):
    """
    Compute gradient of input_x in atan.

    .. math::
        dx = \\frac{1}{1 + x^2} \\cdot dy

    Args:
        head (tvm.tensor.Tensor): Gradient tensor of forward's output with the
                                  same shape and dtype as input_x.
        input_x (tvm.tensor.Tensor): Forward's input tensor support float16
                                     and float32.

    Returns:
        A tvm.tensor.Tensor as gradient of forward's input.
    """
    vc_util.elemwise_shape_check(head.shape, input_x.shape)
    vc_util.elemwise_dtype_check(head.dtype, input_x.dtype,
                                 vc_util.DtypeForDavinci.ALL_FLOAT)

    dtype = input_x.dtype
    tensor_one = dc.one_const(dtype)

    def _compute(*i):
        return tensor_one / (tensor_one + input_x(*i) * input_x(*i)) * head(*i)

    out_tensor = tvm.compute(input_x.shape, _compute, name="out")

    return out_tensor
