# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

"""operator dsl fuction: apply_gradient_descent"""

import akg
from akg import topi, tvm
import akg.utils as utils
import akg.utils as utils
from akg.utils.dsl_create import TensorUtils
from akg.utils.format_transform import get_shape


def _apply_gradient_descent_compute(var, alpha, delta):
    """Compute gradient_descent"""
    # step 1: calculate delta * alpha
    var_change = tvm.compute(delta.shape,
                             lambda *indices: delta(*indices) * alpha[0])
    # step 2: calculate var - delta * alpha
    reuse_var = topi.subtract(var, var_change)
    return reuse_var


@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, (str, type(None)))
def apply_gradient_descent(var, alpha, delta, target=utils.CCE):
    """
    Update var by subtracting alpha * delta from it.

    .. math::
        var_{t} = var_{t-1} - \\alpha \\delta

    Args:
        var (tvm.tensor.Tensor): Input var of dtype float16, float32.
        alpha (tvm.tensor.Tensor): A scalar tensor of same type as input var.
        delta (tvm.tensor.Tensor): A tensor of same shape and dtype as input var.

    Returns:
        tvm.tensor.Tensor, Updated var.
    """
    # check dtypes
    utils.ops_dtype_check(var.dtype, utils.DtypeForDavinci.ALL_FLOAT)
    for i in (alpha, delta):
        utils.elemwise_dtype_check(var.dtype, i.dtype)

    # check shapes
    utils.elemwise_shape_check(var.shape, delta.shape)
    if tuple(get_shape(alpha)) != (1,):
        raise RuntimeError("input alpha only support scalar tensor.")

    # compute
    out_var = _apply_gradient_descent_compute(var, alpha, delta)

    # reuse var
    out_var, binds_info = TensorUtils.inplace_set(var, out_var, "var_buf")
    attrs = {utils.BINDS: binds_info}
    return out_var, attrs
