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

"""l1_loss"""

import akg.topi
from akg.topi.util import get_const_tuple
import akg.utils as utils
import akg.utils as utils
from akg.utils.kernel_exec import product_is_mini

def l1_loss(inputs, outputs, reduction='none', target="cce"):
    """
    Computes l1 loss.

    Args:
        inputs(akg.tvm.Tensor): Supported data type is float16, float32.
        outputs(akg.tvm.Tensor): With same type as inputs.
        reduction(str): Default is 'none', could be 'sum' or 'mean', if 'mean', loss result will be divided
                        by the size of inputs.

    Returns:
        akg.tvm.Tensor of same type as input tensors.
    """
    inputs_dtype = inputs.dtype
    target_dtype = outputs.dtype

    # check inputs data types
    utils.ops_dtype_check([inputs_dtype, target_dtype], utils.DtypeForDavinci.ALL_FLOAT)
    target_shape = [x.value for x in outputs.shape]
    inputs_shape = [x.value for x in inputs.shape]
    utils.elemwise_shape_check(target_shape, inputs_shape)

    inputs_dtype_old = inputs_dtype

    if product_is_mini() and inputs_dtype == 'float32':
        inputs = akg.topi.cast(inputs, "float16")
        outputs = akg.topi.cast(outputs, "float16")
        inputs_dtype = "float16"

    diff = akg.topi.subtract(inputs, outputs)
    loss = akg.topi.abs(diff)
    if reduction == 'sum':
        loss = akg.topi.sum(loss)
    if reduction == 'mean':
        loss = akg.topi.sum(loss)
        deno = 1.0
        for num in inputs.shape:
            deno = deno * num
        deno = akg.topi.cast(deno, dtype=inputs_dtype)
        loss = akg.topi.divide(loss, deno)

    if product_is_mini() and inputs_dtype_old == 'float32':
        loss = akg.topi.cast(loss, inputs_dtype_old)
    return loss
