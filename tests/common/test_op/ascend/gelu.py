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

"""operator dsl function:gelu"""
import akg.topi
import akg.tvm
import akg.utils as utils
from akg.ops.math.ascend import Tanh
from akg.utils.kernel_exec import product_is_mini

@utils.check_input_type(akg.tvm.tensor.Tensor)
def gelu(data):
    """
    gelu activation function.

    ..math:`0.5*data(1+tanh(sqrt(2/pi)(data+0.044715data^3)))`

    Args:
        x (tvm.tensor.Tensor): tensor with type float16 or float32.

    ..math:`0.5*x(1+tanh(sqrt(2/pi)(x+0.044715x^3)))
        data (tvm.tensor.Tensor): tensor with type float16 or float32.

    Returns:
        tvm.tensor.Tensor.
    """
    dtype = data.dtype
    utils.ops_dtype_check(dtype, utils.DtypeForDavinci.ALL_FLOAT)

    if dtype == "float32" and product_is_mini():
        data = akg.tvm.compute(data.shape, lambda *indice: data(*indice).astype("float16"), name='type_cast')
        dtype = "float16"
    tmp0 = akg.topi.multiply(data, data)
    pow0 = akg.topi.multiply(tmp0, data)
    mul0 = pow0 * akg.tvm.const(0.044715, dtype)
    add0 = data + mul0
    mul1 = add0 * akg.tvm.const(0.7978845, dtype)
    tanh_res = Tanh(mul1)
    add1 = tanh_res + akg.tvm.const(1, dtype)
    mul2 = add1 * akg.tvm.const(0.5, dtype)
    mul3 = data * mul2
    res = mul3

    if dtype == "float32" and product_is_mini():
        res = akg.tvm.compute(res.shape, lambda *indice: res(*indice).astype("float16"), name='res')
    return res


