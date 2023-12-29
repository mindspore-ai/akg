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

"""operator dsl function: rsqrt"""
import akg.topi
import akg.tvm
import akg.utils as utils
from akg.utils.kernel_exec import product_is_mini
from akg.utils.format_transform import get_shape


@utils.check_input_type(akg.tvm.tensor.Tensor)
def _rsqrt(data1):
    """
    Computes data1 elementwise.

    Args:
        data1 (tvm.tensor.Tensor): Tensor.

    Returns:
        tvm.tensor.Tensor, inverse sqaure root of data1, with same type as input tensors.
    """
    utils.ops_dtype_check(data1.dtype, ["float32", "float16"])
    utils.check_shape(data1.shape)

    res = akg.topi.rsqrt(data1)

    return res


@utils.check_input_type(akg.tvm.tensor.Tensor)
def _rsqrt_ascend(data):
    """
    Computes reciprocal of square root of x element-wise.

     :math:`y = \frac{1}{\\sqrt x} = x^{-\frac{1}{2}}`

    Note:
        In order to prevent loss of precision, the function uses exponential constant changes:
        :math:`y = [e^{lnx}]^{-\frac{1}{2}}`

    Args:
        data (tvm.tensor.Tensor): Tensor of type float16, float32

    Returns:
        tvm.tensor.Tensor, has same type and shape as data.
    """

    dtype = data.dtype

    shape = get_shape(data)
    utils.ops_dtype_check(dtype, utils.DtypeForDavinci.ALL_FLOAT)
    utils.check_shape(shape)

    if not product_is_mini():
        return akg.topi.rsqrt(data)

    is_needed_conv = (dtype == 'float32')

    data_ = data.astype('float16') if is_needed_conv else data
    power_num = akg.tvm.const(-0.5,
                              dtype=('float16' if is_needed_conv else dtype))

    vlog_t = akg.tvm.compute(
        shape, lambda *indice: akg.tvm.log(data_(*indice)), name="vlog_t")
    vmuls_t = akg.tvm.compute(
        shape, lambda *indice: vlog_t(*indice) * power_num, name="vmuls_t")
    res = akg.tvm.compute(
        shape, lambda *indice: akg.tvm.exp(vmuls_t(*indice)), name="res")

    res = res.astype('float32') if is_needed_conv else res

    return res


def rsqrt(data, target=utils.CCE):
    """
    Computes data1 elementwise.

    Args:
        data1 (tvm.tensor.Tensor): Tensor.

    Returns:
        tvm.tensor.Tensor, inverse sqaure root of data1, with same type as input tensors.

    Supported Platforms:
        'Ascend', 'GPU', 'CPU'
    """
    utils.check_supported_target(target)
    if target == utils.CCE:
        return _rsqrt_ascend(data)
    else:
        return _rsqrt(data)
