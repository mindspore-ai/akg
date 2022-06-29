# Copyright 2020-2022 Huawei Technologies Co., Ltd
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

"""operator dsl function:eltwise"""
import akg.topi
import akg.tvm
import akg.utils as utils
import akg.utils as utils
from akg.utils.format_transform import get_shape
from akg.ops.math import addn

def _max(data):
    """
    return max of data
    Args:
        data (tvm.tensor.Tensor) input
    Returns:
        tvm.tensor.Tensor
    """
    res = data[0]
    for i in range(1, len(data)):
        res = akg.lang.ascend.vmax(res, data[i])
    
    return res


def _product(data):
    """
    return product of data
    Args:
        data (tvm.tensor.Tensor) input
    Returns:
        tvm.tensor.Tensor
    """

    res = data[0]
    for i in range(1, len(data)):
        res = akg.lang.ascend.vmul(res, data[i])
    
    return res

def _addn(data, coeff):
    """
    return sum of data[i]*coeff[i]
    Args:
        data (tvm.tensor.Tensor) input
        coeff (tuple): tuple of scalar, float/int
    Returns:
        tvm.tensor.Tensor
    """
    if coeff[0] == 1:
        res = data[0]
    else:
        pone = akg.tvm.const(coeff[0], dtype=data[0].dtype)
        res = akg.lang.ascend.vmuls(data[0], pone)
    
    for i in range(1, len(data)):
        if coeff[i] != 1:
            pone = akg.tvm.const(coeff[i], dtype=data[0].dtype)
            data[i] = akg.lang.ascend.vmuls(data[i], pone)
        res = akg.lang.ascend.vadd(res, data[i])
    
    return res

@utils.check_input_type(((list, tuple), akg.tvm.tensor.Tensor), (int, None), (tuple, None))
def eltwise(data, mode=1, coeff=()):
    """
    Compute elementwise modes, such as 0:PRODUCT, 1:SUM and 2:MAX.

    Args:
        data (list of tvm.tensor.Tensor): a list of tensor, tensor support fp16 and fp32.
        mode (int): 0:product, 1:sum, 2:max.
        coeff (tuple): tensor name of data should be equal with coeff size, only
                      used by sum, support int and float.
    Returns:
        tvm.tensor.Tensor.
    """
    dtype = data[0].dtype
    utils.ops_dtype_check(dtype, utils.DtypeForDavinci.ALL_FLOAT)

    utils.check_shape(data[0].shape)
    shape_data = get_shape(data[0])

    if not mode in [0, 1, 2]:
        raise RuntimeError("mode only support 0, 1, or 2")

    if not len(data) == len(coeff) and len(coeff) != 0:
        raise RuntimeError(
            "coeff should be [] or its length be same as data")

    tensor_num = len(data)
    #tensor num must be [1, 120]
    if tensor_num < 1 or tensor_num > 120:
        raise RuntimeError("tensor_num need in range [1,120].")

    if mode == 1 and len(coeff) == 0:
        return addn.addn(data)
    
    if len(coeff) != 0:
        if type(coeff[0]) != int and type(coeff[0]) != float:
            raise RuntimeError("ele of coeff must be a number.")

    for i in range(1, len(data)):
        utils.elemwise_dtype_check(data[0].dtype, data[i].dtype)
        utils.elemwise_shape_check(data[0].shape, data[i].shape)

    if mode == 1 and len(coeff) > 0:
        return _addn(data, coeff)
    
    if mode == 0:
        return _product(data)
    
    if mode == 2:
        return _max(data)
