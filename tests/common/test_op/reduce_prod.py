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

"""operator dsl function:reduce_prod"""


import akg.topi
import akg.tvm
from akg.utils.validation_check import check_shape, ops_dtype_check, check_input_type, DtypeForDavinci
from akg.utils import format_transform as ft_util
from akg.ops.math.log import log as akg_log
from akg.ops.math.exp import exp as akg_exp

@check_input_type(akg.tvm.tensor.Tensor, (int, tuple, list, type(None)), (bool, type(None)))
def reduce_prod(data, axis=None, keepdims=False):
    """
    Computes the product of elements along specific axis

    Args:
        data (tvm.tensor.Tensor): indicating the input tensor.
        axis (Union[list, tuple, int, None]): indicating the dimensions to reduce at. if it's None, all dimensions
                                               will be reduced.
        keepdims (Union[bool, None]): if true, keep the dimensions with length 1.

    Returns:
    Tensor, the product of elements of input tensor.
    """
    shape = [x.value for x in data.shape]
    ops_dtype_check(data.dtype, [DtypeForDavinci.ALL_FLOAT, DtypeForDavinci.INT8, DtypeForDavinci.UINT8])

    if axis is None and keepdims is False:
        raise ValueError("keepdims must be True when axis is None!")

    axis_new = ft_util.refine_reduce_axis(data, axis)

    check_shape(shape)
    dtype = data.dtype
    if dtype in ["int8", "uint8"]:
        data = akg.topi.cast(data, "float16")

    vlog_t = akg_log(data)
    res = akg.topi.sum(vlog_t, axis=axis_new, keepdims=keepdims)
    res = akg_exp(res)

    if dtype in ["int8", "uint8"]:
        res = akg.topi.cast(res, dtype)
    return res
