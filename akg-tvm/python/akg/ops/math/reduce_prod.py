# Copyright 2021-2022 Huawei Technologies Co., Ltd
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

"""operator dsl function: prod"""
import akg.topi
import akg.tvm
import akg.utils as utils
from akg.utils.format_transform import refine_reduce_axis
from akg.utils.format_transform import refine_reduce_axis
from .exp import exp
from .log import log


@utils.check_input_type(akg.tvm.tensor.Tensor, (int, tuple, list, type(None)), (bool, type(None)), (str, type(None)))
def reduce_prod(data, axis=None, keepdims=False, target=utils.CCE):
    """
    Computes the product of elements along specific axis

    Args:
        data (tvm.tensor.Tensor): indicating the input tensor.
        axis (Union[list, tuple, int, None]): indicating the dimensions to reduce at. if it's None, all dimensions
                                               will be reduced.
        keepdims (Union[bool, None]): if true, keep the dimensions with length 1.

    Returns:
        Tensor, the product of elements of input tensor.

    Supported Platforms:
        'Ascend', 'GPU'
    """
    utils.check_supported_target(target)
    shape = [x.value for x in data.shape]
    utils.ops_dtype_check(data.dtype, [utils.DtypeForDavinci.ALL_FLOAT,
        utils.DtypeForDavinci.INT8, utils.DtypeForDavinci.UINT8])

    if axis is None and keepdims is False:
        raise ValueError("keepdims must be True when axis is None!")

    axis_new = refine_reduce_axis(data, axis)

    if target == utils.CUDA:
        return akg.topi.prod(data, axis=axis, keepdims=keepdims)

    utils.check_shape(shape)
    dtype = data.dtype
    if dtype in ["int8", "uint8"]:
        data = akg.topi.cast(data, "float16")

    vlog_t = log(data, target)
    res = akg.topi.sum(vlog_t, axis=axis_new, keepdims=keepdims)
    res = exp(res, target)

    if dtype in ["int8", "uint8"]:
        res = akg.topi.cast(res, dtype)
    return res