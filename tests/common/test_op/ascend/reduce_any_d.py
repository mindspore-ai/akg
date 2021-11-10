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

"""operator dsl function: reduce_any_d"""

import akg
from akg import tvm, topi
import akg.utils as utils
from akg.utils.format_transform import refine_reduce_axis

def _reduce_any_d_compute(x, axis=None, keepdims=None):
    """reduce_any_d compute implemention"""
    dtype = x.dtype
    data_fp16 = topi.cast(x, "float16")
    data_abs = topi.abs(data_fp16)

    res_tmp = akg.lang.ascend.reduce_max(data_abs, axis=axis, keepdims=keepdims)
    shape_len = len(x.shape)
    if axis[-1] == shape_len - 1 and not keepdims:
        res_shape = [x.value for x in res_tmp.shape]
        res_shape.pop()
        res_tmp = tvm.compute(res_shape, lambda *indice: res_tmp(*indice, 0), name="reduce_res")
    res_s8 = topi.cast(res_tmp, dtype)
    return res_s8

@utils.check_input_type(akg.tvm.tensor.Tensor, (int, list, tuple, type(None)), (bool, type(None)))
def reduce_any_d(x, axis=None, keepdims=False):
    """
    Reduce a tensor on a certain axis based on max.

    Args:

        x (tvm.tensor.Tensor): The input tensor to reduce. Should be of type int8.
        axis (Union[list, tuple, int, None]): The dimensions to reduce. If None, all dimensions will be reduced.
                                              each dim must be in the range [-len(data.shape), len(data.shape) - 1].
        keepdims (Union[bool, None]): If True, retains reduced dimensions with length 1, defaults to False.

    Returns:
        tvm.tensor.Tensor of same type as input tensor x.
    """
    # check type
    utils.ops_dtype_check(x.dtype, utils.DtypeForDavinci.INT8)
    utils.check_shape(x.shape)
    # check axis
    utils.reduce_axis_check(x.shape, axis)
    refined_axis = refine_reduce_axis(x, axis)
    if len(set(refined_axis)) == len(x.shape) and not keepdims:
        keepdims = True
    res = _reduce_any_d_compute(x, refined_axis, keepdims)
    if len(set(refined_axis)) == len(x.shape):
        res = topi.reshape(res, (1, ))
    return res
