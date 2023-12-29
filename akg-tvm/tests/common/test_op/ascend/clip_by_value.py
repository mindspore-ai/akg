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

"""operator dsl function:clip_by_value"""
import akg.topi
import akg.tvm
import akg.utils as utils
from akg.utils.format_transform import get_shape

@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, (str, type(None)))
def clip_by_value(data, clip_value_min, clip_value_max, target=utils.CCE):
    """
    clip_by_value op.

    ..math:`y=max if(data>max); y=min if(data<min)`

    Args:
        data (tvm.tensor.Tensor): tensor with type int32, float16 or float32.
        clip_value_min (tvm.tensor.Tensor): tensor with type int32, float16 or float32. Should
            be 0-D or has the same shape as data.
        clip_value_max (tvm.tensor.Tensor): tensor with type int32, float16 or float32. Should
            be 0-D or has the same shape as data.
    Returns:
        tvm.tensor.Tensor.
    """
    dtype = data.dtype
    if dtype != clip_value_min.dtype or dtype != clip_value_max.dtype:
        raise TypeError("data dtype and clip_value dtype should be the same")
    utils.ops_dtype_check(dtype, [utils.DtypeForDavinci.ALL_FLOAT, utils.DtypeForDavinci.INT32])

    utils.check_shape(data.shape)
    utils.check_shape(clip_value_min.shape)
    utils.check_shape(clip_value_max.shape)

    shape_min = get_shape(clip_value_min)
    shape_max = get_shape(clip_value_max)
    shape_data = get_shape(data)

    if not len(shape_min) == len(shape_data):
        if len(shape_min) == 1:
            clip_value_min = akg.topi.broadcast_to(clip_value_min, shape_data)
        else:
            raise RuntimeError(
                "clip_min_value's shape is neigther 0-D nor same as data")
    if not len(shape_max) == len(shape_data):
        if len(shape_max) == 1:
            clip_value_max = akg.topi.broadcast_to(clip_value_max, shape_data)
        else:
            raise RuntimeError(
                "clip_max_value's shape is neigther 0-D nor same as data")

    res_max = akg.lang.ascend.vmax(data, clip_value_min)
    res = akg.lang.ascend.vmin(res_max, clip_value_max)

    return res
