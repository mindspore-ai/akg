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

"""operator dsl function: reverse"""
import akg
from akg import tvm
from akg.utils import validation_check as vc_util
from akg.utils.format_transform import get_shape

def _check_axis(axis, shape):
    """double check if axises are valid."""
    shape_tmp = list(shape).copy()
    while shape_tmp[-1] == 1 and len(shape_tmp) > 1:
        shape_tmp.pop()

    if (len(shape_tmp) - 1) in axis or -1 in axis:
        raise RuntimeError("Do not support reverse on last dimension!")

    for i in axis:
        if i not in range(-len(shape_tmp), len(shape_tmp)):
            raise ValueError("Axis is invalid!")

def reverse_compute(input_data, axis):
    """reverse compute implementation."""
    shape = input_data.shape
    axis_flag = [1] * len(shape)
    for i in axis:
        axis_flag[i] = -1

    def _map_index(*index):
        """calculate normal index"""
        begin = [0] * len(shape)
        for i, _ in enumerate(shape):
            if i in axis:
                begin[i] = shape[i] - 1
            if i == 0:
                index_org = (index[i] * axis_flag[i] + begin[i],)
            else:
                index_org = index_org + (index[i] * axis_flag[i] + begin[i],)

        return index_org

    output = tvm.compute(shape, lambda *i: input_data(*_map_index(*i)), name='output')

    return output

@vc_util.check_input_type(akg.tvm.tensor.Tensor, (int, list, tuple))
def reverse(input_data, axis):
    """
    Reverse a tensor on some dimension.
    Args:
        input_data (tvm.tensor.Tensor): Tensor of float16, float32 and int32.
        axis (Union[list, tuple, int]): Because of don't support reverse which contain last dim, so can't equal None.
    Returns:
        tvm.tensor.Tensor,has the same type and shape as input_data
    """
    shape = get_shape(input_data)
    dtype = input_data.dtype
    # check dtype and shape
    vc_util.check_shape(shape)
    vc_util.ops_dtype_check(dtype, [vc_util.DtypeForDavinci.ALL_FLOAT, vc_util.DtypeForDavinci.INT32])
    # check axis
    shape_len = len(shape)
    if hasattr(axis, 'index'):
        axis = list(axis)
    if isinstance(axis, int):
        axis = [axis]
    vc_util.axis_check(shape_len, axis)
    _check_axis(axis, shape)
    # compute res
    res = reverse_compute(input_data, axis)
    return res

