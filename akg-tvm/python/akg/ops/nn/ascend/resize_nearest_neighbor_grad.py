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

"""operator dsl function: resize_nearest_neighbor_grad"""
import akg.tvm as tvm
import akg.utils as utils
from akg.topi import tag

def ResizeNearestNeighborGrad(grad, size, align_corners=True, out_dtype=None, target=utils.CCE):
    """
    Perform resize_nearest_neighbor_grad.

    Supported Platforms:
        'Ascend'
    """

    in_n, in_c, in_h, in_w = grad.shape
    output_shape = [in_n, in_c, size[0], size[1]]

    if align_corners:
        y_ratio = (in_h - 1).astype('float') / (size[0] - 1)
        x_ratio = (in_w - 1).astype('float') / (size[1] - 1)
    else:
        y_ratio = (in_h).astype('float') / (size[0])
        x_ratio = (in_w).astype('float') / (size[1])

    def _get_pixel(n, c, y, x):
        y = tvm.max(tvm.min(y, in_h - 1), 0)
        x = tvm.max(tvm.min(x, in_w - 1), 0)
        return grad(n, c, y, x).astype('float')

    def _get_indices(*indices):
        n, c, y, x = indices    
        return n, c, y, x

    def _cast_output(value):
        if out_dtype:
            dtype = out_dtype
        else:
            dtype = grad.dtype
        return value.astype(dtype)

    # Nearest neighbor computation
    def _nearest_neighbor_grad(*indices):
        n, c, y, x = _get_indices(*indices)

        in_y = y_ratio * y
        in_x = x_ratio * x

        if align_corners:
            yint = tvm.round(in_y).astype('int32')
            xint = tvm.round(in_x).astype('int32')
        else:
            # Add epsilon to floor to prevent gpu rounding errors.
            epsilon = 1e-5
            yint = tvm.floor(in_y + epsilon).astype('int32')
            xint = tvm.floor(in_x + epsilon).astype('int32')
        return _cast_output(_get_pixel(n, c, yint, xint))
 
    compute_func = _nearest_neighbor_grad

    return tvm.compute(output_shape, compute_func, name='resize_nearest_neighbor_grad', tag=tag.INJECTIVE)
