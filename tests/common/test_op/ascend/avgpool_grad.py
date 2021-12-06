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

"""operator dsl function: avgpool_grad"""

import akg.tvm
import akg.utils as  utils
from akg.utils.dsl_create import cal_pad_shapes_by_strategy
from akg.utils import custom_tiling as ct_util
from akg.tvm.hybrid import script
from akg.utils.format_transform import get_shape
from akg.dim import DIM

set_dim_map_ = {
    str(((10, 3, 16, 16, 16), (4, 4), (3, 3), 'VALID', 'float16')): (
        (2, 2), (3, 3), (16, 16), (5, 5), (4, 4)),
    str(((10, 3, 16, 16, 16), (4, 4), (3, 3), 'SAME', 'float16')): (
        (2, 2), (3, 3), (16, 16), (6, 6), (4, 4)),
    str(((10, 3, 16, 16, 16), (4, 4), (3, 3), (0, 0, 0, 0), 'float16')): (
        (2, 2), (3, 3), (16, 16), (5, 5), (4, 4)),
    str(((1, 3, 64, 64, 16), (4, 4), (3, 3), 'VALID', 'float16')): (
        (1, 1), (16, 16), (21, 21), (4, 4)),
    str(((1, 3, 64, 64, 16), (4, 4), (3, 3), 'SAME', 'float16')): (
        (1, 1), (1, 1), (22, 22), (4, 4), (64, 64)),
    str(((1, 3, 64, 64, 16), (4, 4), (3, 3), (0, 0, 0, 0), 'float16')): (
        (1, 1), (16, 16), (21, 21), (4, 4)),
    str(((1, 8, 100, 200, 16), (100, 200), (1, 1), 'VALID', 'float16')): (
        (1, 1), (1, 1), (10, 10), (200, 200)),
    str(((1, 8, 100, 200, 16), (100, 200), (1, 1), 'SAME', 'float16')): (
        (1, 1), (1, 1), (10, 10), (200, 200)),
    str(((1, 8, 100, 200, 16), (100, 200), (1, 1), (0, 0, 0, 0), 'float16')): (
        (1, 1), (1, 1), (10, 10), (200, 200)),
}


def set_dim_func_(x, dy, kernel, stride, pad):
    """dim function"""
    key = []
    key.append(tuple([t.value for t in x.shape]))
    key.append(kernel)
    key.append(stride)
    if isinstance(pad, list):
        pad = tuple(pad)
    key.append(pad)
    key.append(x.dtype)
    hash_key = str(tuple(key))

    if hash_key in set_dim_map_.keys():
        return ct_util.set_dims(set_dim_map_[hash_key]), hash_key
    else:
        return "", hash_key




@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor,
                          (list, tuple), (list, tuple), (str, list, tuple), (str, type(None)))
def avgpool_grad(x, dy, kernel, stride, pad, target=utils.CCE):
    """
    Gradient for avgpool.

    Args:
        x (tvm.tensor.Tensor): Forward input tensor of type float16.
        dy (tvm.tensor.Tensor): Gradient for forward output of type float16.
        kernel (Union[list, tuple]): Two int numbers for window size of H and W for pooling.
        stride (Union[list, tuple]): Two int numbers for stride size of H and W for pooling.
        pad (Union[str, list, tuple]): Padding strategy for pooling.

    Returns:
        Gradient of forward input tensor.
    """
    dtype = x.dtype
    utils.ops_dtype_check(dtype, utils.DtypeForDavinci.FLOAT16)
    shape = get_shape(x)
    utils.check_shape(shape)

    if len(shape) != 5:
        raise RuntimeError("Only support 5-dim pooling!")
    if shape[-1] % 16 != 0:
        raise RuntimeError("Last shape must be divisible by 16!")
    if len(kernel) != 2:
        raise RuntimeError("Only support 2-dim kernel!")
    if len(stride) != 2:
        raise RuntimeError("Only support 2-dim stride!")
    if isinstance(pad, (list, tuple)) and len(pad) != 4:
        raise RuntimeError("Only support string or list/tuple of 4 int numbers!")

    dim_info, _ = set_dim_func_(x, dy, kernel, stride, pad)
    attrs = {DIM: dim_info}

    @script
    def grad(zero, one_div_ksize, x, dy, kh, kw, sh, sw, ph_h, ph_t, pw_h, pw_t):
        tmpdx = allocate(
            (x.shape[0], x.shape[1],
             x.shape[2] + ph_h + ph_t, x.shape[3] + pw_h + pw_t, x.shape[4]),
            x.dtype)
        dy_tmp = allocate(dy.shape, dy.dtype)
        dx = output_tensor(x.shape, x.dtype)

        for n in range(tmpdx.shape[0]):
            for c1 in range(tmpdx.shape[1]):
                for h in range(tmpdx.shape[2]):
                    for w in range(tmpdx.shape[3]):
                        for c0 in range(tmpdx.shape[4]):
                            tmpdx[n, c1, h, w, c0] = zero

        for n in range(dy.shape[0]):
            for c1 in range(dy.shape[1]):
                for i in range(dy.shape[2]):
                    for j in range(dy.shape[3]):
                        for c0 in range(dy.shape[4]):
                            dy_tmp[n, c1, i, j, c0] = dy[n, c1, i, j, c0] * one_div_ksize
                            for ah in range(kh):
                                for aw in range(kw):
                                    if dy.shape[2] == 1 and dy.shape[3] == 1:
                                        tmpdx[n, c1, i * sh + ah, j * sw + aw, c0] = dy_tmp[n, c1, i, j, c0]
                                    else:
                                        tmpdx[n, c1, i * sh + ah, j * sw + aw, c0] = \
                                            tmpdx[n, c1, i * sh + ah, j * sw + aw, c0] + dy_tmp[n, c1, i, j, c0]

        if ph_h > 0 or ph_t > 0 or pw_h > 0 or pw_t > 0:
            for n in range(dx.shape[0]):
                for c1 in range(dx.shape[1]):
                    for h in range(dx.shape[2]):
                        for w in range(dx.shape[3]):
                            for c0 in range(dx.shape[4]):
                                dx[n, c1, h, w, c0] = tmpdx[n, c1, h + ph_h, w + pw_h, c0]
            return dx
        else:
            return tmpdx

    kh, kw = kernel
    sh, sw = stride

    [ph_h, ph_t, pw_h, pw_t], _ = cal_pad_shapes_by_strategy(
        shape, kernel, stride, pad)

    zero = akg.tvm.const(0.0, dtype=dtype)
    one_div_ksize = akg.tvm.const(1.0 / (kh * kw), dtype=dtype)
    params = [kh, kw, sh, sw, ph_h, ph_t, pw_h, pw_t]
    output = grad(zero, one_div_ksize, x, dy, *tuple(akg.tvm.convert(i) for i in params))

    attrs["loop_partition_unroll"] = 1
    return output, attrs
