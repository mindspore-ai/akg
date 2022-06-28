# Copyright 2019-2022 Huawei Technologies Co., Ltd
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

"""operator dsl function: conv_backprop_input"""
import logging
import akg
import akg.tvm
import akg.utils as utils
from akg import dim
from akg.ops.math.cast import cast
from akg.utils.validation_check import comp_conv_backprop_out_shape


def get_conv_backprop_input_tiling_args():
    conv_backprop_input_tiling_args = {
        str(((1, 1024, 14, 14), (2048, 1024, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1))): [2, 32, 64, 96, 128],
        str(((1, 1024, 14, 14), (256, 1024, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))): [14, 256, 208, 64, 112],
        str(((1, 1024, 14, 14), (512, 1024, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))): [4, 128, 48, 352, 16, 14],
        str(((1, 1024, 14, 14), (512, 1024, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1))): [14, 512, 49, 32, 512],
        str(((1, 128, 28, 28), (128, 128, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1))): [28, 128, 128, 144, 128],
        str(((1, 128, 28, 28), (512, 128, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))): [28, 128, 784, 16, 32],
        str(((1, 128, 56, 56), (128, 128, 3, 3), (0, 1, 0, 1), (2, 2), (1, 1))): [6, 32, 112, 160, 32, 58],
        str(((1, 16, 224, 224), (64, 16, 7, 7), (3, 3, 3, 3), (2, 2), (1, 1))): [10, 16, 16, 49 * 16, 16, 10],
        str(((1, 2048, 7, 7), (512, 2048, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))): [7, 512, 49, 32, 512],
        str(((1, 256, 13, 13), (384, 256, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1))): [13, 64, 80, 48, 16, 15],
        str(((1, 256, 14, 14), (1024, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))): [14, 256, 112, 32, 240],
        str(((1, 256, 14, 14), (256, 256, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1))): [14, 128, 196, 144, 128],
        str(((1, 256, 28, 28), (256, 256, 3, 3), (0, 1, 0, 1), (2, 2), (1, 1))): [9, 16, 48, 448, 16, 30],
        str(((1, 256, 28, 28), (256, 256, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1))): [14, 128, 196, 144, 128],
        str(((1, 256, 56, 56), (128, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))): [8, 128, 240, 128, 128, 56],
        str(((1, 256, 56, 56), (128, 256, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1))): [7, 128, 252, 64, 128],
        str(((1, 256, 56, 56), (512, 256, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1))): [3, 32, 32, 32, 32],
        str(((1, 256, 56, 56), (64, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))): [16, 64, 280, 16, 64],
        str(((1, 3, 224, 224), (64, 3, 7, 7), (3, 3, 3, 3), (2, 2), (1, 1))): [10, 16, 16, 49 * 16, 16, 10],
        str(((1, 384, 13, 13), (256, 384, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1))): [12, 192, 16, 240, 96, 15],
        str(((1, 384, 13, 13), (384, 384, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1))): [9, 128, 96, 176, 80, 15],
        str(((1, 512, 14, 14), (512, 512, 3, 3), (0, 1, 0, 1), (2, 2), (1, 1))): [10, 16, 80, 64, 16, 16],
        str(((1, 512, 28, 28), (1024, 512, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1))): [2, 64, 112, 32, 512],
        str(((1, 512, 28, 28), (128, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))): [14, 128, 448, 16, 64],
        str(((1, 512, 28, 28), (256, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))): [10, 256, 128, 32, 256, 28],
        str(((1, 512, 28, 28), (256, 512, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1))): [7, 256, 98, 64, 256],
        str(((1, 512, 7, 7), (2048, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))): [7, 128, 49, 256, 128],
        str(((1, 512, 7, 7), (512, 512, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1))): [7, 64, 49, 144, 64],
        str(((1, 6, 14, 14), (16, 6, 5, 5), (0, 0, 0, 0), (1, 1), (1, 1))): [18, 16, 64, 240, 16, 18],
        str(((1, 64, 56, 56), (256, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))): [14, 256, 784, 16, 32],
        str(((1, 64, 56, 56), (64, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))): [56, 64, 784, 16, 32],
        str(((1, 64, 56, 56), (64, 64, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1))): [56, 64, 128, 144, 64],
        str(((1, 96, 28, 28), (256, 96, 5, 5), (2, 2, 2, 2), (1, 1), (1, 1))): [14, 48, 32, 384, 48, 32],
        str(((32, 1024, 14, 14), (2048, 1024, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1))): [2, 224, 32, 32, 144, 14],
        str(((32, 1024, 14, 14), (256, 1024, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))): [14, 224, 192, 64, 48, 14],
        str(((32, 1024, 14, 14), (512, 1024, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))): [7, 352, 96, 80, 176, 14],
        str(((32, 1024, 14, 14), (512, 1024, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1))): [14, 512, 49, 32, 512],
        str(((32, 128, 28, 28), (128, 128, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1))): [18, 64, 208, 144, 64, 30],
        str(((32, 128, 28, 28), (512, 128, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))): [4, 384, 112, 16, 336, 28],
        str(((32, 128, 56, 56), (128, 128, 3, 3), (0, 1, 0, 1), (2, 2), (1, 1))): [6, 112, 112, 144, 112, 58],
        str(((32, 16, 224, 224), (64, 16, 7, 7), (3, 3, 3, 3), (2, 2), (1, 1))): [10, 16, 16, 49 * 16, 16, 10],
        str(((32, 2048, 7, 7), (512, 2048, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))): [7, 32, 48, 272, 32, 7],
        str(((32, 256, 13, 13), (384, 256, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1))): [13, 64, 80, 48, 16, 15],
        str(((32, 256, 14, 14), (1024, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))): [2, 416, 32, 752, 16, 14],
        str(((32, 256, 14, 14), (256, 256, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1))): [16, 112, 112, 144, 112, 14],
        str(((32, 256, 28, 28), (256, 256, 3, 3), (0, 1, 0, 1), (2, 2), (1, 1))): [6, 144, 112, 144, 112, 30],
        str(((32, 256, 28, 28), (256, 256, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1))): [14, 128, 196, 144, 128],
        str(((32, 256, 56, 56), (128, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))): [4, 128, 224, 64, 112, 56],
        str(((32, 256, 56, 56), (128, 256, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1))): [8, 128, 224, 96, 48, 56],
        str(((32, 256, 56, 56), (512, 256, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1))): [2, 288, 112, 144, 32, 56],
        str(((32, 256, 56, 56), (64, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))): [8, 64, 448, 64, 32, 56],
        str(((32, 3, 224, 224), (64, 3, 7, 7), (2, 3, 2, 3), (2, 2), (1, 1))): [13, 16, 16, 49 * 16, 16, 13],
        str(((32, 3, 224, 224), (64, 3, 7, 7), (3, 3, 3, 3), (2, 2), (1, 1))): [10, 16, 16, 49 * 16, 16, 10],
        str(((32, 3, 32, 32), (6, 3, 5, 5), (0, 0, 0, 0), (1, 1), (1, 1))): [16, 16, 16, 16, 16, 16],
        str(((32, 384, 13, 13), (256, 384, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1))): [12, 192, 16, 240, 96, 15],
        str(((32, 384, 13, 13), (384, 384, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1))): [9, 128, 96, 176, 80, 15],
        str(((32, 512, 14, 14), (512, 512, 3, 3), (0, 1, 0, 1), (2, 2), (1, 1))): [10, 96, 112, 144, 48, 16],
        str(((32, 512, 28, 28), (1024, 512, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1))): [2, 336, 64, 80, 208, 28],
        str(((32, 512, 28, 28), (128, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))): [4, 64, 112, 80, 64, 28],
        str(((32, 512, 28, 28), (256, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))): [8, 224, 112, 64, 96, 28],
        str(((32, 512, 28, 28), (256, 512, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1))): [7, 192, 96, 48, 192, 28],
        str(((32, 512, 7, 7), (2048, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))): [7, 128, 49, 256, 128],
        str(((32, 512, 7, 7), (512, 512, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1))): [9, 80, 64, 144, 80, 9],
        str(((32, 6, 14, 14), (16, 6, 5, 5), (0, 0, 0, 0), (1, 1), (1, 1))): [18, 16, 64, 240, 16, 18],
        str(((32, 64, 56, 56), (256, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))): [4, 224, 112, 224, 80, 56],
        str(((32, 64, 56, 56), (64, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))): [10, 64, 336, 16, 16, 56],
        str(((32, 64, 56, 56), (64, 64, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1))): [15, 64, 112, 144, 64, 58],
        str(((32, 96, 27, 27), (256, 96, 5, 5), (2, 2, 2, 2), (1, 1), (1, 1))): [7, 32, 80, 48, 32, 31],
        str(((32, 96, 28, 28), (256, 96, 5, 5), (2, 2, 2, 2), (1, 1), (1, 1))): [14, 48, 32, 384, 48, 32],
    }
    return conv_backprop_input_tiling_args


def get_cast_tiling_args():
    cast_tiling_args = {
        str(((1, 1024, 14, 14), (2048, 1024, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1))): [2, 16, 64, 96, 128],
        str(((1, 1024, 14, 14), (256, 1024, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))): [14, 256 // 2, 208, 64, 112],
        str(((1, 1024, 14, 14), (512, 1024, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1))): [14, 16, 50, 32, 512],
        str(((1, 128, 28, 28), (128, 128, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1))): [28, 128 // 4, 128, 144, 128],
        str(((1, 128, 28, 28), (512, 128, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))): [28, 128 // 2, 784, 16, 32],
        str(((1, 2048, 7, 7), (512, 2048, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))): [7, 512 // 8, 49, 32, 512],
        str(((1, 256, 14, 14), (1024, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))): [14, 256 // 8, 112, 32, 240],
        str(((1, 256, 14, 14), (256, 256, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1))): [14, 128 // 8, 196, 144, 128],
        str(((1, 256, 56, 56), (128, 256, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1))): [7, 128, 252, 64, 128],
        str(((1, 256, 56, 56), (64, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))): [16, 64, 280, 16, 64],
        str(((1, 3, 224, 224), (64, 3, 7, 7), (3, 3, 3, 3), (2, 2), (1, 1))): [10, 16, 16, 49 * 16, 16, 10],
        str(((1, 16, 224, 224), (64, 16, 7, 7), (3, 3, 3, 3), (2, 2), (1, 1))): [10, 16, 16, 49 * 16, 16, 10],
        str(((1, 512, 28, 28), (128, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))): [14, 128, 448, 16, 64],
        str(((1, 512, 28, 28), (256, 512, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1))): [7, 256 // 4, 98, 64, 256],
        str(((1, 512, 7, 7), (2048, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))): [7, 128 // 8, 49, 256, 128],
        str(((1, 512, 7, 7), (512, 512, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1))): [7, 64 // 4, 49, 144, 64],
        str(((1, 64, 56, 56), (256, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))): [14, 256 // 8, 784, 16, 32],
        str(((1, 64, 56, 56), (64, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))): [56, 64, 784, 16, 32],
        str(((1, 64, 56, 56), (64, 64, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1))): [56, 64, 128, 144, 64],
        str(((1, 256, 56, 56), (512, 256, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1))): [3, 16, 32, 32, 32],
        str(((1, 512, 28, 28), (1024, 512, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1))): [2, 16, 112, 32, 512],
    }
    return cast_tiling_args


def gen_key(fmap_shape, filter_shape, pad_, stride_, dilation_):
    """generate key."""
    key = str((tuple(fmap_shape), tuple(filter_shape),
              tuple(pad_), tuple(stride_), tuple(dilation_)))
    return key


def conv_backprop_input_compute(data, output_shape, filter_shape, input_shape, pad_, stride_,
                                block_size=16, attrs=None, key=None):
    """core computation of conv_backprop_input."""
    _, in_c, w_h, w_w = filter_shape

    stride_h, stride_w = stride_
    if stride_h != stride_w:
        raise ValueError("stride_h must be equal to stride_w.")

    # output shape (NCHW -> NC1HWC0)
    in_nn, in_cc, in_hh, in_ww = output_shape
    if in_c % block_size != 0:
        raise ValueError("in_c must be divided by block_size.")
    input_shape_nc1hwc0 = (in_nn, in_cc // block_size, in_hh, in_ww, block_size)
    in_nn, _, in_hh, in_ww, _ = input_shape_nc1hwc0
    input_trans_shape_nc1hwc0 = (in_nn, in_cc // block_size, in_hh * stride_h, in_ww * stride_w, block_size)
    in_n, in_c1, in_h, in_w, _ = input_trans_shape_nc1hwc0

    # kernel shape (NCHW -> NC1HWC0 -> Fractal)
    k_n, k_c, k_h, k_w = filter_shape
    if k_c % block_size != 0:
        raise ValueError("k_c must be divided by block_size.")
    kernel_shape_nc1hwc0 = (k_n, k_c // block_size, k_h, k_w, block_size)
    k_n, k_c1, k_h, k_w, k_c0 = kernel_shape_nc1hwc0
    kernel_shape_trans = (k_n // block_size * k_h * k_w, k_c // block_size, block_size, block_size)
    k_c1 = k_n // block_size
    k_n = k_c

    _, _, input_h, input_w = input_shape

    # padding ((padding_h, padding_w) -> (padding_top, padding_bottom, padding_left, padding_right))
    padding = (pad_[0], pad_[1], pad_[2], pad_[3])
    pad_t, pad_b, pad_l, pad_r = padding

    # padHT -> padHT'
    p_top = k_h - pad_t - 1
    # padHB -> padHB'
    p_bottom = input_h + pad_t - stride_h * ((input_h + pad_t + pad_b - k_h) // stride_h + 1)
    # padWL -> padWL'
    p_left = k_w - pad_l - 1
    # padWR -> padWR'
    p_right = input_w + pad_l - stride_w * ((input_w + pad_l + pad_r - k_w) // stride_w + 1)

    s_h = 1
    s_w = 1

    # NC1HWCO
    a_value = data[0]

    if data[1].dtype == 'float32':
        b_value = cast(data[1], 'float16', utils.CCE)
        tiling_args = get_cast_tiling_args()
    else:
        b_value = data[1]
        tiling_args = get_conv_backprop_input_tiling_args()

    # Create reduction variables
    kc1 = akg.tvm.reduce_axis((0, k_c1), name='kc1')
    kh = akg.tvm.reduce_axis((0, k_h), name='kh')
    kw = akg.tvm.reduce_axis((0, k_w), name='kw')
    kc0 = akg.tvm.reduce_axis((0, k_c0), name='kc0')
    use_auto_tiling = False
    if attrs is not None and 'conv_tile' in attrs and len(attrs['conv_tile']) >= 5:
        tile_value = attrs['conv_tile']
    elif key in tiling_args:
        tile_value = tiling_args[key]
    else:
        use_auto_tiling = True

    out_h = (in_h + p_top + p_bottom - k_h) // (s_h) + 1
    out_w = (in_w + p_left + p_right - k_w) // (s_w) + 1
    out_shape_nc1hwc0 = (in_n, k_n // block_size, out_h, out_w, block_size)
    out_n, out_c1, out_h, out_w, out_c0 = out_shape_nc1hwc0

    # set dim
    info = dim.Dim()
    index_ = 0

    if not use_auto_tiling:
        tile_hh = tile_value[0]
        if tile_hh == input_h:
            tile_hh += pad_t + pad_b

        tile_coco = tile_value[1]
        tile_coco = (tile_coco + block_size - 1) // block_size * block_size

        tile_mm = tile_value[2]
        tile_mm = (tile_mm + block_size - 1) // block_size * block_size

        tile_kk = tile_value[3]
        if not tile_kk % (block_size * w_h * w_w) == 0:
            logging.warning(
                "Warning: tile_k must be a multiple of (block_size * w_h * w_w)")
        tile_kk = (tile_kk + block_size * w_h * w_w - 1) // (block_size * w_h * w_w) * (block_size * w_h * w_w)

        tile_nn = tile_value[4]
        tile_nn = (tile_nn + block_size - 1) // block_size * block_size

        tile_ww = input_w
        if len(tile_value) >= 6 and tile_value[5] > 0:
            tile_ww = tile_value[5]
        if tile_ww == input_w:
            tile_ww += pad_l + pad_r

        if tile_hh == in_h:
            tile_hh += p_top + p_bottom
        tile_out_h = (tile_hh - k_h) // s_h + 1

        if tile_ww == in_w:
            tile_ww += p_left + p_right
        tile_out_w = (tile_ww - k_w) // s_w + 1

        if tile_coco > 0:
            c1_cut = tile_coco // block_size
        else:
            c1_cut = out_c1

        if out_n > 1:
            info.setdim(index=index_, axis=0, tilel1=1, tilel0=0)  # n
        if out_c1 > 1:
            info.setdim(index=index_, axis=1, tilel1=c1_cut, tilel0=0)  # c1
        if out_h > 1:
            info.setdim(index=index_, axis="H", tilel1=tile_out_h, tilel0=0)  # h
        if out_w > 1:
            info.setdim(index=index_, axis="W", tilel1=tile_out_w, tilel0=0)  # w
        if out_c0 > 1:
            info.setdim(index=index_, axis=4, tilel1=out_c0, tilel0=0)  # c0
        if in_c1 > 1:
            info.setdim(index=index_, axis=5, tilel1=in_c1, tilel0=0)  # kc1
        if k_h > 1:
            info.setdim(index=index_, axis=5, tilel1=k_h, tilel0=0)  # kh
        if k_w > 1:
            info.setdim(index=index_, axis=5, tilel1=k_w, tilel0=0)  # kw

        info = str(info)
    else:
        info = ""
    # Compute the convolution below

    output_name = "output0"

    # assume that the weight has index [ko, no, ni, ki]
    # write it in 6d format [ co_1, kh, kw, ci_1, ci_0, co_0 ]
    # where kernel weight is kw = ko % k_w, and kernel height is kh = ko // k_w % k_h
    # out channel is co_1 = ko // k_w // k_h and inner channel is ci_1 = no
    # after flipping and transpoing
    # the index of weight is [ ci_1, kh', kw', co_1, co_0, ci_0 ]
    # write it back to 4d format, the index of weight
    # is given by [ no, k_h - ko // k_w % k_h - 1, k_w - ko % k_w - 1, ko // k_w // k_h, co_0, ci_0 ]
    b_trans = akg.tvm.compute(kernel_shape_trans,
                              lambda ko, no, ni, ki: b_value[((no * k_h + k_h - 1 - ko // k_w % k_h)
                                                              * k_w + k_w - 1 - ko % k_w), ko // (k_h * k_w), ki, ni],
                              name='B_trans')

    if ((stride_h > 1) or (stride_w > 1)):
        @akg.tvm.hybrid.script
        def data_trans_hybrid(output, inputs, const_zero):
            """Implements data_trans ( B[n, c1, h * strideH, w * strideW, c0] = A[n, c1, h, w, c0] )."""

            stride_h = output.shape[2] // inputs.shape[2]
            stride_w = output.shape[3] // inputs.shape[3]

            b = allocate(output.shape, output.dtype, 'local')
            for n in range(output.shape[0]):
                for c1 in range(output.shape[1]):
                    for h in range(output.shape[2]):
                        for w in range(output.shape[3]):
                            for c0 in range(output.shape[4]):
                                b[n, c1, h, w, c0] = const_zero
                                if h % stride_h == 0 and w % stride_w == 0:
                                    b[n, c1, h, w, c0] = inputs[n, c1, h // stride_h, w // stride_w, c0]

            return b

        a_trans_init = akg.tvm.placeholder(
            input_trans_shape_nc1hwc0, dtype="float16", name='a_trans')
        const_zero = akg.tvm.const(0, 'float16')
        a_trans = data_trans_hybrid(a_trans_init, a_value, const_zero)
    else:
        a_trans = a_value
    conv_attrs = {
        "pragma_conv_kernel_n": k_n,
        "pragma_conv_kernel_h": k_h,
        "pragma_conv_kernel_w": k_w,
        "pragma_conv_padding_top": p_top,
        "pragma_conv_padding_bottom": p_bottom,
        "pragma_conv_padding_left": p_left,
        "pragma_conv_padding_right": p_right,
        "pragma_conv_bypass_l1": 0,
        "pragma_conv_backprop_input": 1,
        "pragma_conv_stride_h": s_h,
        "pragma_conv_stride_w": s_w,
        "pragma_conv_dilation_h": 1,
        "pragma_conv_dilation_w": 1,
        "pragma_conv_fm_n": in_n,
        "pragma_conv_fm_c": in_c,
        "pragma_conv_fm_h": in_h,
        "pragma_conv_fm_w": in_w,
        "feature": a_trans.op.name,
        "filter": b_value.op.name,
        "bias": 'None',
        "res": output_name}
    if not use_auto_tiling:
        conv_attrs["pragma_conv_h_cut"] = (tile_out_h - 1) * s_h + k_h
        conv_attrs["pragma_conv_w_cut"] = (tile_out_w - 1) * s_w + k_w
        conv_attrs["pragma_conv_co_cut"] = c1_cut * k_c0
        conv_attrs["pragma_conv_m_cut"] = tile_mm
        conv_attrs["pragma_conv_k_cut"] = tile_kk
        conv_attrs["pragma_conv_n_cut"] = tile_nn
    res_c = akg.tvm.compute(out_shape_nc1hwc0,
                            lambda n, c1, h, w, c0: akg.lang.ascend.mmad(
                                (akg.tvm.if_then_else(akg.tvm.any((h * s_h + kh) < p_top,
                                                                  (h * s_h + kh) > (in_h + p_top - 1),
                                                                  (w * s_w + kw) < p_left,
                                                                  (w * s_w + kw) > (in_w + p_left - 1)),
                                                      akg.tvm.const(0.0, 'float16'),
                                                      a_trans[n, kc1, (h * s_h + kh - p_top),
                                                              (w * s_w + kw - p_left), kc0])
                                 * b_trans[(kc1 * k_h + kh) * k_w + kw, c1, c0, kc0]).astype("float32"),
                                axis=[kc1, kh, kw, kc0]), name=output_name,
                            attrs=conv_attrs)

    res_c = cast(res_c, "float16", utils.CCE)

    return res_c, {"dim": info, "pragma_rmselfdep": 0}


@utils.check_input_type((list, tuple), (list, tuple), (list, tuple), (list, tuple), (list, tuple), (list, tuple),
                        (dict, type(None)), (str, type(None)))
def conv_backprop_input(data, fmap_shape, filter_shape, pad_, stride_, dilation_, attrs=None):
    """
    Computes dx according "conv forward".

    Args:
        data (list[tvm.tensor.Tensor]): a list with length 2.
              data[0](consider as dy) Tensor of type float16 ,shape 5D(out_n, out_c//C0, out_h, out_w,C0)
              data[1](consider as w)  Tensor of type float16 ,shape 4D(wC//C0*wH*wW, wN//C0, C0,C0)
        fmap_shape (list[int]): [fN, fC, fH, fW]
        filter_shape (list[int]): [wN, wC, wH, wW]
        pad_ (list[int]): [pad_left, pad_right, pad_top, pad_bottom]
        stride_ (list[int]): [stride_h, stride_w]
        dilation_ (list[int]): [dilation_h, dilation_w]
        attrs (dict): a dict with keys like conv_tile,bypass and etc.

    Returns:
        tvm.tensor.Tensor.
        configs.

    Supported Platforms:
        'Ascend'
    """

    if len(data) != 2:
        raise IndexError("data contains output tensor and filter tensor")

    block_size = 16
    x_shape, _, w_shape = comp_conv_backprop_out_shape(fmap_shape, filter_shape, pad_, stride_, dilation_)

    key = gen_key(fmap_shape, filter_shape, pad_, stride_, dilation_)
    res_c, configs = conv_backprop_input_compute(data, x_shape, w_shape, fmap_shape, pad_, stride_,
                                                 block_size=block_size, attrs=attrs, key=key)

    return res_c, configs
