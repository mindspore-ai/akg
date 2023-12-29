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

"""operator dsl function: conv_bn1"""
from functools import reduce
import akg.topi
import akg.tvm
import akg
from akg.ops.math.cast  import cast
from akg.ops.nn.ascend.conv import conv_core
from akg.ops.nn.ascend.conv import conv_set_dim_func
import akg.utils as utils

conv_bn1_set_dim_map = {
    str(((1, 1024, 14, 14), (2048, 1024, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False)):
        ([14, 2048, 64, 96, 128], {"bypass": 1}),
    str(((1, 1024, 14, 14), (256, 1024, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([14, 256, 208, 64, 112], {"bypass": 1}),
    str(((1, 1024, 14, 14), (512, 1024, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False)):
        ([13, 144, 48, 48, 128, 13], {"bypass": 0}),
    str(((1, 128, 28, 28), (128, 128, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False)):
        ([30, 128, 240, 48, 64, 30], {"bypass": 0}),
    str(((1, 128, 28, 28), (512, 128, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([8, 256, 112, 16, 48, 28], {"bypass": 0}),
    str(((1, 2048, 7, 7), (512, 2048, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([7, 160, 48, 48, 96, 7], {"bypass": 0}),
    str(((1, 256, 14, 14), (1024, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([14, 256, 48, 64, 256, 14], {"bypass": 0}),
    str(((1, 256, 14, 14), (256, 256, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False)):
        ([14, 192, 64, 128, 160, 16], {"bypass": 0}),
    str(((1, 256, 56, 56), (128, 256, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False)):
        ([7, 128, 112, 48, 16, 55], {"bypass": 0}),
    str(((1, 256, 56, 56), (64, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([6, 64, 224, 32, 16, 56], {"bypass": 0}),
    str(((1, 3, 224, 224), (64, 3, 7, 7), (2, 3, 2, 3), (2, 2), (1, 1), False)):
        ([97, 64, 128, 128, 64, 229], {"bypass": 0}),
    str(((1, 512, 28, 28), (128, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([28, 64, 48, 304, 32, 28], {"bypass": 0}),
    str(((1, 512, 28, 28), (256, 512, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False)):
        ([27, 192, 64, 48, 160, 27], {"bypass": 0}),
    str(((1, 512, 7, 7), (2048, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([7, 128, 48, 176, 80, 7], {"bypass": 0}),
    str(((1, 512, 7, 7), (512, 512, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False)):
        ([9, 512, 64, 128, 96, 9], {"bypass": 1}),
    str(((1, 64, 56, 56), (256, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([56, 256, 392, 16, 32], {"bypass": 1}),
    str(((1, 64, 56, 56), (64, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([32, 64, 224, 32, 64, 56], {"bypass": 0}),
    str(((1, 64, 56, 56), (64, 64, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False)):
        ([10, 64, 224, 48, 48, 58], {"bypass": 1}),
    str(((1, 256, 56, 56), (512, 256, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False)):
        ([7, 320, 112, 160, 48, 55], {"bypass": 0}),
    str(((1, 512, 28, 28), (1024, 512, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False)):
        ([13, 496, 96, 176, 144, 27], {"bypass": 0}),
    str(((1, 256, 56, 56), (128, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([12, 128, 112, 64, 128, 56], {"bypass": 0}),
    str(((1, 512, 28, 28), (256, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([28, 176, 224, 112, 80, 28], {"bypass": 0}),
    str(((1, 1024, 14, 14), (512, 1024, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([7, 384, 96, 48, 224, 14], {"bypass": 0}),
    str(((1, 128, 56, 56), (128, 128, 3, 3), (0, 1, 0, 1), (2, 2), (1, 1), False)):
        ([37, 128, 224, 96, 96, 57], {"bypass": 0}),
    str(((1, 256, 28, 28), (256, 256, 3, 3), (0, 1, 0, 1), (2, 2), (1, 1), False)):
        ([29, 256, 80, 224, 144, 29], {"bypass": 1}),
    str(((1, 512, 14, 14), (512, 512, 3, 3), (0, 1, 0, 1), (2, 2), (1, 1), False)):
        ([15, 512, 64, 64, 272, 15], {"bypass": 1}),

    str(((2, 1024, 14, 14), (2048, 1024, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False)):
        ([13, 112, 48, 176, 80, 13], {"bypass": 0}),
    str(((2, 1024, 14, 14), (256, 1024, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([14, 128, 48, 48, 64, 14], {"bypass": 0}),
    str(((2, 1024, 14, 14), (512, 1024, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False)):
        ([13, 144, 48, 48, 128, 13], {"bypass": 0}),
    str(((2, 128, 28, 28), (128, 128, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False)):
        ([30, 128, 240, 48, 64, 30], {"bypass": 0}),
    str(((2, 128, 28, 28), (512, 128, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([8, 256, 112, 16, 48, 28], {"bypass": 0}),
    str(((2, 2048, 7, 7), (512, 2048, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([7, 160, 48, 48, 96, 7], {"bypass": 0}),
    str(((2, 256, 14, 14), (1024, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([14, 256, 48, 64, 256, 14], {"bypass": 0}),
    str(((2, 256, 14, 14), (256, 256, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False)):
        ([14, 192, 64, 128, 160, 16], {"bypass": 0}),
    str(((2, 256, 56, 56), (128, 256, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False)):
        ([7, 128, 112, 48, 16, 55], {"bypass": 0}),
    str(((2, 256, 56, 56), (64, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([6, 64, 224, 32, 16, 56], {"bypass": 0}),
    str(((2, 3, 224, 224), (64, 3, 7, 7), (2, 3, 2, 3), (2, 2), (1, 1), False)):
        ([97, 64, 128, 128, 64, 229], {"bypass": 0}),
    str(((2, 512, 28, 28), (128, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([28, 64, 48, 304, 32, 28], {"bypass": 0}),
    str(((2, 512, 28, 28), (256, 512, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False)):
        ([27, 192, 64, 48, 160, 27], {"bypass": 0}),
    str(((2, 512, 7, 7), (2048, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([7, 128, 48, 176, 80, 7], {"bypass": 0}),
    str(((2, 512, 7, 7), (512, 512, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False)):
        ([9, 512, 64, 128, 96, 9], {"bypass": 1}),
    str(((2, 64, 56, 56), (256, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([56, 256, 784, 16, 32], {"bypass": 1}),
    str(((2, 64, 56, 56), (64, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([32, 64, 224, 32, 64, 56], {"bypass": 0}),
    str(((2, 64, 56, 56), (64, 64, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False)):
        ([10, 64, 224, 48, 48, 58], {"bypass": 1}),
    str(((2, 256, 56, 56), (512, 256, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False)):
        ([7, 320, 112, 160, 48, 55], {"bypass": 0}),
    str(((2, 512, 28, 28), (1024, 512, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False)):
        ([13, 496, 96, 176, 144, 27], {"bypass": 0}),
    str(((2, 256, 56, 56), (128, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([12, 128, 224, 64, 128, 56], {"bypass": 0}),
    str(((2, 512, 28, 28), (256, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([28, 176, 224, 112, 80, 28], {"bypass": 0}),
    str(((2, 1024, 14, 14), (512, 1024, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([7, 384, 96, 48, 224, 14], {"bypass": 0}),
    str(((2, 128, 56, 56), (128, 128, 3, 3), (0, 1, 0, 1), (2, 2), (1, 1), False)):
        ([37, 128, 224, 96, 96, 57], {"bypass": 0}),
    str(((2, 256, 28, 28), (256, 256, 3, 3), (0, 1, 0, 1), (2, 2), (1, 1), False)):
        ([29, 256, 80, 224, 144, 29], {"bypass": 1}),
    str(((2, 512, 14, 14), (512, 512, 3, 3), (0, 1, 0, 1), (2, 2), (1, 1), False)):
        ([15, 512, 64, 64, 272, 15], {"bypass": 1}),

    str(((32, 1024, 14, 14), (2048, 1024, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False)):
        ([13, 64, 48, 128, 64, 13], {"bypass": 0}),
    str(((32, 1024, 14, 14), (256, 1024, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([14, 128, 48, 48, 64, 14], {"bypass": 0}),
    str(((32, 1024, 14, 14), (512, 1024, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False)):
        ([13, 144, 48, 48, 128, 13], {"bypass": 0}),
    str(((32, 128, 28, 28), (128, 128, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False)):
        ([30, 128, 240, 48, 64, 30], {"bypass": 0}),
    str(((32, 128, 28, 28), (512, 128, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([8, 256, 224, 80, 48, 28], {"bypass": 0}),
    str(((32, 2048, 7, 7), (512, 2048, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([7, 128, 48, 112, 112, 7], {"bypass": 0}),
    str(((32, 256, 14, 14), (1024, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([14, 256, 48, 64, 256, 14], {"bypass": 0}),
    str(((32, 256, 14, 14), (256, 256, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False)):
        ([16, 96, 80, 96, 96, 16], {"bypass": 0}),
    str(((32, 256, 56, 56), (128, 256, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False)):
        ([23, 128, 112, 240, 48, 55], {"bypass": 1}),
    str(((32, 256, 56, 56), (64, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([26, 64, 448, 32, 16, 56], {"bypass": 0}),
    str(((32, 3, 224, 224), (64, 3, 7, 7), (2, 3, 2, 3), (2, 2), (1, 1), False)):
        ([61, 64, 224, 48, 64, 229], {"bypass": 0}),
    str(((32, 512, 28, 28), (128, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([8, 96, 224, 48, 16, 28], {"bypass": 0}),
    str(((32, 512, 28, 28), (256, 512, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False)):
        ([27, 48, 48, 160, 48, 27], {"bypass": 0}),
    str(((32, 512, 7, 7), (2048, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([7, 864, 48, 288, 16, 7], {"bypass": 0}),
    str(((32, 512, 7, 7), (512, 512, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False)):
        ([9, 512, 64, 128, 96, 9], {"bypass": 1}),
    str(((32, 64, 56, 56), (256, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([8, 64, 448, 64, 32, 56], {"bypass": 0}),
    str(((32, 64, 56, 56), (64, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([14, 64, 112, 64, 16, 56], {"bypass": 0}),
    str(((32, 64, 56, 56), (64, 64, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False)):
        ([6, 64, 224, 64, 64, 58], {"bypass": 0}),
    str(((32, 256, 56, 56), (512, 256, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False)):
        ([7, 320, 112, 160, 48, 55], {"bypass": 0}),
    str(((32, 512, 28, 28), (1024, 512, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False)):
        ([13, 496, 96, 176, 144, 27], {"bypass": 0}),
    str(((32, 256, 56, 56), (128, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([2, 96, 112, 48, 96, 56], {"bypass": 0}),
    str(((32, 512, 28, 28), (256, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([20, 128, 16, 448, 64, 28], {"bypass": 0}),
    str(((32, 1024, 14, 14), (512, 1024, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([7, 384, 48, 48, 224, 14], {"bypass": 0}),
    str(((32, 128, 56, 56), (128, 128, 3, 3), (0, 1, 0, 1), (2, 2), (1, 1), False)):
        ([9, 128, 64, 64, 128, 57], {"bypass": 0}),
    str(((32, 256, 28, 28), (256, 256, 3, 3), (0, 1, 0, 1), (2, 2), (1, 1), False)):
        ([29, 256, 80, 224, 144, 29], {"bypass": 1}),
    str(((32, 512, 14, 14), (512, 512, 3, 3), (0, 1, 0, 1), (2, 2), (1, 1), False)):
        ([15, 512, 64, 64, 272, 15], {"bypass": 1}),
    # alexnet
    str(((32, 3, 227, 227), (96, 3, 11, 11), (0, 0, 0, 0), (4, 4), (1, 1), False)):
        ([63, 96, 208, 32, 96, 227], {"bypass": 0}),
    str(((32, 96, 27, 27), (256, 96, 5, 5), (2, 2, 2, 2), (1, 1), (1, 1), False)):
        ([21, 160, 176, 32, 96, 31], {"bypass": 0})
}


@utils.check_input_type((list, tuple), (list, tuple), (list, tuple), (list, tuple), (list, tuple), (list, tuple),
                          (bool, type(None)), (dict, type(None)), (str, type(None)))
def ConvBn1(data, fmap_shape, filter_shape, pad, stride, dilation, use_bias=False, attrs=None, target=utils.CCE):
    """
    Computes sums of 5-D convolutions and use convolution's fp32 result to compute first part of Fused_batch_norm.

    Fused_batch_norm's first part:

    \f[
     m = N \times H \times W \\
     \\mu_{tmp} = \\sum_{n, h, w}{\frac{x}{m}} \\
     \\sigma^2_{tmp} = \\sum_{n, h, w}{\frac{x^2}{m}}
    \f]

    Args:
        data (list[tvm.tensor.Tensor]): the size is 3 if use_bias else the size is 2;
              data[0] Tensor of type float16 ,shape 5D (fN, fC // C0, C0, fH, fW)
              data[1] Tensor of type float16 ,shape 4D (wC // C0 * wH * wW, wN // C0, C0, C0)
              data[2] Tensor of type float16 ,shape 5D (1, wN // C0, 1, 1, 16)
        fmap_shape (list[int]): [fN, fC, fH, fW]
        filter_shape (list[int]): [wN, wC, wH, wW]
        pad (list[int]): [pad_top, pad_bottom, pad_left, pad_right]
        stride (list[int]): [stride_h, stride_w]
        dilation (list[int]): [dilation_h, dilation_w]
        use_bias (bool): bool var.
        attrs (dict): dict with keys for example: conv_tile,bypass

    Returns:
        tvm.tensor.Tensor of same type as data, shape is 5D(oN, oC // C0, oH, oW, C0)
    
    Supported Platforms:
        'Ascend'
    """

    if use_bias:
        raise ValueError("do not support bias yet !!!")

    block_size = 16
    dim_info, conv_tile, bypass, _ = conv_set_dim_func(fmap_shape, filter_shape, pad, stride, dilation, use_bias,
                                                       block_size, attrs, conv_bn1_set_dim_map)
    if attrs is None:
        attrs = {"conv_tile": conv_tile, "bypass": bypass}
    else:
        attrs['conv_tile'] = conv_tile
        attrs['bypass'] = bypass

    conv_res_32 = conv_core(data, fmap_shape, filter_shape, pad, stride, dilation, use_bias, attrs)

    conv_res_16 = cast(conv_res_32, "float16", utils.CCE)

    axes = [3, 2, 0]
    conv_res_32_shape = [x.value for x in conv_res_32.shape]
    num = reduce(lambda i, j: i * j, [conv_res_32_shape[i] for i in axes])
    avg_num = round(float(1) / float(num), 12)

    res_sum = akg.topi.sum(conv_res_32, axes, keepdims=True)
    mean = akg.lang.ascend.vmuls(res_sum, avg_num)

    res_square = akg.tvm.compute(conv_res_32.shape, lambda *i: conv_res_32[i] * conv_res_32[i], name="res_square")
    square_sum = akg.topi.sum(res_square, axes, keepdims=True)
    var_part = akg.lang.ascend.vmuls(square_sum, avg_num)

    # need pragma_force_rmselfdep to enable multicore using atomic add
    # because default pragma_rmselfdep=1 will disable multicore of reduce axes
    attrs = {"dim": dim_info, "enable_bisect_optimize": 0,
             "pragma_rmselfdep": 0, "pragma_force_rmselfdep": 1}

    return conv_res_16, var_part, mean, attrs
