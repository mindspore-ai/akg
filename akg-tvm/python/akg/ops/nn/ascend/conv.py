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

"""operator dsl function: conv"""
import akg
import akg.tvm
import akg.lang.ascend
import akg.utils as utils
from akg import dim
from akg.ops.math.cast  import cast
from akg.utils.format_transform import get_shape
from akg.utils.dynamic_shape import set_poly_upper_bound_for_tensor

K_H_FAKE = 11
K_W_FAKE = 31
P_TOP_FAKE = 9
P_BOTTOM_FAKE = 8
P_LEFT_FAKE = 23
P_RIGHT_FAKE = 21
S_H_FAKE = 7
S_W_FAKE = 17
C1_CUT_FAKE = 67
TILE_OUT_H_FAKE = 47
TILE_OUT_W_FAKE = 37
M_CUT_FAKE = 53 * 16
K_CUT_FAKE = 59 * 16
N_CUT_FAKE = 61 * 16

conv_set_dim_map = {
    str(((1, 1024, 14, 14), (2048, 1024, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), True)):
        ([14, 2048, 64, 96, 128], {"bypass": 1}),
    str(((1, 1024, 14, 14), (256, 1024, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True)):
        ([14, 256, 208, 64, 112], {"bypass": 1}),
    str(((1, 1024, 14, 14), (512, 1024, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), True)):
        ([14, 512, 49, 32, 512], {"bypass": 1}),
    str(((1, 128, 28, 28), (128, 128, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False)):
        ([28, 128, 400, 32, 128], {"bypass": 1}),
    str(((1, 128, 28, 28), (128, 128, 3, 3), (1, 1, 1, 1), (1, 1), (2, 2), True)):
        ([28, 128, 400, 32, 128], {"bypass": 1}),
    str(((1, 128, 28, 28), (512, 128, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True)):
        ([28, 512, 784, 16, 32], {"bypass": 1}),
    str(((1, 128, 28, 32), (128, 128, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), True)):
        ([28, 128, 16, 72 * 16, 16], {"bypass": 1}),
    str(((1, 128, 34, 34), (128, 128, 3, 3), (1, 1, 1, 1), (1, 1), (2, 2), True)):
        ([36, 128, 64, 32, 64], {"bypass": 1}),
    str(((1, 128, 36, 36), (128, 128, 3, 3), (0, 0, 0, 0), (1, 1), (2, 2), False)):
        ([36, 128, 64, 32, 64], {"bypass": 1}),
    str(((1, 16, 16, 16), (64, 16, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), True)):
        [16, 4 * 16, 16 * 16, 3 * 16, 4 * 16, 16 + 2],
    str(((1, 2048, 7, 7), (512, 2048, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True)):
        ([7, 512, 49, 32, 512], {"bypass": 1}),
    str(((1, 256, 14, 14), (1024, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True)):
        ([14, 944, 112, 32, 240], {"bypass": 1}),
    str(((1, 256, 14, 14), (256, 256, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False)):
        ([14, 256, 196, 64, 256], {"bypass": 1}),
    str(((1, 256, 14, 14), (256, 256, 3, 3), (1, 1, 1, 1), (1, 1), (2, 2), True)):
        ([14, 256, 196, 64, 256], {"bypass": 1}),
    str(((1, 256, 56, 56), (128, 256, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), True)):
        ([7, 128, 252, 64, 128], {"bypass": 1}),
    str(((1, 256, 56, 56), (512, 256, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), True)):
        ([7, 512, 196, 64, 256], {"bypass": 1}),
    str(((1, 256, 56, 56), (64, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True)):
        ([16, 64, 280, 16, 64], {"bypass": 1}),
    str(((1, 3, 224, 224), (64, 3, 7, 7), (2, 3, 2, 3), (2, 2), (1, 1), True)):
        ([117, 64, 448, 32, 64], {"bypass": 1}),
    str(((1, 3, 224, 224), (64, 3, 7, 7), (3, 3, 3, 3), (2, 2), (1, 1), True)):
        ([65, 64, 448, 32, 64], {"bypass": 1}),
    str(((1, 512, 14, 14), (512, 512, 3, 3), (1, 1, 1, 1), (2, 2), (1, 1), False)):
        ([14, 512, 64, 48, 128, 16], {"bypass": 1}),
    str(((1, 512, 28, 28), (1024, 512, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), True)):
        ([13, 1024, 112, 32, 512], {"bypass": 1}),
    str(((1, 512, 28, 28), (128, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True)):
        ([14, 128, 448, 16, 64], {"bypass": 1}),
    str(((1, 512, 28, 28), (256, 512, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), True)):
        ([11, 256, 98, 64, 256], {"bypass": 1}),
    str(((1, 512, 7, 7), (2048, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True)):
        ([7, 2048, 49, 16, 512], {"bypass": 1}),
    str(((1, 512, 7, 7), (512, 512, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False)):
        ([7, 512, 49, 32, 512], {"bypass": 1}),
    str(((1, 512, 7, 7), (512, 512, 3, 3), (1, 1, 1, 1), (1, 1), (2, 2), True)):
        ([7, 512, 49, 32, 512], {"bypass": 1}),
    str(((1, 64, 112, 112), (64, 64, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False)):
        ([56, 64, 784, 16, 32], {"bypass": 1}),
    str(((1, 64, 56, 56), (256, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True)):
        ([56, 256, 784, 16, 32], {"bypass": 1}),
    str(((1, 64, 56, 56), (64, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True)):
        ([56, 64, 784, 16, 32], {"bypass": 1}),
    str(((1, 64, 56, 56), (64, 64, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False)):
        ([56, 64, 336, 16, 64], {"bypass": 1}),
    str(((2, 1024, 48, 72), (2048, 1024, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([5, 128, 288, 16, 112, 72], {"bypass": 0}),
    str(((2, 1024, 48, 72), (256, 1024, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([4, 128, 256, 96, 64, 72], {"bypass": 0}),
    str(((2, 1024, 48, 72), (512, 1024, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([3, 512, 160, 160, 96, 72], {"bypass": 1}),
    str(((2, 128, 192, 288), (256, 128, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([3, 128, 288, 48, 48, 288], {"bypass": 0}),
    str(((2, 128, 192, 288), (64, 128, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([3, 64, 176, 80, 64, 288], {"bypass": 1}),
    str(((2, 128, 48, 72), (512, 128, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([8, 256, 256, 16, 48, 72], {"bypass": 0}),
    str(((2, 128, 96, 144), (128, 128, 3, 3), (1, 1, 1, 1), (2, 2), (1, 1), False)):
        ([17, 16, 256, 96, 16, 145], {"bypass": 0}),
    str(((2, 128, 96, 144), (128, 128, 3, 3), (2, 2, 2, 2), (1, 1), (1, 1), False)):
        ([17, 32, 320, 48, 32, 148], {"bypass": 0}),
    str(((2, 128, 96, 144), (512, 128, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([6, 256, 160, 16, 192, 144], {"bypass": 0}),
    str(((2, 1280, 48, 72), (256, 1280, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([4, 64, 48, 160, 64, 72], {"bypass": 0}),
    str(((2, 16, 768, 1152), (64, 16, 3, 3), (1, 1, 1, 1), (2, 2), (1, 1), False)):
        ([25, 16, 256, 48, 16, 1153], {"bypass": 0}),
    str(((2, 2048, 1, 1), (256, 2048, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([1, 64, 16, 176, 64, 1], {"bypass": 0}),
    str(((2, 2048, 48, 72), (256, 2048, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([3, 256, 48, 432, 64, 72], {"bypass": 1}),
    str(((2, 2048, 48, 72), (512, 2048, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([2, 512, 144, 64, 80, 72], {"bypass": 1}),
    str(((2, 256, 192, 288), (64, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([3, 64, 256, 128, 64, 288], {"bypass": 0}),
    str(((2, 256, 48, 72), (1024, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([6, 64, 144, 176, 16, 72], {"bypass": 0}),
    str(((2, 256, 48, 72), (256, 256, 3, 3), (2, 2, 2, 2), (1, 1), (1, 1), False)):
        ([19, 16, 240, 96, 16, 76], {"bypass": 0}),
    str(((2, 256, 48, 72), (3, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([6, 16, 16, 208, 16, 72], {"bypass": 0}),
    str(((2, 256, 96, 144), (128, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([96, 128, 128, 64, 64, 16], {"bypass": 0}),
    str(((2, 256, 96, 144), (512, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([8, 512, 336, 48, 112, 144], {"bypass": 0}),
    str(((2, 512, 48, 72), (1024, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([4, 256, 112, 32, 176, 72], {"bypass": 0}),
    str(((2, 512, 48, 72), (2048, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([4, 64, 288, 48, 32, 72], {"bypass": 0}),
    str(((2, 512, 48, 72), (256, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([4, 64, 96, 16, 32, 72], {"bypass": 0}),
    str(((2, 512, 96, 144), (128, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([4, 128, 272, 32, 96, 144], {"bypass": 0}),
    str(((2, 64, 192, 288), (256, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([6, 64, 192, 16, 64, 288], {"bypass": 0}),
    str(((2, 64, 192, 288), (64, 64, 3, 3), (1, 1, 1, 1), (2, 2), (1, 1), False)):
        ([193, 16, 352, 80, 16, 33], {"bypass": 0}),
    str(((2, 64, 192, 288), (64, 64, 3, 3), (2, 2, 2, 2), (1, 1), (1, 1), False)):
        ([9, 16, 1968, 16, 16, 292], {"bypass": 0}),
    str(((2, 64, 384, 576), (128, 64, 3, 3), (2, 2, 2, 2), (1, 1), (1, 1), False)):
        ([5, 16, 1600, 16, 16, 580], {"bypass": 0}),
    str(((2, 64, 384, 576), (64, 64, 3, 3), (2, 2, 2, 2), (1, 1), (1, 1), False)):
        ([388, 32, 448, 32, 32, 18], {"bypass": 0}),
    str(((2, 64, 96, 144), (256, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([96, 256, 224, 32, 96, 16], {"bypass": 1}),
    str(((32, 1024, 14, 14), (2048, 1024, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False)):
        ([13, 2048, 64, 48, 336, 13], {"bypass": 1}),
    str(((32, 1024, 14, 14), (256, 1024, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([14, 256, 112, 112, 128, 14], {"bypass": 1}),
    str(((32, 1024, 14, 14), (512, 1024, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([14, 272, 160, 32, 176, 14], {"bypass": 0}),
    str(((32, 1024, 14, 14), (512, 1024, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False)):
        ([13, 512, 64, 80, 112, 13], {"bypass": 1}),
    str(((32, 128, 28, 28), (128, 128, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False)):
        ([18, 128, 336, 32, 80, 30], {"bypass": 0}),
    str(((32, 128, 28, 28), (512, 128, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([4, 288, 112, 96, 272, 28], {"bypass": 0}),
    str(((32, 128, 56, 56), (128, 128, 3, 3), (0, 1, 0, 1), (2, 2), (1, 1), False)):
        ([57, 128, 288, 48, 128, 41], {"bypass": 0}),
    str(((32, 16, 33, 33), (64, 16, 3, 3), (0, 0, 0, 0), (2, 2), (1, 1), False)):
        [128, 128, 64, 128, 64],
    str(((32, 16, 34, 34), (64, 16, 3, 3), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        [128, 128, 64, 128, 64],
    str(((32, 2048, 7, 7), (512, 2048, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([7, 512, 64, 80, 176, 7], {"bypass": 1}),
    str(((32, 256, 14, 14), (1024, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([14, 352, 176, 128, 192, 14], {"bypass": 0}),
    str(((32, 256, 14, 14), (256, 256, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False)):
        ([14, 192, 160, 96, 160, 16], {"bypass": 0}),
    str(((32, 256, 28, 28), (256, 256, 3, 3), (0, 1, 0, 1), (2, 2), (1, 1), False)):
        ([17, 256, 112, 112, 160, 29], {"bypass": 1}),
    str(((32, 256, 56, 56), (128, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([10, 128, 336, 32, 128, 56], {"bypass": 0}),
    str(((32, 256, 56, 56), (128, 256, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False)):
        ([15, 128, 16, 256, 80, 55], {"bypass": 1}),
    str(((32, 256, 56, 56), (512, 256, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False)):
        ([13, 384, 32, 32, 384, 55], {"bypass": 0}),
    str(((32, 256, 56, 56), (64, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([12, 64, 336, 16, 64, 56], {"bypass": 0}),
    str(((32, 3, 224, 224), (64, 3, 7, 7), (2, 3, 2, 3), (2, 2), (1, 1), False)):
        ([17, 64, 336, 48, 64, 229], {"bypass": 0}),
    str(((32, 3, 224, 224), (64, 3, 7, 7), (3, 3, 3, 3), (2, 2), (1, 1), False)):
        ([149, 64, 96, 35, 64, 117], {"bypass": 0}),
    str(((32, 512, 14, 14), (512, 512, 3, 3), (0, 1, 0, 1), (2, 2), (1, 1), False)):
        ([15, 512, 64, 32, 368, 15], {"bypass": 1}),
    str(((32, 512, 28, 28), (1024, 512, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False)):
        ([27, 1024, 160, 112, 96, 27], {"bypass": 1}),
    str(((32, 512, 28, 28), (128, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([6, 128, 160, 16, 128, 28], {"bypass": 0}),
    str(((32, 512, 28, 28), (256, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([28, 128, 144, 160, 128, 28], {"bypass": 0}),
    str(((32, 512, 28, 28), (256, 512, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False)):
        ([27, 256, 160, 144, 32, 27], {"bypass": 1}),
    str(((32, 512, 7, 7), (2048, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([7, 2048, 64, 48, 336, 7], {"bypass": 1}),
    str(((32, 512, 7, 7), (512, 512, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False)):
        ([9, 512, 64, 256, 64, 9], {"bypass": 1}),
    str(((32, 64, 56, 56), (256, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([42, 176, 224, 48, 96, 56], {"bypass": 0}),
    str(((32, 64, 56, 56), (64, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([12, 64, 32, 64, 64, 56], {"bypass": 1}),
    str(((32, 64, 56, 56), (64, 64, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False)):
        ([40, 64, 1120, 16, 48, 58], {"bypass": 0}),
    str(((32, 96, 28, 28), (256, 96, 5, 5), (2, 2, 2, 2), (1, 1), (1, 1), False)):
        ([28, 64, 112, 160, 32, 32], {"bypass": 0}),
    str(((8, 512, 26, 38), (512, 512, 3, 3), (0, 0, 0, 0), (1, 1), (1, 1), False)):
        ([26, 512, 128, 96, 80, 11], {"bypass": 1}),
    str(((32, 3, 227, 227), (96, 3, 11, 11), (0, 0, 0, 0), (4, 4), (1, 1), False)):
        ([63, 96, 208, 32, 96, 227], {"bypass": 0}),
    str(((32, 96, 27, 27), (256, 96, 5, 5), (2, 2, 2, 2), (1, 1), (1, 1), False)):
        ([21, 160, 352, 32, 96, 31], {"bypass": 0})
}


def conv_set_dim_func(fmap_shape, filter_shape, pad, stride, dilation,
                      use_bias=False, block_size=16, attrs=None, setdim_map=None):
    """set dim info in attrs by conv_set_dim_map."""
    if isinstance(stride, int):
        stride = [stride] * 2
    elif isinstance(stride, (list, tuple)) and len(stride) == 1:
        stride = list(stride) * 2
    elif isinstance(stride, (list, tuple)) and len(stride) == 2:
        pass
    else:
        raise IndexError("stride para illegal !!!")

    if isinstance(pad, int):
        pad = [pad] * 4
    elif isinstance(pad, (list, tuple)) and len(pad) == 1:
        pad = list(pad) * 4
    elif isinstance(pad, (list, tuple)) and len(pad) == 4:
        pass
    else:
        raise IndexError("pad para illegal !!!")

    if isinstance(dilation, int):
        dilation = [dilation] * 2
    elif isinstance(dilation, (list, tuple)) and len(dilation) == 1:
        dilation = list(dilation) * 2
    elif isinstance(dilation, (list, tuple)) and len(dilation) == 2:
        pass
    else:
        raise IndexError("dilation para illegal !!!")

    key = []

    key.append(tuple(fmap_shape))
    key.append(tuple(filter_shape))
    key.append(tuple(pad))
    key.append(tuple(stride))
    key.append(tuple(dilation))
    key.append(use_bias)

    hash_key = str(tuple(key))

    # input shape (NCHW -> NC1HWC0)
    in_n, in_c, in_h, in_w = fmap_shape
    in_c = (in_c + block_size - 1) // block_size * block_size

    # kernel shape (NCHW -> NC1HWC0 -> Fractal)
    k_n, k_c, k_h, k_w = filter_shape
    k_c = (k_c + block_size - 1) // block_size * block_size
    k_n = (k_n + block_size - 1) // block_size * block_size

    # padding(padding_top, padding_bottom, padding_left, padding_right)
    padding = (pad[0], pad[1], pad[2], pad[3])
    p_top, p_bottom, p_left, p_right = padding

    # stride (stride_h, stride_w)
    s_h, s_w = stride

    # dilation (dilation_h, dilation_w)
    d_h, d_w = dilation

    k_w_d = (k_w - 1) * d_w + 1
    out_w = (in_w + p_left + p_right - k_w_d) // (s_w) + 1

    all_dynamic = 0      # kh kw pad stride
    partial_dynamic = 0  # fn fc1 fh fw wN wC
    if attrs is None:
        attrs = {}
    if attrs.get("dynamic"):
        all_dynamic = 1
    if attrs.get("partial_dynamic"):
        partial_dynamic = 1
    dynamic = partial_dynamic or all_dynamic

    bypass_list = [0, 1]
    bypass = 0 if dynamic else 1
    dynamic_tiling = 1 if attrs.get("dynamic") else 0  # tile size is a parameter
    use_autotiling = 1 if dynamic and not dynamic_tiling else 0

    dynamic_ci_c1 = 128

    if attrs is not None and "conv_tile" in attrs and len(attrs["conv_tile"]) >= 5:
        use_autotiling = 0
        tiles = attrs["conv_tile"]
        tile_hh = attrs["conv_tile"][0]
        tile_coco = attrs["conv_tile"][1]
        tile_mm = attrs["conv_tile"][2]
        tile_kk = attrs["conv_tile"][3]
        tile_nn = attrs["conv_tile"][4]
        if len(attrs["conv_tile"]) > 5:
            tile_ww = attrs["conv_tile"][5]
            if dynamic and not use_autotiling and len(attrs["conv_tile"]) == 7:
                dynamic_ci_c1 = attrs["conv_tile"][6]
        else:
            tile_ww = (out_w - 1) * s_w + k_w_d
        if "bypass" in attrs:
            bypass = attrs["bypass"]
    elif hash_key in setdim_map:
        configs = setdim_map[hash_key]
        if isinstance(configs, tuple):
            tiles = configs[0]
            if "bypass" in configs[1]:
                bypass = configs[1]["bypass"]
        else:
            tiles = configs
        if len(tiles) > 5:
            tile_hh, tile_coco, tile_mm, tile_kk, tile_nn, tile_ww = tiles
        else:
            tile_hh, tile_coco, tile_mm, tile_kk, tile_nn = tiles
            tile_ww = (out_w - 1) * s_w + k_w_d
    else:
        win_cut_h = 1
        k_h_d = (k_h - 1) * d_h + 1
        win_h = (in_h + p_top + p_bottom - k_h_d) // (s_h) + 1
        if not dynamic:
            while win_cut_h <= win_h:
                if (((win_h + win_cut_h - 1) // win_cut_h - 1) * win_cut_h - 1) * s_h + k_h_d <= in_h + p_top:
                    break
                win_cut_h += 1
        tile_hh = (win_cut_h - 1) * s_h + k_h_d
        tile_ww = (out_w - 1) * s_w + k_w_d
        tile_coco = 16
        tile_mm = 16
        tile_kk = 16
        tile_nn = 16
        tiles = [tile_hh, tile_coco, tile_mm, tile_kk, tile_nn, tile_ww]
    if bypass not in bypass_list:
        raise ValueError("conv_cce ony supports %s while bypass is %d" % (",".join(str(bypass_list)), bypass))

    if tile_hh == in_h:
        tile_hh += p_top + p_bottom
    tile_coco = (tile_coco + block_size - 1) // block_size * block_size
    tile_mm = (tile_mm + block_size - 1) // block_size * block_size
    tile_kk = (tile_kk + block_size - 1) // block_size * block_size
    tile_nn = (tile_nn + block_size - 1) // block_size * block_size

    input_shape_nc1hwc0 = (in_n, in_c // block_size, in_h, in_w, block_size)
    in_n, in_c1, in_h, in_w, _ = input_shape_nc1hwc0

    kernel_shape_nc1hwc0 = (k_n, k_c // block_size, k_h, k_w, block_size)
    k_n, _, k_h, k_w, _ = kernel_shape_nc1hwc0

    k_h_d = (k_h - 1) * d_h + 1
    k_w_d = (k_w - 1) * d_w + 1
    out_h = (in_h + p_top + p_bottom - k_h_d) // (s_h) + 1
    tile_out_h = (tile_hh - k_h_d) // s_h + 1
    out_w = (in_w + p_left + p_right - k_w_d) // (s_w) + 1
    tile_out_w = (tile_ww - k_w_d) // s_w + 1

    out_shape_nc1hwc0 = (in_n, k_n // block_size, out_h, out_w, block_size)
    out_n, out_c1, out_h, out_w, out_c0 = out_shape_nc1hwc0

    if tile_coco > 0:
        c1_cut = tile_coco // block_size
    else:
        c1_cut = out_c1

    # set dim
    def gen_static_dim():
        info = dim.Dim()
        if out_n > 1:
            info.setdim(index=0, axis=0, tilel1=1, tilel0=0)  # n
        if out_c1 > 1:
            info.setdim(index=0, axis=0, tilel1=c1_cut, tilel0=0)  # c1
        if out_h > 1:
            info.setdim(index=0, axis="H", tilel1=tile_out_h, tilel0=0)  # h
        if out_w > 1:
            info.setdim(index=0, axis="W", tilel1=tile_out_w, tilel0=0)  # w
        if out_c0 > 1:
            info.setdim(index=0, axis=4, tilel1=out_c0, tilel0=0)  # c0

        if in_c1 > 1:
            info.setdim(index=0, axis=5, tilel1=in_c1, tilel0=0)  # kc1
        if k_h > 1:
            info.setdim(index=0, axis=5, tilel1=k_h, tilel0=0)  # kh
        if k_w > 1:
            info.setdim(index=0, axis=5, tilel1=k_w, tilel0=0)  # kw
        info.setdim(index=0, axis="KC0", tilel1=block_size, tilel0=0)  # kc0
        return info

    def gen_dynamic_dim():
        info = dim.Dim()
        if dynamic:
            info.setdim(index=0, axis=0, tilel1=1, tilel0=0)  # n
        elif out_n > 1:
            info.setdim(index=0, axis=0, tilel1=1, tilel0=0)  # n

        if dynamic_tiling:
            info.setdim(index=0, axis=0, tilel1=C1_CUT_FAKE, tilel0=0)  # c1
        elif dynamic or out_c1 > 1:
            info.setdim(index=0, axis=0, tilel1=c1_cut, tilel0=0)  # c1

        if dynamic_tiling:
            info.setdim(index=0, axis="H", tilel1=TILE_OUT_H_FAKE, tilel0=0)  # h
        elif dynamic or out_h > 1:
            info.setdim(index=0, axis="H", tilel1=tile_out_h, tilel0=0)  # h

        if dynamic_tiling:
            info.setdim(index=0, axis="W", tilel1=TILE_OUT_W_FAKE, tilel0=0)  # w
        elif dynamic or out_w > 1:
            info.setdim(index=0, axis="W", tilel1=tile_out_w, tilel0=0)  # w

        if dynamic or out_c0 > 1:
            info.setdim(index=0, axis=4, tilel1=out_c0, tilel0=0)  # c0

        if dynamic and not use_autotiling:
            info.setdim(index=0, axis=5, tilel1=dynamic_ci_c1, tilel0=0)  # kc1
        elif dynamic or in_c1 > 1:
            info.setdim(index=0, axis=5, tilel1=in_c1, tilel0=0)  # kc1

        if dynamic or k_h > 1:
            info.setdim(index=0, axis=5, tilel1=k_h, tilel0=0)  # kh

        if dynamic or k_w > 1:
            info.setdim(index=0, axis=5, tilel1=k_w, tilel0=0)  # kw

        info.setdim(index=0, axis="KC0", tilel1=block_size, tilel0=0)  # kc0
        return info

    if dynamic:
        info = gen_dynamic_dim()
    else:
        info = gen_static_dim()
    tiling = str(info)
    if use_autotiling:
        tiling = ""
        dynamic_ci_c1 = 215
    return tiling, tiles, bypass, dynamic_ci_c1


def conv_core(data, fmap_shape, filter_shape, pad, stride, dilation, use_bias=False, attrs=None):
    """core computation for op conv."""
    if use_bias:
        if len(data) != 3:
            raise IndexError("data should contain 3 tensors, i.e. feature map, filter and bias")
        if data[2].dtype != "float16":
            raise TypeError("data type of bias should be float16")
    else:
        if len(data) != 2:
            raise IndexError("data should contain 2 tensors, i.e. feature map and filter")
    if data[0].dtype != "float16":
        raise TypeError("data type of feature map should be float16")
    if data[1].dtype != "float16":
        raise TypeError("data type of filter should be float16")
    if not isinstance(use_bias, bool):
        raise TypeError("use_bias should be set as False or True")

    all_dynamic = 0      # kh kw pad stride
    partial_dynamic = 0  # fn fc1 fh fw wN wC
    if attrs is None:
        attrs = {}
    if attrs.get("dynamic"):
        all_dynamic = 1
    if attrs.get("partial_dynamic"):
        partial_dynamic = 1
    dynamic = partial_dynamic or all_dynamic
    dynamic_tiling = 1 if attrs.get("dynamic") else 0
    use_autotiling = 1 if dynamic and not dynamic_tiling else 0
    block_size = 16

    if not dynamic:
        utils.convolution_format_check(fmap_shape, filter_shape, pad, stride, dilation)
        for tmp_data in data:
            shape = [x.value for x in tmp_data.shape]
            utils.check_shape(shape)
        utils.check_shape(fmap_shape)
        utils.check_shape(filter_shape)

    stride_len = 2
    pad_len = 4
    dilation_len = 2
    zero = 0
    max_s = 63
    max_d = 255

    if isinstance(stride, int):
        stride = [stride] * stride_len
    elif isinstance(stride, (list, tuple)) and len(stride) == 1:  # only has one element
        stride = list(stride) * stride_len
    elif isinstance(stride, (list, tuple)) and len(stride) == stride_len:
        pass
    else:
        raise IndexError("stride para illegal !!!")

    if not dynamic:
        for val in stride:
            if val <= zero:
                raise ValueError("elements in stride should be greater than Zero !!!")
            if val > max_s:
                raise ValueError("elements in stride should be less than 64 !!!")

    if isinstance(pad, int):
        pad = [pad] * pad_len
    elif isinstance(pad, (list, tuple)) and len(pad) == 1:  # only has one element
        pad = list(pad) * pad_len
    elif isinstance(pad, (list, tuple)) and len(pad) == pad_len:
        pass
    else:
        raise IndexError("pad para illegal !!!")

    if not dynamic:
        for val in pad:
            if val < zero:
                raise ValueError("elements in pad should not be less than Zero !!!")
            if val > max_d:
                raise ValueError("elements in pad should be less than 256 !!!")

    if isinstance(dilation, int):
        dilation = [dilation] * dilation_len
    elif isinstance(dilation, (list, tuple)) and len(dilation) == 1:  # only has one element
        dilation = list(dilation) * dilation_len
    elif isinstance(dilation, (list, tuple)) and len(dilation) == dilation_len:
        pass
    else:
        raise IndexError("dilation para illegal !!!")

    for val in dilation:
        if val <= zero:
            raise ValueError("elements in dilation should be greater than Zero !!!")
        if val > max_d:
            raise ValueError("elements in dilation should be less than 256 !!!")

    if len(stride) != stride_len or len(pad) != pad_len or len(dilation) != dilation_len:
        raise IndexError(" shape of parameters must be as expected")

    block_size_sub_one = block_size - 1
    # input shape (NCHW -> NC1HWC0)
    in_n, in_c, in_h, in_w = fmap_shape
    in_c = (in_c + block_size_sub_one) // block_size * block_size

    # kernel shape (NCHW -> NC1HWC0 -> Fractal)
    k_n, k_c, k_h, k_w = filter_shape
    k_c = (k_c + block_size_sub_one) // block_size * block_size
    k_n = (k_n + block_size_sub_one) // block_size * block_size

    # padding(padding_top, padding_bottom, padding_left, padding_right)
    p_top, p_bottom, p_left, p_right = pad

    # stride (stride_h, stride_w)
    s_h, s_w = stride

    k_h_real = k_h
    k_w_real = k_w
    p_top_real = p_top
    p_bottom_real = p_bottom
    p_left_real = p_left
    p_right_real = p_right
    s_h_real = s_h
    s_w_real = s_w

    if dynamic_tiling:
        k_h = K_H_FAKE
        k_w = K_W_FAKE
        p_top = P_TOP_FAKE
        p_bottom = P_BOTTOM_FAKE
        p_left = P_LEFT_FAKE
        p_right = P_RIGHT_FAKE
        s_h = S_H_FAKE
        s_w = S_W_FAKE

    # dilation (dilation_h, dilation_w)
    d_h, d_w = dilation

    # tiling
    key = []
    key.append(tuple(fmap_shape))
    key.append(tuple(filter_shape))
    key.append(tuple(pad))
    key.append(tuple(stride))
    key.append(tuple(dilation))
    key.append(use_bias)

    hash_key = str(tuple(key))

    k_w_d = (k_w - 1) * d_w + 1
    out_w = (in_w + p_left + p_right - k_w_d) // (s_w) + 1

    bypass_list = [0, 1]
    bypass = 0 if dynamic else 1

    # (NC1HWCO)
    a_value = data[0]

    # (fractal)
    b_value = data[1]
    setdim_map = conv_set_dim_map

    conv_tile_num = 5
    if attrs is not None and "conv_tile" in attrs and len(attrs["conv_tile"]) >= conv_tile_num:
        use_autotiling = 0
        tile_hh = attrs["conv_tile"][0]
        tile_coco = attrs["conv_tile"][1]
        tile_mm = attrs["conv_tile"][2]
        tile_kk = attrs["conv_tile"][3]
        tile_nn = attrs["conv_tile"][4]
        if len(attrs["conv_tile"]) > conv_tile_num:
            tile_ww = attrs["conv_tile"][conv_tile_num]
        else:
            tile_ww = (out_w - 1) * s_w + k_w_d
        if "bypass" in attrs:
            bypass = attrs["bypass"]
    elif hash_key in setdim_map:
        configs = setdim_map[hash_key]
        if isinstance(configs, tuple):
            tiles = configs[0]
            if "bypass" in configs[1]:
                bypass = configs[1]["bypass"]
        else:
            tiles = configs
        if len(tiles) > conv_tile_num:
            tile_hh, tile_coco, tile_mm, tile_kk, tile_nn, tile_ww = tiles
        else:
            tile_hh, tile_coco, tile_mm, tile_kk, tile_nn = tiles
            tile_ww = (out_w - 1) * s_w + k_w_d
    else:
        win_cut_h = 1
        k_h_d = (k_h - 1) * d_h + 1
        win_h = (in_h + p_top + p_bottom - k_h_d) // (s_h) + 1
        if not dynamic:
            while win_cut_h <= win_h:
                if (((win_h + win_cut_h - 1) // win_cut_h - 1) * win_cut_h - 1) * s_h + k_h_d <= in_h + p_top:
                    break
                win_cut_h += 1
        tile_hh = (win_cut_h - 1) * s_h + k_h_d
        tile_ww = (out_w - 1) * s_w + k_w_d
        tile_coco = block_size
        tile_mm = block_size
        tile_kk = block_size
        tile_nn = block_size
    if bypass not in bypass_list:
        raise ValueError("bypass of conv only supports %s" % (",".join(str(bypass_list))))

    if tile_hh == in_h:
        tile_hh += p_top + p_bottom

    if tile_ww == in_w:
        tile_ww += p_left + p_right

    tile_coco = (tile_coco + block_size_sub_one) // block_size * block_size
    tile_mm = (tile_mm + block_size_sub_one) // block_size * block_size
    tile_kk = (tile_kk + block_size_sub_one) // block_size * block_size
    tile_nn = (tile_nn + block_size_sub_one) // block_size * block_size

    input_shape_nc1hwc0 = get_shape(data[0])
    if not dynamic and input_shape_nc1hwc0 != [in_n, in_c // block_size, in_h, in_w, block_size]:
        raise ValueError("feature map tensor data[0] shape illegal !!!")
    in_n, c1_in, in_h, in_w, _ = input_shape_nc1hwc0

    if not dynamic:
        kernel_shape_nc1hwc0 = (k_n, k_c // block_size, k_h, k_w, block_size)
    else:
        kernel_shape_nc1hwc0 = (k_n, c1_in, k_h, k_w, block_size)  # simplify for dynamic case
    k_n, k_c1, k_h, k_w, k_c0 = kernel_shape_nc1hwc0
    kernel_shape_fractal = get_shape(data[1])
    if not dynamic and kernel_shape_fractal != [k_c1 * k_h * k_w, k_n // block_size, block_size, k_c0]:
        raise ValueError("filter tensor data[1] shape illegal !!!")

    if use_bias:
        bias_value = data[2]
        bias_name = bias_value.op.name
        bias_shape = [x.value for x in data[2].shape]
        if bias_shape != [1, k_n // block_size, 1, 1, block_size]:
            raise ValueError("bias tensor data[2] shape illegal !!!")
    else:
        bias_name = "None"
        bias_value = None

    # Create reduction variables
    kc1 = akg.tvm.reduce_axis((0, k_c1), name="kc1")
    kh = akg.tvm.reduce_axis((0, k_h), name="kh")
    kw = akg.tvm.reduce_axis((0, k_w), name="kw")
    kc0 = akg.tvm.reduce_axis((0, k_c0), name="kc0")

    k_h_d = (k_h - 1) * d_h + 1
    k_h_d_real = (k_h_real - 1) * d_h + 1
    k_w_d = (k_w - 1) * d_w + 1
    k_w_d_real = (k_w_real - 1) * d_w + 1
    out_h = (in_h + p_top + p_bottom - k_h_d) // (s_h) + 1
    tile_out_h = (tile_hh - k_h_d) // s_h + 1
    tile_out_h_real = (tile_hh - k_h_d_real) // s_h_real + 1
    out_w = (in_w + p_left + p_right - k_w_d) // (s_w) + 1
    tile_out_w = (tile_ww - k_w_d) // s_w + 1
    tile_out_w_real = (tile_ww - k_w_d_real) // s_w_real + 1

    if not dynamic:
        out_shape_nc1hwc0 = (in_n, k_n // block_size, out_h, out_w, block_size)
    else:
        _, c1_out, _, _ = data[1].shape
        out_shape_nc1hwc0 = (in_n, c1_out, out_h, out_w, block_size)
    _, out_c1, out_h, out_w, _ = out_shape_nc1hwc0

    if tile_coco > 0:
        c1_cut = tile_coco // block_size
    else:
        c1_cut = out_c1

    # Compute the convolution
    output_name = "output0"
    conv_attr = {
        "pragma_conv_kernel_n": k_n,
        "pragma_conv_kernel_h": k_h,
        "pragma_conv_kernel_w": k_w,
        "pragma_conv_padding_top": p_top,
        "pragma_conv_padding_bottom": p_bottom,
        "pragma_conv_padding_left": p_left,
        "pragma_conv_padding_right": p_right,
        "pragma_conv_bypass_l1": bypass,
        "pragma_conv_stride_h": s_h,
        "pragma_conv_stride_w": s_w,
        "pragma_conv_dilation_h": d_h,
        "pragma_conv_dilation_w": d_w,
        "pragma_conv_fm_n": in_n,
        "pragma_conv_fm_c": in_c,
        "pragma_conv_fm_h": in_h,
        "pragma_conv_fm_w": in_w,
        "feature": a_value.op.name,
        "filter": b_value.op.name,
        "bias": bias_name,
        "res": output_name}

    if dynamic_tiling:
        conv_attr["pragma_conv_h_cut"] = (TILE_OUT_H_FAKE - 1) * s_h + k_h_d
        conv_attr["pragma_conv_w_cut"] = (TILE_OUT_W_FAKE - 1) * s_w + k_w_d
        conv_attr["pragma_conv_co_cut"] = C1_CUT_FAKE * 16
        conv_attr["pragma_conv_m_cut"] = M_CUT_FAKE
        conv_attr["pragma_conv_k_cut"] = K_CUT_FAKE
        conv_attr["pragma_conv_n_cut"] = N_CUT_FAKE
        conv_attr["pragma_conv_tile_co"] = c1_cut
        conv_attr["pragma_conv_tile_ho"] = tile_out_h_real
        conv_attr["pragma_conv_tile_wo"] = tile_out_w_real
        conv_attr["pragma_conv_tile_mo"] = tile_mm // 16
        conv_attr["pragma_conv_tile_ko"] = tile_kk // 16
        conv_attr["pragma_conv_tile_no"] = tile_nn // 16
        conv_attr["pragma_conv_real_kh"] = k_h_real
        conv_attr["pragma_conv_real_kw"] = k_w_real
        conv_attr["pragma_conv_real_sh"] = s_h_real
        conv_attr["pragma_conv_real_sw"] = s_w_real
        conv_attr["pragma_conv_real_pt"] = p_top_real
        conv_attr["pragma_conv_real_pb"] = p_bottom_real
        conv_attr["pragma_conv_real_pl"] = p_left_real
        conv_attr["pragma_conv_real_pr"] = p_right_real
    elif not use_autotiling:
        conv_attr["pragma_conv_h_cut"] = (tile_out_h - 1) * s_h + k_h_d
        conv_attr["pragma_conv_w_cut"] = (tile_out_w - 1) * s_w + k_w_d
        conv_attr["pragma_conv_co_cut"] = c1_cut * k_c0
        conv_attr["pragma_conv_m_cut"] = tile_mm
        conv_attr["pragma_conv_k_cut"] = tile_kk
        conv_attr["pragma_conv_n_cut"] = tile_nn
    c_value = akg.tvm.compute(out_shape_nc1hwc0,
                              lambda n, c1, h, w, c0: akg.lang.ascend.mmad(
                                  (akg.tvm.if_then_else(akg.tvm.any((h * s_h + kh) < p_top,
                                                                    (h * s_h + kh) > (in_h + p_top - 1),
                                                                    (w * s_w + kw) < p_left,
                                                                    (w * s_w + kw) > (in_w + p_left - 1)),
                                                        akg.tvm.const(0.0, "float16"),
                                                        a_value[n, kc1, (h * s_h + (kh * d_h) - p_top),
                                                                (w * s_w + (kw * d_w) - p_left), kc0])
                                   * b_value[(kc1 * k_h + kh) * k_w + kw, c1, c0, kc0]).astype("float32"),
                                  axis=[kc1, kh, kw, kc0]), name=output_name,
                              attrs=conv_attr)
    return c_value

@utils.check_input_type((list, tuple), (list, tuple), (list, tuple), (list, tuple), (list, tuple), (list, tuple),
                          (bool, type(None)), (dict, type(None)), (list, tuple, type(None)), (str, type(None)))
def Conv(data, fmap_shape, filter_shape, pad, stride, dilation, use_bias=False, attrs=None,
        params=None, target=utils.CCE):
    """
    Computes sums of 5-D convolutionis.

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
    c_value = conv_core(data, fmap_shape, filter_shape, pad, stride, dilation, use_bias, attrs)
    c_value = cast(c_value, "float16", utils.CCE)

    if use_bias:
        bias_value = data[2]
        output_bias_name = "output1"
        cube = akg.tvm.compute(c_value.shape, lambda n, c1, h, w, c0: c_value[n, c1, h, w, c0] +
                               bias_value[0, c1, 0, 0, c0],
                               name=output_bias_name)
    else:
        cube = c_value

    block_size = 16
    dim_info, _, _, dynamic_ci_c1 = conv_set_dim_func(fmap_shape, filter_shape, pad, stride, dilation,
                                                      use_bias, block_size, attrs, conv_set_dim_map)

    all_dynamic = 0      # kh kw pad stride
    partial_dynamic = 0  # fn fc1 fh fw wN wC
    dynamic_tiling_full_dynamic = 1  # kh, kw, pad, stride are parameters if dynamic_tiling is enabled

    if attrs is None:
        attrs = {}
    if attrs.get("dynamic"):
        all_dynamic = 1
    if attrs.get("partial_dynamic"):
        partial_dynamic = 1
    dynamic = partial_dynamic or all_dynamic
    dynamic_tiling = 1 if attrs.get("dynamic") else 0

    if not dynamic:
        attrs = {"dim": dim_info, "pragma_rmselfdep": 0}
    else:
        attrs = {"dim": dim_info,
                 "pragma_rmselfdep": 0,
                 "enable_fix_loop_extent": 0,
                 "enable_post_poly_loop_partition": 0,
                 "enable_isolate_loop": 0,
                 "enable_isolate_min_max": 1,
                 "enable_conv_analyze_align": 0,
                 "enable_double_buffer": 1,
                 "enable_multicore": 1,
                 "enable_invariant_hoist": 1,
                 "pragma_keep_outer_band_order": 1,
                 "enable_algebra_simplify": 1,
                 "dynamic_shape_conv_full_parametric": dynamic_tiling and dynamic_tiling_full_dynamic,
                 }
        attrs["pragma_outerband_need_split"] = 1
        attrs["pragma_is_conv"] = 1
        if dynamic_tiling:
            attrs["dynamic_shape"] = set_poly_upper_bound_for_tensor(data[0], 129, 1)  # pos 1 of data[0] is CI1 axis
        else:
            attrs["dynamic_shape"] = set_poly_upper_bound_for_tensor(
                data[0], dynamic_ci_c1 + 1, 1)  # pos 1 of data[0] is CI1 axis
        if dynamic_tiling:
            attrs["pragma_tilesize_is_var"] = 1
            attrs["enable_stride_kernel_op"] = 0

    return cube, attrs
