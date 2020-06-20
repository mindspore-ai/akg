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

"""operator dsl function: detection_four2five"""
import akg
import akg.tvm
from akg.utils import validation_check as vc_util

@vc_util.check_input_type(akg.tvm.tensor.Tensor, int)
def detection_four2five(data, slice_idx):
    """
    Change data from specific four dims to five dims format.

    Shape changes: [N, box_num * H * W, 4, 1] -> [N, ceil((box_num * 4) / 16), H, W, 16].

    Note:
        With slice + detection_four2five, it can make data with shape
        [16, 8732, 4, 1] to six data with shape [16, 16//16, 38, 38, 16],
        [16, 24//16+1, 19, 19, 16], [16, 24//16+1, 10, 10, 16],
        [16, 24//16+1, 5, 5, 16], [16, 16//16, 3, 3, 16].

    Args:
        data (tvm.tensor.Tensor): Tensor of type float16 with four dims format which the
                       length of last dim is 1.
        slice_idx (int): Index of slice number.

    Returns:
        A tensor with five dims shape.
    """
    vc_util.ops_dtype_check(data.dtype, vc_util.DtypeForDavinci.FLOAT16)

    bs = data.shape[0]
    shape_list = [(bs, 16), (bs, 144), (bs, 600), (bs, 2400), (bs, 8664), (bs, 23104)]
    res = None
    if slice_idx == 0:
        res = akg.tvm.compute(shape_list[0], lambda i, j: data[i][j], name="shape1")
    elif slice_idx == 1:
        res = akg.tvm.compute(shape_list[1], lambda i, j: data[i][j + 16], name="shape2")
    elif slice_idx == 2:
        res = akg.tvm.compute(shape_list[2], lambda i, j: data[i][j + 160], name="shape3")
    elif slice_idx == 3:
        res = akg.tvm.compute(shape_list[3], lambda i, j: data[i][j + 760], name="shape4")
    elif slice_idx == 4:
        res = akg.tvm.compute(shape_list[4], lambda i, j: data[i][j + 3160], name="shape5")
    elif slice_idx == 5:
        res = akg.tvm.compute(shape_list[5], lambda i, j: data[i][j + 11824], name="shape6")
    else:
        raise ValueError("slice index {} not support!".format(slice_idx))

    return res
