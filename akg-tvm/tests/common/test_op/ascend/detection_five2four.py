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

"""operator dsl function: detection_five2four"""
import akg
from akg.tvm.hybrid import script
import akg.utils as utils

@utils.check_input_type(akg.tvm.tensor.Tensor, int, (str, type(None)))
def detection_five2four(data, box_num, target=utils.CCE):
    """
    Change data from five dims to specific four dims format.

    Shape changes: [N, ceil((box_num * 4) / 16), H, W, 16] -> [N, box_num * H * W, 4, 1].

    Note:
        With detection_five2four + concat, it can make datas with
        shape [16, 16//16, 38, 38, 16], [16, 24//16+1, 19, 19, 16],
        [16, 24//16+1, 10, 10, 16], [16, 24//16+1, 5, 5, 16], [16, 16//16, 3, 3, 16]
        and [16, 16//16, 1, 1, 16] to one data with shape [16, 8732, 4, 1].

    Args:
        data (Tensor): tvm.Tensor of type float16 with five dims format which
                       the length of its third and fourth dim is equal (H == W).
        box_num (Integer): number of box.

    Returns:
        A tvm.Tensor with 4 dims shape which the length of its last dim is 1.
    """
    utils.ops_dtype_check(data.dtype, utils.DtypeForDavinci.FLOAT16)

    block_size = 16
    batch_size, c1, wh, _, c0 = data.shape
    # each box has 4 numbers
    pad = (box_num * 4) % block_size

    @script(capture=locals())
    def reshape(inputs):
        out = allocate((batch_size, wh * wh, box_num * 4), 'float16', 'local')
        for i in range(batch_size):
            for j in range(c1):
                for k in range(wh):
                    for l in range(wh):
                        for m in range(c0):
                            out[i, k * wh + l, j * c0 + m] = inputs[i, j, k, l, m]
        return out

    @script(capture=locals())
    def reshape_with_pad(inputs):
        out = allocate((batch_size, wh * wh, box_num * 4), 'float16', 'local')
        for i in range(batch_size):
            for j in range(box_num // 4 + 1):
                for k in range(wh):
                    for l in range(wh):
                        if j == box_num // 4:
                            for m1 in range(pad):
                                out[i, k * wh + l, j * block_size + m1] = inputs[i, j, k, l, m1]
                        else:
                            for m in range(c0):
                                out[i, k * wh + l, j * block_size + m] = inputs[i, j, k, l, m]
        return out

    if pad == 0:
        data_rs = reshape(data)
    else:
        data_rs = reshape_with_pad(data)

    return data_rs
