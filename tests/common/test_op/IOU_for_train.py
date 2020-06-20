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

"""
dropout dsl
"""
import akg.tvm
from akg.lang import cce as dav
from akg.utils import custom_tiling as ct_util


def iou_set_dim_func(anchor_box, ground_truth_box):
    tile_list = []
    if anchor_box.shape[0].value > 1:
        tile_list.append((1, 1))
    if anchor_box.shape[1].value > 16:
        tile_list.append((16, 16))
    if len(tile_list) == 0:
        return ""
    return ct_util.set_dims(tuple(tile_list))


@ct_util.reg_set_dim_func(iou_set_dim_func)
def iou_for_train(anchor_box, ground_truth_box):
    """
    Computes anchor_box and ground_truth_box's intersection-over-union

    Args:
        anchor_box (tvm.tensor.Tensor): Tensor of type float16.
        ground_truth_box (tvm.tensor.Tensor): Tensor of type float16.

    Returns:
        tvm.tensor.Tensor, has same type and shape as anchor_box.
    """
    anchor_box_dtype = anchor_box.dtype
    ground_truth_box_dtype = ground_truth_box.dtype
    shape1 = [x.value for x in anchor_box.shape]
    shape2 = [x.value for x in ground_truth_box.shape]
    out_shape = [shape1[0], shape1[1], shape2[1]]
    check_list = ["float16"]
    if not (anchor_box_dtype in check_list or ground_truth_box_dtype in check_list):
        raise RuntimeError(
            "dropout_do_mask only support %s while dtype is %s" % (",".join(check_list), anchor_box_dtype))

    if len(shape1) != 3 or shape1[2] != 8:
        raise ValueError("proposal box should be allocated as [batch_size, boxes, 8]")
    if len(shape2) != 3 or shape2[2] != 8:
        raise ValueError("proposal box should be allocated as [batch_size, boxes, 8]")
    if shape1[1] % 16 != 0 or shape2[1] % 16 != 0:
        raise ValueError(
            'proposal box number only support in multiles of 16, please pad the data before implement this ops')

    reducer = akg.tvm.comm_reducer(lambda x, y: y, lambda t: akg.tvm.const(0, dtype=t), name="reducer")
    k = akg.tvm.reduce_axis((0, 8), name='k')
    res = akg.tvm.compute(out_shape,
                          lambda bs, i, j: reducer(
                              dav.iou(anchor_box[bs, i, k], ground_truth_box[bs, j, k]), axis=k),
                          name="iou_area")

    return res
