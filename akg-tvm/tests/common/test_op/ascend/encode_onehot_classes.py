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

"""encode_onehot_classes"""

import akg.tvm
from akg.utils import custom_tiling as ct_util
import akg.utils as utils
from akg.utils.format_transform import get_shape
from akg.ops.array.ascend import OneHot

encode_one_hot_set_dim_map = {
    str(((8, 16), (8, 4718), 12, 'int32')): ((1, 1), (16, 1), (4718, 1)),
    str(((8, 16), (8, 8732), 12, 'int32')): ((1, 1), (16, 1), (4366, 1)),
}


def encode_one_hot_set_dim_func(groundtruth_class, anchor_sample, class_num):
    """setdim function"""
    shape1 = []
    shape2 = []
    for x in groundtruth_class.shape:
        shape1.append(x)
    for x in anchor_sample.shape:
        shape2.append(x)
    hash_key = str((tuple(shape1), tuple(shape2), class_num, groundtruth_class.dtype))

    return ct_util.set_dims_by_key(hash_key, encode_one_hot_set_dim_map), hash_key

@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, int)
def encode_onehot_classes(groundtruth_class, anchor_sample, class_num):
    """
    One-hot encode the anchor_class.
    
    This op uses `anchor_sample` to dereference `groundtruth_class`,
    and then encode the one-hot result for each anchor.
    
    Args:
        groundtruth_class (tvm.tensor.Tensor): The `class_id` of each groundtruth, shape (batchsize, num_groundtruths).
        anchor_sample (tvm.tensor.Tensor): The `groundtruth_id` of each anchor. shape (batchsize, num_anchors).
        class_num (int): Class number.
    
    Returns:
        akg.tvm.Tensor, shape `(batchsize, num_anchors, class_num)`
    """
    utils.check_shape(groundtruth_class, 2, "groundtruth_class")
    utils.check_shape(anchor_sample, 2, "anchor_sample")
    utils.check_equal("batchsize in groundtruth_class", "batchsize in anchor_sample",
        groundtruth_class.shape[0].value, anchor_sample.shape[0].value)
    utils.ops_dtype_check([groundtruth_class.dtype, anchor_sample.dtype], utils.DtypeForDavinci.INT32)
    utils.check_greater("class_num", "zero", class_num, 0)

    dim_info, _ = encode_one_hot_set_dim_func(groundtruth_class, anchor_sample, class_num)
    attrs = {"dim": dim_info}

    onehot_res, _ = OneHot(groundtruth_class, class_num, groundtruth_class.dtype, on_value=1, off_value=0, axis=-1)

    an_shape = get_shape(anchor_sample)
    out_shape = an_shape + [class_num]
    res = akg.tvm.compute(out_shape, lambda b, a, c: onehot_res[b, anchor_sample[b, a], c], name="encode_result")
    return res, attrs
