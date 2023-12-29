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

"""nms"""
import akg.tvm
import akg.utils as utils
import akg.utils as utils
from akg.utils import custom_tiling as ct_util
from akg.utils.format_transform import get_shape_from_tensor
from akg.lang import ascend as dav


def nms_tiling_strategy(tensor):
    """Custom tiling strategy for nms op"""
    strategy = list()
    tensor_shape = get_shape_from_tensor(tensor)
    for i, _ in enumerate(tensor_shape):
        if i == 0:
            strategy += ct_util.create_constraint_on_tensor(tensor=tensor,
                                                            values=1,
                                                            constraints=ct_util.TileConstraint.FACTOR,
                                                            tensor_pos=i)
        else:
            strategy += ct_util.create_constraint_on_tensor(tensor=tensor,
                                                            values="FULL",
                                                            constraints=ct_util.TileConstraint.MAX,
                                                            tensor_pos=i)
    return strategy


def nms(data, iou_thre, target="cce"):
    utils.ops_dtype_check(data.dtype, utils.DtypeForDavinci.FLOAT16)
    data_shape = [x.value for x in data.shape]  # n*8
    if len(data_shape) != 3 or data_shape[2] != 8:
        raise ValueError("proposal box should be allocated as [batch_size, boxes, 8]")
    if data_shape[1] % 16 != 0:
        raise ValueError(
            'proposal box number only support in multiples of 16, please pad the data before implement this op')
    out_shape = [data_shape[0], data_shape[1]]

    cor_reducer = akg.tvm.comm_reducer(lambda x, y: dav.nms(x, y, akg.tvm.const(iou_thre, 'float16')),
                                       lambda t: akg.tvm.const(0, dtype=t), name="cor_reducer")
    k = akg.tvm.reduce_axis((0, data_shape[1]), name='k')
    output = akg.tvm.compute(out_shape, lambda bs, i: cor_reducer(data[bs, i, k], axis=k), name='nms_output')
    attrs = {"custom_tiling": nms_tiling_strategy(data)}

    return output, attrs
