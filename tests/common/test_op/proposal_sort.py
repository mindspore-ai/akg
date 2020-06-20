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

"""proposal_sort"""
import akg.tvm
from akg.utils import kernel_exec as utils
from akg.utils import validation_check as vc_util
from akg.utils import custom_tiling as ct_util
from akg.utils.format_transform import get_shape
from akg.lang import cce as dav


def proposal_sort_tiling_strategy(tensor, tensor_shape):
    """
    Custom tiling strategy for proposal_sort op
    """
    strategy = list()
    for i, sh in enumerate(tensor_shape):
        if i == 0 and sh > 1:
            strategy.append(ct_util.create_constraint_on_tensor(tensor=tensor,
                                                                values=1,
                                                                constraints=ct_util.TileConstraint.FACTOR,
                                                                tensor_pos=0)[0])
        if i == 1 and sh > 4096:
            strategy.append(ct_util.create_constraint_on_tensor(tensor=tensor,
                                                                values=1,
                                                                constraints=ct_util.TileConstraint.FACTOR,
                                                                tensor_pos=2)[0])
    return strategy


def proposal_sort(data, topk):
    """
    Computes the k largest entries from input.

    Args:
        data: akg.tvm.Tensor of type float16, float32.
        topk: an integer indicating the top kth entries.

    Returns:
        sorted_data: akg.tvm.Tensor of top kth number of rows.
        attr_map:  optional parameter for setting tiling strategy. 
    """
    vc_util.ops_dtype_check(data.dtype, vc_util.DtypeForDavinci.FLOAT16)
    bs, box_num, _ = data.shape
    result_shape = (bs, topk, data.shape[-1])
    attr_map = {}
    if int(box_num) > 4096:
        reducer = akg.tvm.comm_reducer(lambda x, y: dav.topk_sort(x, y, akg.tvm.const(topk, 'uint16')),
                                       lambda t: akg.tvm.const(0, dtype=t), name="cor_reducer")
        k = akg.tvm.reduce_axis((0, box_num), name='k')
        sorted_data = akg.tvm.compute(result_shape, lambda bs, i, j: reducer(data[bs, k, j], axis=k), name="sort")
    else:

        reducer = akg.tvm.comm_reducer(lambda x, y: dav.proposal_sort(x, y, akg.tvm.const(topk, 'uint16')),
                                       lambda t: akg.tvm.const(0, dtype=t), name="cor_reducer")
        k = akg.tvm.reduce_axis((0, box_num), name='k')
        sorted_data = akg.tvm.compute(result_shape,
                                      lambda bs, i, j: reducer(data[bs, k, j], axis=k),
                                      name="proposal_sort_output")
        attr_map["custom_tiling"] = proposal_sort_tiling_strategy(sorted_data, get_shape(data))
    return sorted_data, attr_map
