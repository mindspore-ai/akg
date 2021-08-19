# Copyright 2021 Huawei Technologies Co., Ltd
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

"""operator dsl function: scatter_add"""
import akg.tvm
from akg.utils import validation_check as vc_util
from .gather_nd import gather_nd
from .unsorted_segment_sum import unsorted_segment_sum
from ..math_gpu.mul import mul
from ..math_gpu.reduce_sum import reduce_sum

@vc_util.check_input_type(
    akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, int, bool, int)
def fused_gather_nd_reduce_sum_mul_unsorted_segment_sum(input1, input2, input3, input4, input5, axis=0, keepdims=False, num=0):
    item_get = gather_nd(input1, input2)
    sum_axis = reduce_sum(item_get, axis, keepdims)
    prod = mul(sum_axis, input3)
    res1 = unsorted_segment_sum(prod, input4, num, op_id=0)
    res2 = unsorted_segment_sum(prod, input5, num, op_id=1)
    return res1, res2



