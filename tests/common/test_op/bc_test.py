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

"""operator dsl function:bc_test"""

import akg.tvm
from akg.lang.cce import vadd
from akg.utils import validation_check as vc_util

@vc_util.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor)
def bc_test(first_input, second_input, third_input):
    """
    Test whether the bank conflict optimization recognizes that inputs A and C
    are used in two operations, while other pairs of inputs are only used in
    1 operation each. Therefore, A and C should be placed on the left and right
    side of Ubuf, to minimize read-read conflicts.
    With ilp_rr_cost=False, the total number of RR conflicts is 4, but with
    ilp_rr_cost=True, the total number of RR conflicts is reduced to 2, for
    tensors of shape [256].
    C = (A + C) + (A - C) + (B + C) + (A + B)
    """
    vc_util.check_shape(first_input)
    vc_util.check_shape(second_input)
    vc_util.check_shape(third_input)

    res1 = first_input + third_input
    res2 = first_input - third_input
    res3 = second_input + third_input
    res4 = first_input + second_input
    res5 = vadd(res1, res2)
    res6 = vadd(res3, res4)
    res = vadd(res5, res6)
    return res
