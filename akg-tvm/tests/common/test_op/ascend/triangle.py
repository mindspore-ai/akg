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

"""operator dsl function:triangle"""

import akg.tvm
import akg.utils as utils


def triangle(data, const_value, lower, target="cce"):
    """
    Change matrix to triangle.

    Args:
        data: Tensor.
        const_value: Float or integer. Use to set value.
        lower: Boolean. Decide lower triangle or upper triangle.

    Returns:
        Tensor, has the same type and shape as data.
    """
    shape = data.shape
    dtype = data.dtype
    assert len(shape) <= 2

    output_shape = shape
    if len(shape) == 1:
        output_shape = [shape[0], shape[0]]
        if lower:
            output = akg.tvm.compute(output_shape, lambda k, m: akg.tvm.if_then_else(k >= m, data[m], akg.tvm.const(const_value, dtype)), name="tri_matrix")
        else:
            output = akg.tvm.compute(output_shape, lambda k, m: akg.tvm.if_then_else(k <= m, data[m], akg.tvm.const(const_value, dtype)), name="tri_matrix")
    else:
        if lower:
            output = akg.tvm.compute(output_shape, lambda k, m: akg.tvm.if_then_else(k >= m, data[k, m], akg.tvm.const(const_value, dtype)), name="tri_matrix")
        else:
            output = akg.tvm.compute(output_shape, lambda k, m: akg.tvm.if_then_else(k >= m, data[k, m], akg.tvm.const(const_value, dtype)), name="tri_matrix")
    attrs = {'enable_multicore': 0}
    return output, attrs
