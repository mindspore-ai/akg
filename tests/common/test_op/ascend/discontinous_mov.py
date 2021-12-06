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

"""operator dsl function: discontinous_mov"""

import akg.tvm
import akg.utils as utils


def discontinous_mov(data, out_shape, target=utils.CCE):
    """
    Extract the element with the odd index from the original data and copy it into a tensor with a dimension of
    2 * original dimension/2.

    Args:
        data (tvm.tensor.Tensor): Tensor of type float16, float32.
        out_shape (list): a list of output's shape.

    Returns:
           tvm.tensor.Tensor, has the same type as data, but it's shape changes to out_shape not data's shape.

    Example:
           if data = [1,2,3,4,5,6,7,8,9,10] then the output = [[1,3,5,7,9],[1,3,5,7,9]].
    """

    # check types
    utils.ops_dtype_check(data.dtype, utils.DtypeForDavinci.ALL_FLOAT)
    shape = [x.value for x in data.shape]
    utils.check_shape(shape)
    
    output = akg.tvm.compute(out_shape, lambda j, i: data[i * 2], name="output")

    return output
