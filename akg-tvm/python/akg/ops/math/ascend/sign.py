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

"""operator dsl function:sign"""

import akg.topi
import akg.tvm
import akg.utils as utils


def Sign(inputs, target=utils.CCE):
    """
    the sign of the number in a akg.tvm.Tensor
   
    :math: `y = sign(x) = -1 if x < 0; 0 if x == 0; 1 if x > 0.`

    Args:
        inputs: akg.tvm.Tensor of type float16, float32 int8 uint8

    Returns:
        akg.tvm.Tensor of same shape and type as input

    Raises:
        ValueError: If the type of input is invalid.
    
    Supported Platforms:
        'Ascend'
    """
    dtype = inputs.dtype
    utils.ops_dtype_check(dtype, [utils.DtypeForDavinci.ALL_FLOAT,
                                    utils.DtypeForDavinci.INT8,
                                    utils.DtypeForDavinci.UINT8])

    shape = inputs.shape
    utils.check_shape(shape)

    if dtype == "float16":
        data_f16 = inputs
    else:
        data_f16 = akg.tvm.compute(shape, lambda *i: inputs(*i).astype("float16"), name='data_f16')

    res_tmp = akg.topi.sign(data_f16)

    if dtype == "float16":
        res = res_tmp
    else:
        res = akg.topi.cast(res_tmp, dtype)

    return res
