# Copyright 2019-2022 Huawei Technologies Co., Ltd
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

"""operator dsl function:exp_ad"""

import akg
import akg.utils as utils
from ..exp import exp


@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, (str, type(None)))
def exp_ad(head, in_data, target=utils.CCE):
    """
    Compute gradient of exp operator using automatic differentiate.

    Args:
        head (tvm.tensor.Tensor): Tensor of type float16, float32.
        in_data (tvm.tensor.Tensor): Tensor of type float16, float32.

    Returns:
        tvm.tensor.Tensor has the same shape as input.

    Supported Platforms:
        'Ascend'
    """

    # check head's validation.
    utils.check_shape(head.shape)
    utils.ops_dtype_check(head.dtype, utils.DtypeForDavinci.ALL_FLOAT)
    exp_in_data = exp(in_data, target)
    jacs = list(akg.differentiate(exp_in_data, [in_data], head))
    return jacs[0]
