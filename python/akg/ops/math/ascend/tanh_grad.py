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

"""operator dsl function:tanh_grad"""

import akg
import akg.tvm
import akg.utils as utils


def tanh_grad(data_y, data_dy, target=utils.CCE):
    """
    Compute the backpropogation gradient of tanh.

    Args:
        data_y: Tensor, which equals the output of tanh.
        data_dy: Tensor, the initial gradients.

    Return:
        Tensor, overall gradients.
    
    Supported Platforms:
        'Ascend'
    """
    dtype=data_y.dtype
    utils.ops_dtype_check(data_y.dtype, utils.DtypeForDavinci.ALL_FLOAT)
    shape = [x.value for x in data_y.shape]
    utils.check_shape(shape)

    # dx = dy * (1 - y*y)
    tmp1 = akg.tvm.const(-1, dtype=dtype)
    tmp2 = akg.tvm.const(1, dtype=dtype)
    data1_square = akg.lang.ascend.vmul(data_y, data_y)
    data_tmp = akg.lang.ascend.vmuls(data1_square, tmp1)
    anuminate = akg.lang.ascend.vadds(data_tmp, tmp2)
    res = akg.lang.ascend.vmul(anuminate, data_dy)

    return res
