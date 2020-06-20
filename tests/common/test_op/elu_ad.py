# Copyright 2020 Huawei Technologies Co., Ltd
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

"""operator dsl function: elu_ad"""

import akg
from test_op import elu

def elu_ad(head, x):
    """
    Computes elu_grad.

    Args:
        head (tvm.tensor.Tensor): Tensor of type float16, float32
        x (tvm.tensor.Tensor): Input of elu

    Returns:
        akg.tvm.Tensor of same type and shape as inputs
    """
    y = elu.elu(x)
    jacs = list(akg.differentiate(y, [x], head))
    return akg.lang.cce.cast_to(jacs[0], head.dtype)
