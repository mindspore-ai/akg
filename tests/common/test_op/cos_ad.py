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

"""operator dsl function: cos_ad"""
import akg
from tests.common.test_op import cos


def cos_ad(head, a):
    """
    Computes cosine derivative value of a tensor.

    Args:
        head (tvm,tensor.Tensor): Tensor of type float16, float32
        a (tvm,tensor.Tensor): Tensor of type float16, float32

    Returns:
        akg.tvm.Tensor of same type and shape as inputs
    """
    b, attr = cos.cos(a)
    jacs = list(akg.differentiate(b, [a], head))
    return jacs[0], attr
