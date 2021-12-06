# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

"""operator dsl function: erfc"""
from akg import tvm, topi
import akg.utils.dsl_create as dc
import akg.utils as utils

from tests.common.test_op.ascend.erf import erf

@utils.check_input_type(tvm.tensor.Tensor)
def erfc(input_x):
    r"""
    Computes the complementary error of input_x.

    .. math::
        \operatorname{erfc} (x) = 1 - \operatorname{erf} (x).

    Args:
        input_x (tvm.tensor.Tensor): Input tensor, only support float16, float32.

    Returns:
        tvm.tensor.Tensor with the same shape and dtype as input_x.
    """

    dtype = input_x.dtype

    utils.ops_dtype_check(dtype, utils.DtypeForDavinci.ALL_FLOAT)
    utils.check_shape(input_x.shape)

    erfc_res = topi.add(dc.one_const(dtype),
                        topi.multiply(dc.neg_one_const(dtype), erf(input_x)))

    return erfc_res
