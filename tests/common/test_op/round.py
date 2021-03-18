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

"""operator dsl funnction:round"""
import akg.tvm
import akg
import akg.lang.cce

from akg.utils import validation_check as vc_util


def round_value(input):
    """
    rounds the values of a akg.tvm.tensor to the nearest even(integer), element-wise

    Args:
        input: akg.tvm.Tensor of type float16, float32

    Returns:
        akg.tvm.Tensor of same shape as input, of type int32

    Raises:
        ValueError: If the type of input is invalid.
    """
    dtype = input.dtype
    vc_util.ops_dtype_check(dtype, vc_util.DtypeForDavinci.ALL_FLOAT)

    shape = input.shape
    vc_util.check_shape(shape)

    if dtype == "float16":
        data_f16 = input
    else:
        data_f16 = akg.tvm.compute(shape, lambda *i: input(*i).astype("float16"), name="data_f16")

    res = akg.lang.cce.round(data_f16)

    return res
