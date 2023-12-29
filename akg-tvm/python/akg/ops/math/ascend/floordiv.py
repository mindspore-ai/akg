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

"""operator dsl function:floordiv"""
import akg
import akg.utils as utils
from akg.utils.kernel_exec import product_is_mini
from ..reciprocal import reciprocal


@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, (str, type(None)))
def floor_div(data1, data2, target=utils.CCE):
    """
    Calculate x/y, and always returns an integer which is floored.

    Args:
        data1 (tvm.tensor.Tensor): Tensor of type float16, float32.
        data2 (tvm.tensor.Tensor): Tensor of type float16, float32.

    Returns:
        tvm.tensor.Tensor, has type of int32.

    Supported Platforms:
        'Ascend'
    """
    utils.ops_dtype_check([data1.dtype, data2.dtype], utils.DtypeForDavinci.ALL_FLOAT)
    shape1 = [x.value for x in data1.shape]
    utils.check_shape(shape1)
    shape2 = [x.value for x in data2.shape]
    utils.check_shape(shape2)

    if product_is_mini():
        rec = reciprocal(data2, high_precision=True, target=target)
        res = data1 * rec
    else:
        res = akg.topi.divide(data1, data2)
    res = akg.lang.ascend.floor(res)
    return res
