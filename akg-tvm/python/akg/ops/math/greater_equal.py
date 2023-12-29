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

"""operator dsl function: greaterequal"""
import akg.tvm
import akg.topi
import akg.utils as utils


@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, (str, type(None)))
def greater_equal(data1, data2, target=utils.CCE):
    """
    Check whether input1 greaterquals to input2.

    Args:
        input1 (tvm.tensor.Tensor): Tensor.
        input2 (tvm.tensor.Tensor): Tensor.

    Returns:
        tvm.tensor.Tensor. If input1 greaterquals to input2 return True, else return False.

    Supported Platforms:
        'Ascend', 'GPU', 'CPU'
    """
    utils.check_supported_target(target)
    # check shapes
    shape1 = [x.value for x in data1.shape]
    shape2 = [x.value for x in data2.shape]
    shapes = [shape1, shape2]
    for _, shape in enumerate(shapes):
        utils.check_shape(shape)

    # check types
    dtype = data1.dtype
    dtype2 = data2.dtype
    utils.elemwise_dtype_check(dtype, dtype2)
    if target == utils.CCE:
        utils.ops_dtype_check(dtype, utils.DtypeForDavinci.FLOAT16)
    res = akg.topi.greater_equal(data1, data2)
    return res
