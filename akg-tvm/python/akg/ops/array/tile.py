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

"""operator dsl function: tile"""
import akg.tvm
import akg.topi
import akg.utils as utils


@utils.check_input_type(akg.tvm.tensor.Tensor, (list, tuple), (str, type(None)))
def tile(data, multiples, target=utils.CCE):
    """
    Repeats the data in the specified dimensions according to the multiples.

    Args:
        data (tvm.tensor.Tensor): Tensor of type float16, float32.
        multiples (Union[list, tuple]): Elements must be int. The number of repetitions.

    Returns:
        tvm.tensor.Tensor, has the same dtype as data.

    Supported Platforms:
        'Ascend', 'GPU', 'CPU'
    """
    utils.check_supported_target(target)
    shape = [x.value for x in data.shape]
    dtype = data.dtype
    utils.check_shape(shape)
    utils.ops_dtype_check(dtype, utils.DtypeForDavinci.ALL_TYPES)
    utils.check_int_list(multiples, "multiples")
    output = akg.topi.tile(data, multiples)
    return output
