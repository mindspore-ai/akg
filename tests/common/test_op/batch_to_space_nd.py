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

"""batch to space"""

from functools import reduce
import akg.tvm
import akg.topi
from akg.utils import validation_check as vc_util
from akg.utils.format_transform import get_shape

def check_inputs(data, block_shape, crops):
    """check input shape and types"""
    vc_util.ops_dtype_check(data.dtype, vc_util.DtypeForDavinci.ALL_TYPES)
    vc_util.check_shape(data, tensor_name="data")
    vc_util.check_shape(block_shape, tensor_name="block_shape")
    if not isinstance(crops, (list, tuple)):
        raise RuntimeError("crops must be a 2D list or tuple.")
    for cs in crops:
        if not isinstance(cs, (list, tuple)) or len(cs) != 2:
            raise RuntimeError("crops must be a 2D list or tuple and the 2nd dim has length 2.")
        if cs[0] < 0 or cs[1] < 0:
            raise RuntimeError("all values in crops must be greater than or equal to zero.")
    vc_util.check_equal("length of block_shape", "length of crops", len(block_shape), len(crops))

@vc_util.check_input_type(akg.tvm.tensor.Tensor, (tuple, list), (tuple, list))
def batch_to_space_nd(data, block_shape, crops):
    """
    The N-D version of BatchToSpace.

    Rearrange batch data into spatial data blocks and then crop.

    Args:
        data (tvm.tensor.Tensor): Batch data of type float16.
        block_shape (Union[tuple, list]): 1-D shape of length `L`.
        crops (Union[tuple, list]): 2-D list of shape `[L][2]`, all values must be greater than or equal to zero.
            the i-th block will be cropped in `[crops[i][0] : -crops[i][1]]`.

    Returns:
        tvm.tensor.Tensor, Spatial data with the same type as data.
    """
    check_inputs(data, block_shape, crops)

    input_shape = get_shape(data)
    block_shape = list(block_shape)

    M = len(block_shape)
    batch = input_shape[0]
    prod_of_block_shape = reduce(lambda x, y: x * y, block_shape)

    # step 1
    reshaped_shape = block_shape + [batch // prod_of_block_shape] + input_shape[1:]
    reshaped = akg.topi.reshape(data, reshaped_shape)

    # step 2
    tran_axis = list()
    tran_axis.append(M)  # batch / prod(block_shape)
    for i in range(M):
        tran_axis.append(M + i + 1)  # input_shape[1...M]
        tran_axis.append(i)          # block_shape[0...M-1]
    tran_axis += list(range(len(tran_axis), len(reshaped_shape)))  # input_shape[M+1...N-1]
    permuted = akg.topi.transpose(reshaped, tran_axis)

    # step 3
    reshaped_permuted_shape = [batch // prod_of_block_shape]
    reshaped_permuted_shape += [input_shape[i + 1] * block_shape[i] for i in range(M)]
    reshaped_permuted_shape += input_shape[M + 1:]
    reshaped_permuted = akg.topi.reshape(permuted, reshaped_permuted_shape)

    # step 4
    out_shape = [batch // prod_of_block_shape]
    out_shape += [(reshaped_permuted_shape[i + 1] - crops[i][0] - crops[i][1]) for i in range(M)]
    out_shape += input_shape[M + 1:]
    output = akg.tvm.compute(out_shape,
        lambda *i: reshaped_permuted(i[0], *[i[j + 1] + crops[j][0] for j in range(M)], *i[M + 1:]),
        name="result")
    return output
