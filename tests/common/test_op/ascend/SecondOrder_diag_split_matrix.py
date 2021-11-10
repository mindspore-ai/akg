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

import akg.tvm

split_dim=128

@akg.tvm.hybrid.script
def split_matrix_0(input1):
    """
    Split matrix.

    Args:
        inputs:tvm.Tensor of type float32 with shape [4608,4608].

    Returns:
        result_1:tvm.Tensor of type float32 with shape [32,128,128].
    """
    result_1 = allocate((32, split_dim, split_dim), input1.dtype, 'local')

    for i in range(32):
        for j in range(split_dim):
            for k in range(split_dim):
                result_1[i,j,k] = input1[i * split_dim + j, i * split_dim + k]
    return result_1

@akg.tvm.hybrid.script
def split_matrix_1(input1):
    """
    Split matrix.

    Args:
        inputs:tvm.Tensor of type float32 with shape [4608,4608].

    Returns:
        result_2:tvm.Tensor of type float32 with shape [4,128,128].
    """
    result_2 = allocate((4, split_dim, split_dim), input1.dtype, 'local')
    for i in range(4):
        for j in range(split_dim):
            for k in range(split_dim):
                result_2[i,j,k] = input1[(i + 32) * split_dim + j, (i + 32) * split_dim + k]

    return result_2

@akg.tvm.hybrid.script
def split_matrix_2(input1):
    """
    Split matrix.

    Args:
        inputs:tvm.Tensor of type float32.

    Returns:
        akg.tvm.Tensor of type float32 with 3d shape.
    """
    dim = input1.shape[0]
    split_num = dim // split_dim
    result_3 = allocate((split_num, split_dim, split_dim), input1.dtype, 'local')
    for i in range(split_num):
        for j in range(split_dim):
            for k in range(split_dim):
                result_3[i,j,k] = input1[i * split_dim + j, i * split_dim + k]
    return result_3

@akg.tvm.hybrid.script
def split_matrix_3(input1):
    """
    Split matrix.

    Args:
        inputs:tvm.Tensor of type float32.

    Returns:
        akg.tvm.Tensor of type float32 with 3d shape.
    """
    dim = input1.shape[0]
    res_dim = dim - (dim // split_dim) * split_dim
    split_block = dim // split_dim
    result_3 = allocate((1, res_dim, res_dim), input1.dtype, 'local')

    for j in range(split_dim):
        for k in range(split_dim):
            result_3[0,j,k] = input1[split_block * split_dim + j, split_block * split_dim + k]
    return result_3

def diag_split_matrix_4608(inputs, target="cce"):
    """
    Split matrix.

    Args:
        inputs:tvm.Tensor of type float32 with shape [4608,4608].

    Returns:
        out1:tvm.Tensor of type float32 with shape [32,128,128].
        out2:tvm.Tensor of type float32 with shape [4,128,128].
    """
    out1 = split_matrix_0(inputs)
    out2 = split_matrix_1(inputs)
    return out1, out2

def diag_split_matrix_576(inputs, target="cce"):
    """
    Split matrix.

    Args:
        inputs:tvm.Tensor of type float32 with shape [576,576].

    Returns:
        out1:tvm.Tensor of type float32 with 3d shape [4,128,128].
        out2:tvm.Tensor of type float32 with 3d shape [1,64,64].
    """
    out1 = split_matrix_2(inputs)
    out2 = split_matrix_3(inputs)
    return out1,out2

def diag_split_matrix_small(inputs, target="cce"):
    """
    Split matrix.

    Args:
        inputs:tvm.Tensor of type float32.
    Note:
        inputs dim must be divisible by 128 and inputs dim must be less than 32 * 128.
    Returns:
        akg.tvm.Tensor of type float32 with 3d shape.
    """
    out1 = split_matrix_2(inputs)
    return out1