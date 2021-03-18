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

import akg.tvm
from akg.utils import custom_tiling as ct_util

set_dim_map_ = {
    str([18,128,128]):((1,1),(1,1),(128,1),(128,1)),
    str([36,128,128]):((1,1),(1,1),(128,1),(128,1)),
    str([2,128,128]):((1,1),(1,1),(128,1),(128,1)),
    str([4,128,128]):((1,1),(1,1),(128,1),(128,1)),
    str([8,128,128]):((1,1),(1,1),(128,1),(128,1)),
    str([16,128,128]):((1,1),(1,1),(128,1),(128,1)),
    str([9,128,128]):((1,1),(1,1),(128,1),(128,1)),
    str(([4,128,128],[1,64,64])):((1,1),(1,1),(128,1),(128,1)),
}

def set_dim_func_(input1):
    """
    Tiling func.

    Args:
        input1:tvm.Tensor of type float32, 3d shape.

    Returns:
        Tiling parameter.
    """
    shape1 = input1.shape
    hash_key = str((shape1))
    if hash_key in set_dim_map_.keys():
        print(set_dim_map_.keys())
    return ct_util.set_dims_by_key(hash_key, set_dim_map_), hash_key


def set_dim_func_1(input1, input2):
    """
    Tiling func.

    Args:
        input1:tvm.Tensor of type float32, 3d shape.

    Returns:
        Tiling parameter.
    """
    shape1 = input1.shape
    shape2 = input2.shape

    hash_key = str((shape1, shape2))
    if hash_key in set_dim_map_.keys():
        print(set_dim_map_.keys())
    return ct_util.set_dims_by_key(hash_key, set_dim_map_), hash_key


@akg.tvm.hybrid.script
def combine_matrix_0(input1):
    """
    Combine diag matrix.

    Args:
        inputs:tvm.Tensor of type float32, 3d shape.

    Returns:
        akg.tvm.Tensor with type float32
    """
    batch_dim = input1.shape[1]
    batch_size = input1.shape[0]
    matrix_dim = batch_size * batch_dim
    result = allocate((matrix_dim, matrix_dim), input1.dtype, 'local')

    for i in range(batch_size):
        for j in range(batch_size):
            if i != j:
                for m in range(batch_dim):
                    for n in range(batch_dim):
                        result[i * batch_dim + m, j * batch_dim + n] = 0.0
            else:
                for m in range(batch_dim):
                    for n in range(batch_dim):
                        result[i * batch_dim + m, i * batch_dim + n] = input1[i,m,n]
    return result


@akg.tvm.hybrid.script
def combine_matrix_4_128_1_64(input1, input2):
    """
    Combine diag matrix.

    Args:
        input1:tvm.Tensor of type float32 with shape [4,128,128].
        input2:tvm.Tensor of type float32 with shape [1,64,64].

    Returns:
        akg.tvm.Tensor of type float32 with shape [576,576]
    """
    batch_dim_1 = input1.shape[1]
    batch_size_1 = input1.shape[0]
    batch_dim_2 = input2.shape[1]
    batch_size_2 = input2.shape[0]

    matrix_dim = batch_size_1 * batch_dim_1 + batch_dim_2 * batch_size_2

    result = allocate((matrix_dim, matrix_dim), input1.dtype, 'local')

    for i in range(576):
        for j in range(576):
            result[i,j] = 0.0

    for i in range(64):
        for j in range(64):
            result[512 + i, 512 + j] = input2[0,i,j]

    for i in range(128):
        for j in range(128):
            result[128 * 0 + i, 128 * 0 + j] = input1[0,i,j]
    for i in range(128):
        for j in range(128):
            result[128 * 1 + i, 128 * 1 + j] = input1[1,i,j]
    for i in range(128):
        for j in range(128):
            result[128 * 2 + i, 128 * 2 + j] = input1[2,i,j]
    for i in range(128):
        for j in range(128):
            result[128 * 3 + i, 128 * 3 + j] = input1[3,i,j]
    return result


@ct_util.reg_set_dim_func(set_dim_func_)
def diag_combine_matrix_1(inputs):
    """
    Combine diag matrix.

    Args:
        inputs:tvm.Tensor of type float32, 3d shape.

    Returns:
        akg.tvm.Tensor with type float32
    """
    res = combine_matrix_0(inputs)
    return res


@ct_util.reg_set_dim_func(set_dim_func_1)
def diag_combine_matrix_2(input1, input2):
    """
    Combine diag matrix.

    Args:
        input1:tvm.Tensor of type float32 with shape [4,128,128].
        input2:tvm.Tensor of type float32 with shape [1,64,64].

    Returns:
        akg.tvm.Tensor of type float32 with shape [576,576]
    """
    res = combine_matrix_4_128_1_64(input1, input2)
    return res
