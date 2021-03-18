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

# import akg
# import akg.topi
import akg.tvm
# from akg.ops.cce import util
@akg.tvm.hybrid.script
def trace_extract_hybrid(input1):
    """
    Extract matrix's diag elements.

    Args:
        input1:tvm.Tensor of type float32 with 3d shape [1, matrix_dim, matrix_dim].

    Returns:
        akg.tvm.Tensor of type float32 with 2d shape [1, matrix_dim].
    """
    dim = input1.shape[1]
    trace_tensor = allocate((1,dim), input1.dtype, 'local')
    res1 = allocate(input1.shape, input1.dtype, 'local')
    for i in range(dim):
        for j in range(dim):
            res1[0,i,j] = input1[0,i,j]
    for j in range(dim):
        trace_tensor[0,j] = res1[0,j,j]
    return trace_tensor

def trace_extract(input1):
    """
    Extract matrix's diag elements.

    Args:
        input1:tvm.Tensor of type float32 with 3d shape [1, matrix_dim, matrix_dim].

    Returns:
        akg.tvm.Tensor of type float32 with 2d shape [1, matrix_dim].
    """
    trace_tensor = trace_extract_hybrid(input1)
    return trace_tensor
