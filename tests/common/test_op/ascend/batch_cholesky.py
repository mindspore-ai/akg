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
import akg.utils as utils

def get_attrs():
    """
    get attrs config
    """
    attr_map = {
        "pragma_enable_schedule_max_constant": 1,
        "enable_pre_poly_loop_partition": False,
        "enable_post_poly_loop_partition": False,
    }
    return attr_map


@akg.tvm.hybrid.script
def batch_chol(input):
    dim = input.shape[1]
    batch = input.shape[0]
    res = allocate(input.shape, input.dtype, 'local')
    tmp_mem = allocate(input.shape, input.dtype, 'local')
    tmp_mem_1 = allocate((batch,dim), input.dtype, 'local')
    tmp2 = allocate((1,), input.dtype, 'local')
    for batch_idx in range(batch):
        for i in range(dim):
            for j in range(dim):
                res[batch_idx,i,j] = input[batch_idx, i, j]
        for i in range(dim):
            for j in range(dim):
                tmp_mem_1[batch_idx,j] = exp(log(res[batch_idx, i,j]) * (-0.5))
            for j in range(dim):
                res[batch_idx, i,j] = res[batch_idx,i,j] * tmp_mem_1[batch_idx, i]
            if i != (dim - 1):
                for j in range(dim - i - 1):
                    tmp2[0] = res[batch_idx,i,j + i + 1]
                    for k in range(dim):
                        tmp_mem[batch_idx,j, k] = res[batch_idx,i,k] * tmp2[0]
                    for k in range(dim):
                        res[batch_idx, j+i+1, k] = res[batch_idx, j+i+1,k] - tmp_mem[batch_idx,j,k]
    return res

def batch_cholesky(input, target=utils.CCE):
    dtype = input.dtype
    check_list=["float32"]
    if not (dtype.lower() in check_list):
        raise RuntimeError("cholesky only support %s while dtype is %s" % (",".join(check_list), dtype))
    shape = input.shape
    utils.check_shape(shape)
    # dim = shape[0]
    L = batch_chol(input)
    attrs = get_attrs()

    return L, attrs
