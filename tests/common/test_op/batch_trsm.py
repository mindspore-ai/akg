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

import akg
import akg.topi
import akg.tvm
from akg.utils import kernel_exec as utils
from akg.utils import validation_check as vc_util




@akg.tvm.hybrid.script
def _batch_trsm(input1, input2):
    dim = input1.shape[1]
    batch = input1.shape[0]
    L_T = allocate(input1.shape, input1.dtype, 'local')
    Identity = allocate(input1.shape, input1.dtype, 'local')
    temp_mem = allocate((batch, dim), input1.dtype, 'local')

    for batch_idx in range(batch):
        for i in range(dim):
            for j in range(dim):
                Identity[batch_idx,i,j] = input2[batch_idx,i,j]
                L_T[batch_idx, i, j] = input1[batch_idx,i,j]
        for i in range(dim):
            temp_mem[batch_idx, i] = L_T[batch_idx, dim-1, dim-1]
        for i in range(dim):
            temp_mem[batch_idx, i] = exp(log(temp_mem[batch_idx, i]) * (-1.0))
        for i in range(dim):
            Identity[batch_idx, dim-1, i] = Identity[batch_idx, dim-1, i] * (temp_mem[batch_idx, 0])
        for i in range(dim - 1):
            for j in range(dim):
                if j >= (dim - i - 1):
                    for k in range(dim):
                        temp_mem[batch_idx, k] = L_T[batch_idx, dim - i - 2, j] * Identity[batch_idx, j, k]
                    for k in range(dim):
                        Identity[batch_idx, dim - i - 2, k] = Identity[batch_idx, dim - i - 2, k] - temp_mem[batch_idx, k]

            for k in range(dim):
                temp_mem[batch_idx, k] = L_T[batch_idx, dim - i - 2, dim - i - 2]
            for k in range(dim):
                temp_mem[batch_idx, k] = exp(log(temp_mem[batch_idx, k]) * (-1.0))
            for k in range(dim):
                Identity[batch_idx, dim - i - 2, k] = Identity[batch_idx, dim - i - 2, k] * (temp_mem[batch_idx, k])

    return Identity


def batch_trsm(input1, input2):
    dtype1 = input1.dtype
    dtype2 = input2.dtype
    check_list=["float32"]
    if (not (dtype1.lower() in check_list)) or (not (dtype2.lower() in check_list)):
        raise RuntimeError("batch cholesky and trsm only support %s while dtype is %s" % (",".join(check_list), dtype1))
    shape1 = input1.shape
    vc_util.check_shape(shape1)
    shape2 = input2.shape
    vc_util.check_shape(shape2)

    L_T_inv = _batch_trsm(input1, input2)
    attr_map = {"pragma_disable_schedule_shift": 1}
    return L_T_inv, attr_map
