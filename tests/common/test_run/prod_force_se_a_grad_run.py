# Copyright 2020 Huawei Technologies Co., Ltd
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
from akg.utils import kernel_exec as utils
import numpy as np
from tests.common.tensorio import compare_tensor
from tests.common.gen_random import random_gaussian
from tests.common.base import get_rtol_atol
from akg.ms.cce import prod_force_se_a_grad

import random


def gen_data(input_shapes):
    input1 = random_gaussian(input_shapes[0], miu=1, sigma=0.1)
    input2 = random_gaussian(input_shapes[1], miu=1, sigma=0.1)
    input3 = random_gaussian(input_shapes[2], miu=1, sigma=0.1) * 96

    input1 = input1.astype(np.float32)
    input2 = input2.astype(np.float32)
    input3 = input3.astype(np.int32)

    expect = prod_force_grad_cpu(input1, input2, input3)

    out_shape = expect.shape
    output = np.full(out_shape, np.nan, "float32")
    args = [input1, input2, input3, output]
    return args, expect, input1, input2, input3


def prod_force_grad_cpu(grad_tensor, in_deriv_tensor, nlist_tensor, natoms=192):
    net_deriv_tensor_shape = grad_tensor.shape
    nlist_tensor_shape = nlist_tensor.shape
    natoms = net_deriv_tensor_shape[1]
    nframes = net_deriv_tensor_shape[0]
    nnei = nlist_tensor_shape[2]
    ndescript = nnei * 4


    output_shape = [nframes, natoms, ndescript]
    
    grad_net = np.full(output_shape, 0, "float32")
    for kk in range(nframes):
        for ii in range(natoms):
            for jj in range(ndescript):
                grad_net[kk, ii, jj] = 0.0
    for kk in range(nframes):
        for ii in range(natoms):
            for aa in range(ndescript):
                for cc in range(3):
                    grad_net[kk, ii, aa] -= grad_tensor[kk, ii, cc] * in_deriv_tensor[kk, ii, aa, cc]
            
            for jj in range(nnei):
                j_idx = nlist_tensor[kk, ii, jj]
                if j_idx > -1:
                    for aa in range(jj*4, jj*4 + 4):
                        for cc in range(3):
                            grad_net[kk, ii, aa] += grad_tensor[kk, j_idx, cc] * in_deriv_tensor[kk, ii, aa, cc]
    return grad_net

def prod_force_se_a_grad_run(input_shapes, input_dtype, attrs = {}):
    attrs["pragma_disable_whole_component"] = False
    mod = utils.op_build_test(prod_force_se_a_grad.ProdForceSeAGrad, input_shapes, input_dtype, kernel_name = "force_grad", attrs = attrs)
    args, expect, input1, input2, input3 = gen_data(input_shapes)
    output = utils.mod_launch(mod, args, expect=expect)
    rtol, atol = get_rtol_atol("prod_force_se_a_grad", input_dtype)
    compare_res = compare_tensor(output, expect, rtol=rtol, atol=atol, equal_nan=True)
    return (input1, input2, input3), output, expect, compare_res
