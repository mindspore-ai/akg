#!/usr/bin/env python3
# coding: utf-8
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

"""prod_force_grad"""
import akg
import akg.utils as utils
from akg.tvm.hybrid import script


def prod_force_se_a_grad(grad_tensor, in_deriv_tensor, nlist_tensor, natoms=192, target=utils.CCE):
    """
    Supported Platforms:
        'Ascend'
    """
    if target != utils.CCE:
        raise RuntimeError('operator not supported on %s' % utils.get_backend(target))
    net_deriv_tensor_shape = grad_tensor.shape
    natoms = net_deriv_tensor_shape[1]
    nframes = net_deriv_tensor_shape[0]
    nnei = nlist_tensor.shape[2]
    ndescript = nnei * 4
    output_shape = [nframes, natoms, ndescript]

    @script
    def prod_force_se_a_grad_compute(grad_tensor, in_deriv_tensor, nlist_tensor):
        grad_net = output_tensor(output_shape, dtype=grad_tensor.dtype)

        for kk in range(nframes):
            for ii in range(natoms):
                for jj in range(nnei):
                    for aa in range(jj*4, jj*4+4):
                        grad_net[kk, ii, aa] = 0.0
                        for cc in range(3):
                            grad_net[kk, ii, aa] -= grad_tensor[kk, ii, cc] * in_deriv_tensor[kk, ii, aa, cc]
                            j_idx = nlist_tensor[kk, ii, jj]
                            if j_idx > -1:
                                grad_net[kk, ii, aa] += grad_tensor[kk, j_idx, cc] * in_deriv_tensor[kk, ii, aa, cc]
        return grad_net

    output = prod_force_se_a_grad_compute(grad_tensor, in_deriv_tensor, nlist_tensor)
    attrs = {'enable_post_poly_loop_partition': False,
             'enable_double_buffer': False,
             'enable_cover_protect_optimize': False,
             'enable_feature_library': True,
             'RewriteVarTensorIdx': False}
    if nframes.value > 1:
        attrs['dim'] = "0 0 1 1 0 1 192 1 0 2 1 1 0 3 1 1"
    else:
        attrs['dim'] = "0 0 192 1 0 1 1 1 0 2 1 1 0 3 1 1"

    return output, attrs
