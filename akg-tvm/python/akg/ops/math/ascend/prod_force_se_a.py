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

"""prod_force_se_a"""
import akg
import akg.utils as utils
from akg.tvm.hybrid import script


def prod_force_se_a(net_deriv_tensor, in_deriv_tensor, nlist_tensor, natoms=192, target=utils.CCE):
    """
    Supported Platforms:
        'Ascend'
    """
    if target != utils.CCE:
        raise RuntimeError('operator not supported on %s' % utils.get_backend(target))
    net_deriv_tensor_shape = net_deriv_tensor.shape
    nlist_tensor_shape = nlist_tensor.shape
    natoms = net_deriv_tensor_shape[1]
    nframes = net_deriv_tensor_shape[0]
    ndescript = net_deriv_tensor_shape[2]
    nnei = nlist_tensor_shape[2]

    output_shape = [nframes, natoms, 3]

    @script
    def prod_force_se_a_compute(net_deriv_tensor, in_deriv_tensor, nlist_tensor):
        force = output_tensor(output_shape, dtype=net_deriv_tensor.dtype)
        for kk in range(nframes):
            for ii in range(natoms):
                for cc in range(3):
                    force[kk, ii, cc] = 0.0
            for ii in range(natoms):
                for aa in range(ndescript):
                    for cc in range(3):
                        force[kk, ii, cc] -= net_deriv_tensor[kk, ii, aa] * in_deriv_tensor[kk, ii, aa, cc]
                for jj in range(nnei):
                    j_idx = nlist_tensor[kk, ii, jj]
                    if j_idx > -1:
                        for aa in range(jj*4, jj*4 + 4):
                            for cc in range(3):
                                force[kk, j_idx, cc] += net_deriv_tensor[kk, ii, aa] * in_deriv_tensor[kk, ii, aa, cc]
        return force
    output = prod_force_se_a_compute(net_deriv_tensor, in_deriv_tensor, nlist_tensor)
    attrs = {'enable_post_poly_loop_partition': False,
             'enable_double_buffer': False,
             'enable_feature_library': True,
             'RewriteVarTensorIdx': True}
    return output, attrs
