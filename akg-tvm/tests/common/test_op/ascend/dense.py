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

"""operator dsl function: dense"""

import akg.tvm

def dense(data, w, bias_data=None, bias=False, target="cce"):
    """
    Computes data * w if bias is False, else data * W + bias_data if bias is True.

    Args:
        data(akg.tvm.Tensor): Should be a 2D tensor of type float16 with shape(batch, in_dim).
        w(akg.tvm.Tensor): Should be a 2D tensor of same type as data with shape(out_dim, in_dim).
        bias_data(None, akg.tvm.Tensor): Could be None(if bias is False) or
                                         a 1D akg.tvm.Tensor of same type as data with shape(out_dim,).
        bias(bool): Specifies whether a bias vector will be used or not.

    Returns:
        2D akg.tvm.Tensor of same type as data with shape(batch, out_dim).
    """

    check_list = ["float16"]
    dtype = data.dtype
    if not dtype in check_list:
        raise TypeError("tile_cce only support %s while dtype is %s" % (",".join(check_list), dtype))

    d_shape = [x.value for x in data.shape]
    batch = d_shape[0]
    in_dim = d_shape[1]

    w_shape = [x.value for x in w.shape]

    if bias:
        out_dim = [x.value for x in bias_data.shape][0]
    else:
        out_dim = w_shape[0]

    k = akg.tvm.reduce_axis((0, in_dim), name='k')
    res = akg.tvm.compute((batch, out_dim), lambda i, j: akg.tvm.sum(data[i, k] * w[j, k], axis=k), name='M')

    if bias:
        res = akg.tvm.compute((batch, out_dim), lambda i, j: res[i, j] + bias_data[j])

    return res
