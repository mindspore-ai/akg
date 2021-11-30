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

"""operator dsl function: softmax_ad"""
import akg.tvm
import akg
from akg.ops.nn import softmax
from akg.utils import custom_tiling as ct_util


softmax_ad_set_dim_map = {
    # 1-D
    str(((16,), "float16", -1)): ((1, 16), (1, 16)),
    str(((64,), "float16", -1)): ((1, 16), (1, 16)),
    # 2-D
    str(((1, 16), "float16", -1)): ((1, 16), (1, 16), (1, 16)),
    str(((16, 64), "float16", -1)): ((1, 16), (1, 16), (1, 16)),
    str(((1, 5), "float16", -1)): ((1, 16), (1, 16), (1, 16)),  # softmax in mobilenet
    str(((1, 12), "float16", -1)): ((1, 16), (1, 16), (1, 16)),  # softmax in mobilenet
    str(((1, 1000), "float16", -1)): ((1, 16), (1, 16), (1, 16)),  # softmax in resnet
    str(((4298, 30522), "float16", -1)): ((1, 16), (1, 16), (1, 16)),  # logsoftmax in Bert # out of memory
    # 3-D
    str(((1, 16, 64), "float16", -1)): ((1, 16), (1, 16), (1, 16)),
    str(((1, 64, 128), "float16", -1)): ((1, 16), (1, 16), (1, 16)),
    str(((1, 64, 128), "float16", -1)): ((1, 16), (1, 16), (1, 16)),
    # 4-D
    str(((1, 16, 32, 64), "float16", -1)): ((1, 16), (1, 16), (1, 16), (1, 16)),
}




def softmax_ad_set_dim_func(head, data, axis):
    """Look up the softmax_ad_set_dim_map, and return hash_value, hash_key."""
    key = []
    key.append(tuple(data.shape))
    key.append(data.dtype)
    key.append(axis)
    hash_key = str(tuple(key))

    if hash_key in softmax_ad_set_dim_map.keys():
        return ct_util.set_dims(softmax_ad_set_dim_map[hash_key]), hash_key
    return "", hash_key


@ct_util.reg_set_dim_func(softmax_ad_set_dim_func)
def softmax_ad(head, data, axis):
    out_data = softmax.softmax(data, axis)
    _jacs = list(akg.differentiate(out_data, [data], head))
    return _jacs[0]


@ct_util.reg_set_dim_func(softmax_ad_set_dim_func)
def softmax_ad_optimized(head, data, axis=-1):
    """
    Computes the autodiff of softmax.

    Args:
        head (tvm.tensor.Tensor): Original differentiation values.
        data (tvm.tensor.Tensor): Input of softmax.
        axis (int): Along which axis softmax is performed.

    Returns:
        tvm.tensor.Tensor, the overall differentiation values.
    """
    def get_shape(pld):
        return [d.value for d in pld.shape]

    def temp_compute(shape, grad, sftmx_fwd, *indices):
        shp_len = len(shape)
        grad_index = indices[:(shp_len - 2)] + indices[-1:]
        sftmx_fwd_index = indices[:-1]
        temp = grad(*grad_index) * akg.tvm.expr.Select(indices[-1] == indices[-2],
                                                       sftmx_fwd(*sftmx_fwd_index) * (1 - sftmx_fwd(*sftmx_fwd_index)),
                                                       -sftmx_fwd(*sftmx_fwd_index) * sftmx_fwd(*grad_index))
        return temp

    def temp_sum_compute(shape, temp, *indices):
        kk = akg.tvm.reduce_axis((0, shape[-1]), name='kk')
        index = indices[:] + (kk,)
        temp_sum = akg.tvm.sum(temp(*index), axis=kk)
        return temp_sum

    def custom_softmax_fdiff(out, inputs, grad, ad_attrs, new_pld_array):
        data = inputs[0]
        shape = get_shape(data)
        sftmx_fwd = softmax.softmax(data, -1)[0]
        shape.append(shape[-1])

        temp = akg.tvm.compute(shape, lambda *indices: temp_compute(shape, grad, sftmx_fwd, *indices), name="softmax_select2")
        temp_sum = akg.tvm.compute(shape[:-1], lambda *indices: temp_sum_compute(shape, temp, *indices), name="softmax_ad2")
        return [temp_sum]

    l_up = softmax.softmax(data, axis)[0]

    # For the large expression tree's dl w.r.t. data (where softmax is embedded inside), use the default fdiff.
    # For softmax's dl w.r.t. data (note: l_up is not a direct dependency of data), use the custom_softmax_fdiff.
    # In this case, l_up is the same as l_up, and data same as data, but this needs not be the case.
    [dl_ddata] = akg.differentiate(l_up, [data], head, None, None, override={l_up: ([data], custom_softmax_fdiff)})
    attrs = {}
    return dl_ddata, attrs
