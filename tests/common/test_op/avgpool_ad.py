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

"""operator dsl function: avgpool_ad."""
import akg
from akg.utils import custom_tiling as ct_util
from akg.utils.format_transform import get_shape
from akg.ops.nn import avgpool

avgpool_ad_set_dim_map = {
}


def avgpool_ad_set_dim_func(head, data, kernel, stride, pad):
    """set dim info in attrs by avgpool_ad_set_dim_map."""
    key = []
    key.append(tuple(data.shape))
    key.append(kernel)
    key.append(stride)
    key.append(pad)
    key.append(data.dtype)
    hash_key = str(tuple(key))

    if hash_key in avgpool_ad_set_dim_map.keys():
        return ct_util.set_dims(avgpool_ad_set_dim_map[hash_key]), hash_key
    else:
        return "", hash_key


@ct_util.reg_set_dim_func(avgpool_ad_set_dim_func)
def avgpool_ad(head, data, kernel, stride, pad):
    """Compute gradient of avgpool operator using automatic differentiate."""
    attrs = {"enable_post_poly_loop_partition": False, "enable_pre_poly_loop_partition": False,
             "pragma_reschedule": 1}
    avgpool_fwd, _ = avgpool.avgpool(data, kernel, stride, pad)
    [dl_ddata] = akg.differentiate(avgpool_fwd, [data], head)
    return dl_ddata, attrs


@ct_util.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor,
                          (list, tuple), (list, tuple), (str, list, tuple))
def avgpool_ad_no_custom_diff_manual_schedule(head, data, kernel, stride, pad):
    """automatic differentiate of avgpool with manual schedule."""
    attrs = {"enable_post_poly_loop_partition": False, "enable_pre_poly_loop_partition": False}
    avgpool_fwd, _ = avgpool.avgpool(data, kernel, stride, pad)
    [dl_ddata] = akg.differentiate(avgpool_fwd, [data], head)
    # schedule for differetiation operation
    s = akg.tvm.create_schedule([dl_ddata.op])

    kh, kw = kernel
    shape = get_shape(data)
    ib, ic1, ih, iw, ic0 = shape

    if kh == ih and kw == iw:
        pad2d_input_2_grad = dl_ddata
        res_value_res_grad = pad2d_input_2_grad.op.input_tensors[0]
        head = res_value_res_grad.op.input_tensors[0]

        def comp_func(s):
            head_ub = s.cache_read(head, "local.UB", [res_value_res_grad])
            result_ub = s.cache_write(pad2d_input_2_grad, "local.UB")

            s[res_value_res_grad].set_scope("local.UB")

            b, c1, h, w, c0 = pad2d_input_2_grad.op.axis
            s[head_ub].compute_at(s[pad2d_input_2_grad], b)
            s[res_value_res_grad].compute_at(s[pad2d_input_2_grad], b)
            s[result_ub].compute_at(s[pad2d_input_2_grad], b)
    else:
        pad2d_input_2_grad = dl_ddata
        Broadcast_jac = pad2d_input_2_grad.op.input_tensors[0]
        res_value_res_grad = Broadcast_jac.op.input_tensors[0]
        head = res_value_res_grad.op.input_tensors[0]

        def comp_func(s):
            head_ub = s.cache_read(head, "local.UB", [res_value_res_grad])
            result_ub = s.cache_write(pad2d_input_2_grad, "local.UB")

            s[Broadcast_jac].set_scope("local.UB")
            s[res_value_res_grad].set_scope("local.UB")

            b, c1, h, w, c0 = result_ub.op.axis
            s[result_ub].reorder(*result_ub.op.reduce_axis, b, c1, h, w, c0)

            s[Broadcast_jac].compute_at(s[result_ub], b)

    return dl_ddata, comp_func, attrs
