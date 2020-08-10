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

"""operator dsl function:reduce_min_ad"""

import akg.tvm
import akg
import akg.backend as cce
from akg.utils import custom_tiling as ct_util
from akg.utils import kernel_exec as utils
from akg.ops.math.cast import cast
from test_op import reduce_min


reduce_min_ad_set_dim_map = {
}


def reduce_min_ad_set_dim_func(data, HEAD, axis, keepdims):
    key = []
    key.append(tuple(data.shape))
    key.append(tuple(axis))
    key.append(keepdims)
    hash_key = str(tuple(key))

    if hash_key in reduce_min_ad_set_dim_map.keys():
        return ct_util.set_dims(reduce_min_ad_set_dim_map[hash_key])
    else:
        return ""


@ct_util.reg_set_dim_func(reduce_min_ad_set_dim_func)
def reduce_min_ad(HEAD, data, axis, keepdims):
    B = reduce_min.reduce_min(data, axis, keepdims)
    _jacs = akg.differentiate(B, [data], HEAD)
    return _jacs[0]


def reduce_min_ad_optimized(HEAD, data, axis, keepdims):
    def get_shape(pld): return [d.value for d in pld.shape]

    def grad_compute(grad, *indices):
        indices_list = list(indices)
        axis_list = [x + len(indices_list) if x < 0 else x for x in list(axis)]

        if keepdims:
            grad_indices_list = [indices_list[i] if i not in axis_list else 0 for i in range(len(indices_list))]
        else:
            grad_indices_list = [indices_list[i] for i in range(len(indices_list)) if i not in axis_list]

        grad_ind = tuple(grad_indices_list)

        return grad(*grad_ind)

    def custom_reduce_min_fdiff(out, inputs, grad, ad_attrs, new_pld_array):
        data = inputs[0]
        shape = get_shape(data)
        min_ = akg.lang.cce.reduce_min(data, axis=axis, keepdims=keepdims)
        min_broadcast = akg.lang.cce.broadcast(min_, shape)
        return [akg.tvm.compute(shape,
                                lambda *indices:
                                akg.tvm.expr.Select(data(*indices) == min_broadcast(*indices),
                                                    grad_compute(grad, *indices),
                                                    akg.tvm.const(0, dtype=data.dtype)),
                                name="reduce_min_ad2")]

    L = reduce_min.reduce_min(data, axis, keepdims)

    [dL_ddata] = akg.differentiate(L, [data], HEAD, None, None, override={L: ([data], custom_reduce_min_fdiff)})
    return dL_ddata


def reduce_min_ad_optimized_manual_schedule(input_shape, dtype, axis, keepdims, polyhedral=True, attrs=None):
    def get_shape(pld): return [d.value for d in pld.shape]
    data = akg.tvm.placeholder(input_shape, dtype, name="input_data")

    #only works for last axis and 2D. Need to extend to multiple dimension and axes.
    def custom_reduce_min_fdiff(out, inputs, grad, ad_attrs, new_pld_array):
        data = inputs[0]
        shape = get_shape(data)
        if len(get_shape(data)) == 2:
            # add an extra stage to avoid alignment problem
            min_input = akg.tvm.compute(data.shape, lambda *i: data(*i), name="min_input")
            min_ = akg.lang.cce.reduce_min(min_input, axis=-1, keepdims=True)
            min_broadcast = akg.lang.cce.broadcast(min_, shape)
            if dtype != "float16":
                data = cast(data, "float16")
            return [akg.tvm.compute(shape,
                                    lambda i, j:
                                    akg.tvm.expr.Select(data[i, j] == min_broadcast[i, j],
                                                        grad[i], akg.tvm.const(0, dtype="float16")),
                                    name="reduce_min_ad2")]

    L = reduce_min.reduce_min(data, axis)
    head = akg.tvm.placeholder(L.shape, name="head", dtype=L.dtype)
    head_cast = cast(head, "float16")

    [dL_ddata] = akg.differentiate(L, [data], head_cast, None, None, override={L: ([data], custom_reduce_min_fdiff)})

    s = akg.tvm.create_schedule([dL_ddata.op])

    head_ub = s.cache_read(head, "local.UB", [head_cast])
    if dtype == "float16":
        data_ub = s.cache_read(data, "local.UB", [dL_ddata])
    else:
        data_ub = s.cache_read(data, "local.UB", [dL_ddata.op.input_tensors[0]])
        min_input_ub = s.cache_read(dL_ddata.op.input_tensors[1].op.input_tensors[0].op.input_tensors[0].op.input_tensors[0].op.input_tensors[0],
                                    "local.UB",
                                    [dL_ddata.op.input_tensors[1].op.input_tensors[0].op.input_tensors[0].op.input_tensors[0]])
        s[dL_ddata.op.input_tensors[1].op.input_tensors[0].op.input_tensors[0].op.input_tensors[0]].set_scope("local.UB")

    dL_ddata_ub = s.cache_write(dL_ddata, "local.UB")

    # tiling
    split_axis = {}
    for i in range(len(attrs['tile'])):
        split_axis["axis" + str(i)] = s[dL_ddata].split(dL_ddata.op.axis[i], attrs["tile"][i])

    split_axis_sorted = sorted(split_axis.items())

    if dtype == "float16":
        s[data_ub].compute_at(s[dL_ddata], split_axis_sorted[-1][1][0])
    else:
        s[data_ub].compute_at(s[dL_ddata], split_axis_sorted[-1][1][0])
        s[dL_ddata.op.input_tensors[0]].compute_at(s[dL_ddata], split_axis_sorted[-1][1][0])
        s[dL_ddata.op.input_tensors[0]].set_scope("local.UB")
        s[min_input_ub].compute_at(s[dL_ddata], split_axis_sorted[0][1][1])

    s[head_ub].compute_at(s[dL_ddata], split_axis_sorted[-1][1][0])
    s[head_cast].compute_at(s[dL_ddata], split_axis_sorted[-1][1][0])
    s[head_cast].set_scope("local.UB")
    s[dL_ddata.op.input_tensors[1]].compute_at(s[dL_ddata], split_axis_sorted[-1][1][0])
    s[dL_ddata.op.input_tensors[1]].set_scope("local.UB")
    s[dL_ddata.op.input_tensors[1].op.input_tensors[0]].compute_at(s[dL_ddata], split_axis_sorted[0][1][1])
    s[dL_ddata.op.input_tensors[1].op.input_tensors[0]].set_scope("local.UB")
    s[dL_ddata.op.input_tensors[1].op.input_tensors[0].op.input_tensors[0]].compute_at(s[dL_ddata], split_axis_sorted[0][1][1])
    s[dL_ddata.op.input_tensors[1].op.input_tensors[0].op.input_tensors[0]].set_scope("local.UB")

    # L is not being used for computation
    # s[L].compute_at(s[dL_ddata], split_axis_sorted[-1][1][0])
    # s[L].set_scope("local.UB"1

    s[dL_ddata_ub].compute_at(s[dL_ddata], split_axis_sorted[-1][1][0])

    with akg.build_config(add_lower_pass=cce.debug_mode(0), dump_pass_ir=True):
        mod = akg.build(s, [data, head, dL_ddata], "cce",
                        name="reduce_min_ad_manual_schedule",
                        attrs=attrs, polyhedral=polyhedral)
        source_code = mod.imported_modules[0].get_source()
        kernel_name = "reduce_min_ad_manual_schedule"
        utils.create_code(kernel_name, './', source_code)
    return mod
