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

"""operator dsl function:reduce_max_ad"""
import akg.tvm
import akg
import akg.utils as utils
from akg.utils.kernel_exec import debug_mode, create_code
from akg.utils import custom_tiling as ct_util
from akg.ops.math import Cast, reduce_max
from akg.utils import format_transform as ft_util


reduce_max_ad_set_dim_map = {
}


def reduce_max_ad_set_dim_func(data, head, axis, keepdims):
    key = []
    key.append(tuple(data.shape))
    key.append(tuple(axis))
    key.append(keepdims)
    hash_key = str(tuple(key))

    if hash_key in reduce_max_ad_set_dim_map.keys():
        return ct_util.set_dims(reduce_max_ad_set_dim_map[hash_key])
    else:
        return ""


@ct_util.reg_set_dim_func(reduce_max_ad_set_dim_func)
def reduce_max_ad(head, data, axis, keepdims):
    b = reduce_max(data, axis, keepdims, target=utils.CCE)
    _jacs = akg.differentiate(b, [data], head)
    return _jacs[0]


def reduce_max_ad_optimized(head, data, axis, keepdims, target="cce"):
    def get_shape(pld): return [d.value for d in pld.shape]

    def custom_reduce_max_fdiff(out, inputs, grad, ad_attrs, new_pld_array):
        data = inputs[0]
        shape = get_shape(data)
        max_ = akg.lang.ascend.reduce_max(data, axis=axis, keepdims=keepdims)
        max_broadcast = akg.lang.ascend.broadcast(max_, shape)
        return [akg.tvm.compute(shape,
                                lambda *indices:
                                akg.tvm.expr.Select(data(*indices) == max_broadcast(*indices),
                                                    grad(*get_reduced_indices(*indices, axis=axis, keepdims=keepdims)),
                                                    akg.tvm.const(0, dtype=data.dtype)),
                                name="reduce_max_ad2")]

    l = reduce_max(data, axis, keepdims, target=target)

    [dl_ddata] = akg.differentiate(l, [data], head, None, None, override={l: ([data], custom_reduce_max_fdiff)})
    return dl_ddata



def get_reduced_indices(*indices, axis, keepdims):
    """Get the adjoint for an arbitrary dimension input."""

    # get all indices
    indices_list = list(indices)
    # list of reduction axis: transform negative indices into positive
    # axis in this list wont exist after the reduction
    axis_list = ft_util.refine_reduce_axis(indices_list, list(axis))
    # get indices after reduction
    if keepdims:
        grad_indices_list = [index_i if i not in axis_list else 0 for i, index_i in enumerate(indices_list)]
    else:
        grad_indices_list = [index_i for i, index_i in enumerate(indices_list) if i not in axis_list]
    grad_ind = tuple(grad_indices_list)
    return grad_ind


def reduce_max_ad_optimized_manual_schedule(input_shape, dtype, axis, keepdims, polyhedral=True, attrs=None):

    def custom_reduce_max_fdiff(out, inputs, head_, ad_attrs, new_pld_array):
        data_ = inputs[0]
        shape = data_.shape
        # reduces maximum value for each column
        max_ = akg.lang.ascend.reduce_max(data_, axis=axis, keepdims=True)
        # copies reduced values to get the original shape
        max_broadcast = akg.lang.ascend.broadcast(max_, shape)
        # head broadcast is needed to generate correct cce code for the selection operation
        head_broadcast = akg.tvm.compute(shape,
                                         lambda *indices:
                                         head_(*get_reduced_indices(*indices, axis=axis, keepdims=keepdims)))
        # zero all the values that are not max values on the result, remaining is equal to the adjoint of the output
        max_values_and_zeros = akg.tvm.compute(shape,
                                           lambda *indices: akg.tvm.expr.Select(data_(*indices) == max_broadcast(*indices),
                                                                            head_broadcast(*indices),
                                                                            akg.tvm.const(0, dtype='float16')),
                                           name="reduce_max_ad2")
        # cast data back to the original dtype
        if dtype != 'float16':
            return [Cast(max_values_and_zeros, dtype, target=utils.CCE)]
        else:
            return [max_values_and_zeros]

    # tensor for the input data
    data = akg.tvm.placeholder(input_shape, dtype, name="input_data")

    # computation of reduce max
    # not used on the schedule because this is the diferentiation op
    l = reduce_max(data, axis, keepdims, target=utils.CCE)

    # adjoint tensor for the differentiation
    head = akg.tvm.placeholder(l.shape, name="head", dtype=l.dtype)

    # cast input data
    if dtype != 'float16':
        data_cast = Cast(data, "float16", target=utils.CCE)
        head_cast = Cast(head, "float16", target=utils.CCE)
    else:
        data_cast = data
        head_cast = head

    # override differentiation computation with custom function
    [dl_ddata] = akg.differentiate(l, [data_cast], head_cast, None, None,
                                   override={l: ([data_cast], custom_reduce_max_fdiff)})

    # get tensors from custom function
    if dtype != 'float16':
        max_values_and_zeros = dl_ddata.op.input_tensors[0]
        max_broadcast = max_values_and_zeros.op.input_tensors[1]
        max_ = max_broadcast.op.input_tensors[0]
        head_broadcast = max_values_and_zeros.op.input_tensors[2]
    else:
        max_broadcast = dl_ddata.op.input_tensors[1]
        max_ = max_broadcast.op.input_tensors[0]
        head_broadcast = dl_ddata.op.input_tensors[2]

    # schedule for differetiation operation
    # inputs: data and head
    s = akg.tvm.create_schedule([dl_ddata.op])

    # cache reads of inputs
    if dtype != 'float16':
        head_ub = s.cache_read(head, "local.UB", [head_cast])
        data_ub = s.cache_read(data, "local.UB", [data_cast])
    else:
        # no cast operation
        head_ub = s.cache_read(head_cast, "local.UB", [head_broadcast])
        data_ub = s.cache_read(data_cast, "local.UB", [max_, dl_ddata])

    # cache write for the output
    dl_ddata_ub = s.cache_write(dl_ddata, "local.UB")

    # get tiling attributes
    if attrs is None:
        raise Exception('attrs is None')
    tiling_factors = attrs['tile']
    split_iterators = []
    assert len(tiling_factors) == len(dl_ddata.shape)
    # split the final compute and save the iterators
    for index, factor in enumerate(tiling_factors):
        split_iterators.append(s[dl_ddata].split(dl_ddata.op.axis[index], factor))

    # get iterators
    iterator1 = split_iterators[0][0]

    # move computation of when there is a cast
    if dtype != "float16":
        s[data_cast].compute_at(s[dl_ddata], iterator1)
        s[data_cast].set_scope("local.UB")
        s[head_cast].compute_at(s[dl_ddata], iterator1)
        s[head_cast].set_scope("local.UB")
        s[max_values_and_zeros].compute_at(s[dl_ddata], iterator1)
        s[max_values_and_zeros].set_scope("local.UB")

    # move cache reads and writes
    s[data_ub].compute_at(s[dl_ddata], iterator1)
    s[head_ub].compute_at(s[dl_ddata], iterator1)
    s[dl_ddata_ub].compute_at(s[dl_ddata], iterator1)

    # move computation of the diferentiation
    s[max_].compute_at(s[dl_ddata], iterator1)
    s[max_].set_scope("local.UB")
    s[max_broadcast].compute_at(s[dl_ddata], iterator1)
    s[max_broadcast].set_scope("local.UB")
    s[head_broadcast].compute_at(s[dl_ddata], iterator1)
    s[head_broadcast].set_scope("local.UB")

    with akg.build_config(add_lower_pass=debug_mode(0), dump_pass_ir=True):
        mod = akg.build(s, [head, data, dl_ddata], "cce",
                        name="reduce_max_ad_manual_schedule", attrs=attrs, polyhedral=polyhedral)
        source_code = mod.imported_modules[0].get_source()
        kernel_name = "reduce_max_ad_manual_schedule"
        create_code(kernel_name, './', source_code)
    return mod
