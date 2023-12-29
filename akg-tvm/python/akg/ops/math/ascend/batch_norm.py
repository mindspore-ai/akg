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

"""operator dsl function: batch_norm"""

import akg
import akg.tvm
import akg.utils as utils
from akg.utils.format_transform import get_shape

@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor,
                          akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor,
                          akg.tvm.tensor.Tensor,
                          float, (bool, type(None)), (dict, type(None)), (str, type(None)))
def batch_norm(data, mean, var, gamma, beta, eps, polyhedral=True, attrs=None):
    """
    Batch normalization.

    Args:
        data (tvm.tensor.Tensor): Tensor of type float16, float32.
        mean (tvm.tensor.Tensor): Tensor of type float16, float32 as mean.
        var (tvm.tensor.Tensor): Tensor of type float16, float32 as variance.
        gamma (tvm.tensor.Tensor): Tensor of type float16, float32 for scaling.
        beta (tvm.tensor.Tensor): Tensor of type float16, float32 for bias.
        eps (float): A small float added to variance to avoid dividing by zero.
        polyhedral (bool): Whether to schedule with polyhedral.
        attrs (dict): Schedule attributes for op.

    Returns:
        outs (tvm.tensor.Tensor): Tensor for normalized, scaled, shifted data.

    Supported Platforms:
        'Ascend'
    """
    for tensor in (data, mean, var, gamma, beta):
        utils.check_shape(get_shape(tensor))
        utils.ops_dtype_check(tensor.dtype, utils.DtypeForDavinci.ALL_FLOAT)
    shape = get_shape(data)
    dtype = data.dtype

    if len(shape) != 4 and len(shape) != 5:
        raise RuntimeError("Only support 4-dim OR 5-dim batch norm!")

    inp_eps = akg.tvm.const(eps, dtype=dtype)

    veps = akg.lang.ascend.vadds(var, inp_eps)

    power_num = akg.tvm.const(0.5, dtype=data.dtype)

    if len(shape) == 5:
        _, channel_1, _, _, channel_0 = data.shape

        new_shape = (channel_1, channel_0)
        vlog_t = akg.tvm.compute(new_shape,
                                 lambda c1, c0:
                                 akg.tvm.log(veps[0, c1, 0, 0, c0]),
                                 name="vlog_t")
        vmuls_t = akg.tvm.compute(
            new_shape, lambda c1, c0: vlog_t[c1, c0] * power_num, name="vmuls_t")
        sveps = akg.tvm.compute(new_shape,
                                lambda c1, c0: akg.tvm.exp(vmuls_t[c1, c0]),
                                name="sveps")
        mean2 = akg.lang.ascend.vmuls(mean, akg.tvm.const(-1, data.dtype))

        dmean = akg.tvm.compute(
            shape,
            lambda b, c1, h, w, c0:
            data[b, c1, h, w, c0] + mean2[0, c1, 0, 0, c0],
            name="dmean")
        rsveps = akg.tvm.compute(
            new_shape,
            lambda c1, c0: akg.tvm.const(1, data.dtype) / sveps[c1, c0],
            name="rsveps")
        dmsve = akg.tvm.compute(
            shape,
            lambda b, c1, h, w, c0: dmean[b, c1, h, w, c0] * rsveps[c1, c0],
            name="dmsve")
        dmsveg = akg.tvm.compute(
            shape,
            lambda b, c1, h, w, c0:
            dmsve[b, c1, h, w, c0] * gamma[0, c1, 0, 0, c0],
            name="dmsveg")
        outs = akg.tvm.compute(
            shape,
            lambda b, c1, h, w, c0:
            dmsveg[b, c1, h, w, c0] + beta[0, c1, 0, 0, c0],
            name="output")
    else:
        _, channel, _, _ = data.shape

        vlog_t = akg.tvm.compute(
            (channel,), lambda c: akg.tvm.log(veps[0, c, 0, 0]), name="vlog_t")
        vmuls_t = akg.tvm.compute(
            (channel,), lambda c: vlog_t[c] * power_num, name="vmuls_t")
        sveps = akg.tvm.compute(
            (channel,), lambda c: akg.tvm.exp(vmuls_t[c]), name="sveps")
        mean2 = akg.lang.ascend.vmuls(mean, akg.tvm.const(-1, data.dtype))

        dmean = akg.tvm.compute(shape,
                                lambda b, c, h, w:
                                data[b, c, h, w] + mean2[0, c, 0, 0],
                                name="dmean")
        rsveps = akg.tvm.compute((channel,),
                                 lambda c:
                                 akg.tvm.const(1, data.dtype) / sveps[c],
                                 name="rsveps")
        dmsve = akg.tvm.compute(shape,
                                lambda b, c, h, w:
                                dmean[b, c, h, w] * rsveps[c], name="dmsve")
        dmsveg = akg.tvm.compute(shape,
                                 lambda b, c, h, w:
                                 dmsve[b, c, h, w] * gamma[0, c, 0, 0],
                                 name="dmsveg")
        outs = akg.tvm.compute(shape,
                               lambda b, c, h, w:
                               dmsveg[b, c, h, w] + beta[0, c, 0, 0],
                               name="output")

    if polyhedral:
        return outs

    def comp_func(s):
        """schedule function"""
        data_ub = s.cache_read(data, "local.UB", [dmean])
        mean_ub = s.cache_read(mean, "local.UB", [mean2])
        gamma_ub = s.cache_read(gamma, "local.UB", [dmsveg])
        var_ub = s.cache_read(var, "local.UB", [veps])
        beta_ub = s.cache_read(beta, "local.UB", [outs])
        outs_ub = s.cache_write(outs, "local.UB")

        split_axis = {}
        for i in range(len(attrs["tile"])):
            split_axis["axis" + str(i)] = s[outs].split(outs.op.axis[i], attrs["tile"][i])

        split_axis_sorted = sorted(split_axis.items())

        s[data_ub].compute_at(s[outs], split_axis_sorted[-1][1][0])
        s[mean_ub].compute_at(s[outs], split_axis_sorted[-1][1][0])
        s[var_ub].compute_at(s[outs], split_axis_sorted[-1][1][0])
        s[gamma_ub].compute_at(s[outs], split_axis_sorted[-1][1][0])

        s[beta_ub].compute_at(s[outs], split_axis_sorted[-1][1][0])

        s[dmsveg].compute_at(s[outs], split_axis_sorted[-1][1][0])
        s[dmsve].compute_at(s[outs], split_axis_sorted[-1][1][0])
        s[rsveps].compute_at(s[outs], split_axis_sorted[-1][1][0])
        s[dmean].compute_at(s[outs], split_axis_sorted[-1][1][0])

        s[mean2].compute_at(s[outs], split_axis_sorted[-1][1][0])
        s[sveps].compute_at(s[outs], split_axis_sorted[-1][1][0])
        s[vmuls_t].compute_at(s[outs], split_axis_sorted[-1][1][0])
        s[vlog_t].compute_at(s[outs], split_axis_sorted[-1][1][0])

        s[veps].compute_at(s[outs], split_axis_sorted[-1][1][0])

        s[veps].set_scope("local.UB")
        s[vlog_t].set_scope("local.UB")
        s[vmuls_t].set_scope("local.UB")
        s[sveps].set_scope("local.UB")
        s[mean2].set_scope("local.UB")
        s[dmean].set_scope("local.UB")
        s[rsveps].set_scope("local.UB")
        s[dmsve].set_scope("local.UB")
        s[dmsveg].set_scope("local.UB")

        s[outs_ub].compute_at(s[outs], split_axis_sorted[-1][1][0])
    return outs, comp_func
