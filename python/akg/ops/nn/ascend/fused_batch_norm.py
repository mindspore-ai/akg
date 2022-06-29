# Copyright 2019-2022 Huawei Technologies Co., Ltd
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

"""operator dsl function: fused batch norm"""
from __future__ import absolute_import
import akg
import akg.utils as utils
from akg.utils.kernel_exec import TensorUtils, BINDS
from akg.utils import custom_tiling as ct_util
from akg.utils.dsl_create import mul_axis_sum, update_by_moving_average
from akg.utils.format_transform import get_shape
from akg.ops.math.rsqrt import rsqrt
from akg.utils.dynamic_shape import shape_is_dynamic
from akg.utils.validation_check import comp_output_params, check_inputs_in_rank, check_input_shape_equal_5


def get_attrs(tensor):
    """get attrs config"""
    attrs_map = {
        "pragma_checkcoincident": 0,
        "pragma_modshift": 1,
        "disable_cse": 1,
        "enable_bisect_optimize": 0,
        "enable_remove_broadcast_copy": True,
    }
    if shape_is_dynamic(tensor):
        attrs_map["pragma_analyze_reuse_buffer"] = True
    return attrs_map


DTYPE_FLOAT32 = "float32"


def batch_norm_tiling_strategy_dynamic(tensor):
    """Custom tiling strategy for fused_batch_norm op with dynamic shape."""
    strategy = list()
    forbid_iso = False
    full_tile_reduce = False

    multicore_axis = 0
    c0_axis = 4
    w_axis = 3
    h_axis = 2
    c1_axis = 1

    for i, _ in enumerate(tensor.shape):
        if i in [w_axis, c0_axis]:
            strategy += ct_util.create_constraint_on_tensor(tensor=tensor,
                                                            values="FULL",
                                                            constraints=ct_util.TileConstraint.MAX,
                                                            tensor_pos=i)

        elif i == h_axis and full_tile_reduce:
            strategy += ct_util.create_constraint_on_tensor(tensor=tensor,
                                                            values="FULL",
                                                            constraints=ct_util.TileConstraint.MAX,
                                                            tensor_pos=i)
        elif i == c1_axis and full_tile_reduce:
            strategy += ct_util.create_constraint_on_tensor(tensor=tensor,
                                                            values=4,
                                                            constraints=ct_util.TileConstraint.FACTOR,
                                                            tensor_pos=i)
        elif i == multicore_axis:
            strategy += ct_util.create_constraint_on_tensor(tensor=tensor,
                                                            values=1,
                                                            constraints=ct_util.TileConstraint.FACTOR,
                                                            tensor_pos=i)
        elif forbid_iso:
            strategy += ct_util.create_constraint_on_tensor(tensor=tensor,
                                                            values=1,
                                                            constraints=ct_util.TileConstraint.FORBID_ISOLATE,
                                                            tensor_pos=i)

    return strategy


def batch_norm_tiling_strategy(tensor, tensor_format):
    """Custom tiling strategy for fused_batch_norm op"""
    if tensor_format == "DefaultFormat":
        return list()
    if tensor_format == "NC1HWC0":
        multi_core_axis = 1
        c0_axis = 4
        dim = 5
    elif tensor_format == "NHWC":
        multi_core_axis = 3
        c0_axis = None
        dim = 4
    else:
        multi_core_axis = 1
        c0_axis = None
        dim = 4

    strategy = list()
    if dim != 4 or get_shape(tensor)[multi_core_axis] != 1:
        strategy += ct_util.create_constraint_on_tensor(
            tensor=tensor,
            values=1,
            constraints=ct_util.TileConstraint.FACTOR,
            tensor_pos=multi_core_axis)
        if c0_axis:
            strategy += ct_util.create_constraint_on_tensor(
                tensor=tensor,
                values="FULL",
                constraints=ct_util.TileConstraint.MAX,
                tensor_pos=c0_axis)

        for i in range(dim):
            strategy += ct_util.create_constraint_on_tensor(
                tensor=tensor,
                values=1,
                constraints=ct_util.TileConstraint.FORBID_ISOLATE,
                tensor_pos=i)
        strategy.append(ct_util.modify_common_constraints(
            value=0.7, constraint=ct_util.TileConstraint.SET_MEM_RATIO))
    return strategy


def check_inputs(inputs, data_format, axis):
    """check inputs availability for fused_batch_norm op and get params"""
    if len(inputs) != 5:
        raise ValueError(
            "Input tensors number should be 5, but get %s." % len(inputs))
    data = inputs[0]
    gamma = inputs[1]
    beta = inputs[2]
    moving_mean = inputs[3]
    moving_variance = inputs[4]
    if any(data.dtype != t.dtype for t in
           [gamma, beta, moving_mean, moving_variance]):
        raise AssertionError("All input tensors should have same dtype!")
    utils.ops_dtype_check(data.dtype, utils.DtypeForDavinci.ALL_FLOAT)
    dataformat_list = ["NHWC", "NC1HWC0", "NCHW", "DefaultFormat"]
    if data_format not in dataformat_list:
        raise AssertionError("fused_batch_norm only support %s while data_format "
                             "is %s" % (",".join(dataformat_list), data_format))

    shape = get_shape(data)
    in_rank = len(shape)
    is_special5d = (data_format == "NC1HWC0")

    axis = check_inputs_in_rank(data, axis, in_rank, data_format)

    if is_special5d:
        check_input_shape_equal_5(
            data, shape,  gamma, beta, moving_mean, moving_variance)
    else:
        if len(gamma.shape) != 1 or (gamma.shape[0].value != shape[axis]):
            raise AssertionError("gamma mismatch the channel axis(while gamma "
                                 "shape is {}, input shape is {}, and axis is "
                                 "{})!!!".format(gamma.shape, data.shape, axis))
        if len(beta.shape) != 1 or (beta.shape[0].value != shape[axis]):
            raise AssertionError("beta mismatch the channel axis(while beta shape"
                                 " is {}, input shape is {}, and axis is "
                                 "{})!!!".format(beta.shape, data.shape, axis))
        if len(moving_mean.shape) != 1 \
                or (moving_mean.shape[0].value != shape[axis]):
            raise AssertionError("moving_mean mismatch the channel axis(while "
                                 "moving_mean shape is {}, input shape is {}, "
                                 "and axis is {})!!!".format(
                                     moving_mean.shape, data.shape, axis))
        if len(moving_variance.shape) != 1 \
                or (moving_variance.shape[0].value != shape[axis]):
            raise AssertionError("moving_variance mismatch the channel axis(while"
                                 " moving_variance shape is {}, input shape is "
                                 "{}, and axis is {})!!!".format(moving_variance.shape, data.shape, axis))

    out_params = comp_output_params(is_special5d, shape, in_rank, axis)
    return out_params


def sum_data(data, axes, keepdims, single_sum=False):
    """different solutions for sum multi axes"""
    if single_sum:
        data = akg.topi.sum(data, axis=axes, keepdims=keepdims)
    else:
        data = mul_axis_sum(data, axes, keepdims)
    return data


@utils.check_input_type((list, tuple), dict)
def fused_batch_norm(inputs, attrs):
    r"""
    Batch normalization.

    See Source:
    <a href="https://arxiv.org/abs/1502.03167">
        Batch Normalization: Accelerating Deep Network Training by Reducing
        Internal Covariate Shift; S. Ioffe, C. Szegedy.
    </a>

    .. math::
        \begin{array}{ll} \\
            \mu = \frac{1}{m} \sum^m_{i=1}{x_i} \\
            \sigma^2 = \frac{1}{m} \sum^m_{i=1}{(x_i-\mu)^2} \\
            \hat{x_i} = \frac{x_i - \mu}{ \sqrt{\sigma^2 + \epsilon} } \\
            y_i = \gamma \hat{x_i} + \beta \equiv BN_{\gamma, \beta}(x_i)
        \end{array}

    This momentum argument is different from one used in optimizer classes and
    the conventional notion of momentum. Mathematically, the update rule for
    running statistics here is

    .. math::
        \hat{z_{new}} = momentum \cdot \hat{z} + (1-momentum) \cdot z_t

    where :math:`\hat{z}` is the estimated statistic and :math:`z_t` is the
    new observed value.

    Note:
        When data_format is \"NC1HWC0\", the `gamma`, `beta`, `moving_mean`
        and `moving_variance` should be 5D tensors of shape
        `(1, C1, 1, 1, C0)`, otherwise, they should be 1D tensors
        of shape `(C,)`.

    Args:
        inputs:
            data (tvm.tensor.Tensor): Tensor of type float16, float32. (:math:`x_i`)
            gamma (tvm.tensor.Tensor): Tensor for scaling (:math:`\gamma`).
            beta (tvm.tensor.Tensor): Tensor for bias (:math:`\beta`).
            moving_mean (tvm.tensor.Tensor): Tensor for population mean used for
                                            inference.
            moving_variance (tvm.tensor.Tensor): Tensor for population variance used
                                             for inference.
        attrs:
            momentum (float): A float number used for the moving_mean and
                            moving_variance computation.
            eps (float): A small float added to variance to avoid dividing by zero.
            is_training (bool): A bool value to specify if the operation is used for
                                training or inference.
            data_format (str): Support format, \"DefaultFormat\", \"NCHW\", \"NHWC\"
                            or \"NC1HWC0\".
            axis (Union[int, list, tuple]): Integer to specify the channel axis when
                                            data_format is \"DefaultFormat\". List
                                            or tuple for \"NC1HWC0\". When format is
                                            \"NCHW\" or \"NHWC\", it's not work.
                                            Must be in the range
                                            [-rank(data), rank(data)).
            single_sum (bool): whether use "mul_axis_sum".

    Returns:
        outs (tvm.tensor.Tensor): Tensor for normalized, scaled, shifted data.
        new_moving_mean (tvm.tensor.Tensor): Tensor of same type and shape as
                                             `moving_mean`. The `moving_mean`
                                             updated by data. Only returns when
                                             `is_training` is True.
        new_moving_variance (tvm.tensor.Tensor): Tensor of same type and shape as
                                                 `moving_variance`. The
                                                 `moving_variance` updated by
                                                 data. Only returns when
                                                 `is_training` is True.
        sample_mean (tvm.tensor.Tensor): Tensor of same type and shape as
                                         `moving_mean`. The mean of `data`. Only
                                         returns when `is_training` is True.
        sample_var (tvm.tensor.Tensor): Tensor of same type and shape as
                                        `moving_variance`. The variance of `data`.
                                        Only returns when `is_training` is True.
    """
    if len(inputs) != 5:
        raise ValueError(
            "Input tensors number should be 5, but get %s." % len(inputs))
    data_format = attrs.get("data_format", "DefaultFormat")
    params = check_inputs(inputs, data_format, attrs.get("axis", 1))

    data = inputs[0]
    gamma = inputs[1]
    beta = inputs[2]
    moving_mean = inputs[3]
    moving_variance = inputs[4]
    ori_dtype = data.dtype
    shape = get_shape(data)
    axes = params.get("axes", (0,))
    keepdims = params.get("is_special5d", False)
    mid_shape = params.get("mid_shape", [1, ])
    data = akg.tvm.compute(data.shape, lambda *i: data(*i),
                           "batchnorm_" + data_format)
    ori_moving_mean = moving_mean
    ori_moving_variance = moving_variance
    if ori_dtype != DTYPE_FLOAT32:
        data = akg.topi.cast(data, DTYPE_FLOAT32)
        gamma = akg.topi.cast(gamma, DTYPE_FLOAT32)
        beta = akg.topi.cast(beta, DTYPE_FLOAT32)
        moving_mean = akg.topi.cast(moving_mean, DTYPE_FLOAT32)
        moving_variance = akg.topi.cast(moving_variance, DTYPE_FLOAT32)

    ######## following is dsl ########
    is_training = attrs.get("is_training", True)
    if is_training:
        value_num = 1
        for index in axes:
            value_num *= shape[index]

        avg_num = round(float(1) / float(value_num), 12)

        data_square = akg.tvm.compute(data.shape,
                                      lambda *i: data(*i) * data(*i),
                                      name="data_square")
        # cal mean
        data_mean = akg.lang.ascend.vmuls(
            sum_data(data, axes, keepdims, attrs.get("single_sum", False)), avg_num)
        data_square_mean = akg.lang.ascend.vmuls(sum_data(data_square, axes, keepdims, attrs.get("single_sum", False)),
                                                 avg_num)
        data_mean_square = akg.tvm.compute(data_mean.shape,
                                           lambda *i: data_mean(*i) *
                                           data_mean(*i),
                                           name="data_mean_square")

        data_variance = akg.tvm.compute(data_mean.shape,
                                        lambda *i:
                                        data_square_mean(
                                            *i) - data_mean_square(*i),
                                        name="data_variance")

        mean_new = update_by_moving_average(
            moving_mean, data_mean, attrs.get("momentum", 0.99))
        variance_new = update_by_moving_average(moving_variance,
                                                data_variance, attrs.get("momentum", 0.99))
    else:
        # no_bc version
        data_variance = moving_variance
        data_mean = moving_mean

    rsveps = akg.lang.ascend.vadds(data_variance, akg.tvm.const(
        attrs.get("eps", 1e-3), dtype=DTYPE_FLOAT32))
    rsveps = rsqrt(rsveps, utils.CCE)
    rsveps = akg.lang.ascend.broadcast(rsveps, shape)

    mean2 = akg.lang.ascend.vmuls(data_mean, akg.tvm.const(-1, data.dtype))
    mean2 = akg.lang.ascend.broadcast(mean2, shape)

    dmean = akg.tvm.compute(
        shape, lambda *i: data(*i) + mean2(*i), name="dmean")
    dmsve = akg.tvm.compute(shape, lambda *i: dmean(*i)
                            * rsveps(*i), name="dmsve")

    if not keepdims:
        gamma = akg.topi.reshape(gamma, mid_shape)
        beta = akg.topi.reshape(beta, mid_shape)
    gamma_bc = akg.lang.ascend.broadcast(gamma, shape)
    beta_bc = akg.lang.ascend.broadcast(beta, shape)
    dmsveg = akg.tvm.compute(shape, lambda *i: dmsve(*i) * gamma_bc(*i),
                             name="dmsveg")
    outs = akg.tvm.compute(shape, lambda *i: dmsveg(*i) + beta_bc(*i),
                           name="output")
    out_attrs = get_attrs(outs)

    if is_training:
        if ori_dtype != DTYPE_FLOAT32:
            outs = akg.topi.cast(outs, ori_dtype)
            mean_new = akg.topi.cast(mean_new, ori_dtype)
            variance_new = akg.topi.cast(variance_new, ori_dtype)
            data_mean = akg.topi.cast(data_mean, ori_dtype)
            data_variance = akg.topi.cast(data_variance, ori_dtype)

        mean_new, binds_info_mean = TensorUtils.inplace_set(
            ori_moving_mean, mean_new, buffer_name="mean_buf")
        variance_new, binds_info_var = TensorUtils.inplace_set(
            ori_moving_variance, variance_new, buffer_name="var_buf")
        binds_info_all = binds_info_mean
        binds_info_all.update(binds_info_var)
        out_attrs[BINDS] = binds_info_all

        # the new moving_mean and moving_var are updated inplace in
        # inputs(moving_mean and moving_var). But Mindspore needs
        # These two fake outputs though it never uses them
        fake_moving_mean = akg.tvm.compute(mean_new.shape,
                                           lambda *indices: mean_new(*indices),
                                           "fake_moving_mean")
        fake_moving_var = akg.tvm.compute(mean_new.shape,
                                          lambda *indices: variance_new(
                                              *indices),
                                          "fake_moving_var")
        out_tensors = (outs, fake_moving_mean, fake_moving_var, data_mean,
                       data_variance, mean_new, variance_new,)
    else:
        if ori_dtype != DTYPE_FLOAT32:
            outs = akg.topi.cast(outs, ori_dtype)
        out_tensors = (outs,)
    out_tensors = list(out_tensors) if isinstance(
        out_tensors, tuple) else out_tensors
    if shape_is_dynamic(out_tensors):
        out_attrs["custom_tiling"] = batch_norm_tiling_strategy_dynamic(outs)
    else:
        out_attrs["custom_tiling"] = batch_norm_tiling_strategy(
            outs, data_format)
    out_tensors.append(out_attrs)

    return out_tensors
