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

"""operator dsl function: fused_batch_norm_grad"""
from functools import reduce
import akg
import akg.utils as utils
from akg.dim import DIM
from akg.ops.math.add import add
from akg.utils import custom_tiling as ct_util
from akg.utils.kernel_exec import product_is_mini
from akg.utils.format_transform import get_shape
from akg.utils.validation_check import comp_output_params, check_inputs_in_rank, check_input_shape_equal_5


def check_inputs(inputs, data_format, axis):
    """check inputs"""
    if len(inputs) != 5:
        raise ValueError(
            "Input tensors number should be 5, but get %s." % len(inputs))
    dy = inputs[0]
    data = inputs[1]
    mean = inputs[2]
    var = inputs[3]
    gamma = inputs[4]

    utils.ops_dtype_check([data.dtype, dy.dtype],
                          utils.DtypeForDavinci.ALL_FLOAT)
    if data.dtype != dy.dtype:
        raise TypeError(
            "fused_batch_norm_grad require same dtype for dy and data.")
    utils.ops_dtype_check([mean.dtype, var.dtype, gamma.dtype],
                          utils.DtypeForDavinci.FLOAT32)

    dataformat_checklist = ["NHWC", "NC1HWC0", "NCHW", "DefaultFormat"]
    if data_format not in dataformat_checklist:
        raise RuntimeError(
            "fused_batch_norm_grad only support %s while data_format is %s" %
            (",".join(dataformat_checklist), data_format))

    shape = get_shape(data)
    in_rank = len(shape)
    is_special5d = (data_format == "NC1HWC0")

    axis = check_inputs_in_rank(data, axis, in_rank, data_format)

    if any((x.value != y.value for x, y in zip(data.shape, dy.shape))):
        raise RuntimeError("the shape of data and dy must be equal.")
    if is_special5d:
        check_input_shape_equal_5(data, shape, mean, var, gamma)
    else:
        if len(gamma.shape) != 1 or (gamma.shape[0].value != shape[axis]):
            raise RuntimeError("gamma mismatch the channel axis(while gamma "
                               "shape is {}, inputs shape is {}, and axis is {}"
                               ")".format(
                                   gamma.shape, data.shape, axis))
        if len(mean.shape) != 1 or (mean.shape[0].value != shape[axis]):
            raise RuntimeError("mean mismatch the channel axis(while mean "
                               "shape is {}, inputs shape is {}, and axis is "
                               "{})".format(
                                   mean.shape, data.shape, axis))
        if len(var.shape) != 1 or (var.shape[0].value != shape[axis]):
            raise RuntimeError("var mismatch the channel axis(while var shape "
                               "is {}, inputs shape is {}, and axis is {})"
                               "".format(
                                   var.shape, data.shape, axis))

    out_params = comp_output_params(is_special5d, shape, in_rank, axis)
    return out_params


set_dim_map_ = {
    # key: (rank, dtype, axis, is_special5d)
    str((2, "float16", 1, False)): ((0, 0, 1, 1), (0, 1, 1, 1), (1, 0, 0, 1), (1, 1, 1, 1)),
    str((2, "float32", 1, False)): ((0, 0, 0, 1), (0, 1, 1, 1), (1, 0, 0, 1), (1, 1, 1, 1)),
    str((5, "float16", 1, True)): ((0, 0, 0, 1), (0, 1, 0, 1), (0, 2, 1, 1), (0, 3, 1, 1), (0, 4, 1, 1),
                                   (1, 0, 0, 1), (1, 1, 0, 1), (1, 2, 1, 1), (1, 3, 1, 1), (1, 4, 1, 1)),
    str((5, "float32", 1, True)): ((0, 0, 0, 1), (0, 1, 0, 1), (0, 2, 1, 1), (0, 3, 1, 1), (0, 4, 1, 1),
                                   (1, 0, 0, 1), (1, 1, 0, 1), (1, 2, 1, 1), (1, 3, 1, 1), (1, 4, 1, 1)),
    str(((32, 4, 112, 112, 16), "float32")): ((1, 1), (16, 1), (1, 1), (1, 1), (112, 1)),
    str(((32, 4, 56, 56, 16), "float32")): ((1, 1), (16, 1), (1, 1), (1, 1), (56, 1)),
    str(((32, 16, 56, 56, 16), "float32")): ((1, 1), (16, 1), (1, 1), (1, 1), (56, 1)),
    str(((32, 8, 28, 28, 16), "float32")): ((1, 1), (16, 1), (1, 1), (1, 1), (28, 1)),
    str(((32, 32, 28, 28, 16), "float32")): ((1, 1), (16, 1), (1, 1), (1, 1), (28, 1)),
    str(((32, 16, 14, 14, 16), "float32")): ((1, 1), (16, 1), (1, 1), (1, 1), (14, 1)),
    str(((32, 64, 14, 14, 16), "float32")): ((1, 1), (16, 1), (1, 1), (1, 1), (14, 1)),
    str(((32, 32, 7, 7, 16), "float32")): ((1, 1), (16, 1), (1, 1), (1, 1), (7, 1)),
    str(((32, 128, 7, 7, 16), "float32")): ((4, 1), (16, 1), (1, 1), (1, 1), (7, 1)),
    str((4, "float16", 1, False)): ((0, 0, 0, 1), (0, 1, 1, 1), (0, 2, 1, 1), (0, 3, 1, 1),
                                    (1, 0, 0, 1), (1, 1, 1, 1), (1, 2, 1, 1), (1, 3, 1, 1)),
    str((4, "float32", 1, False)): ((0, 0, 0, 1), (0, 1, 1, 1), (0, 2, 1, 1), (0, 3, 1, 1),
                                    (1, 0, 0, 1), (1, 1, 1, 1), (1, 2, 1, 1), (1, 3, 1, 1)),
}


def get_attrs():
    """get attrs config"""
    attrs = {
        "pragma_checkcoincident": 0,
        "pragma_modshift": 1,
        "enable_bisect_optimize": 0,
    }
    return attrs


DTYPE_FLOAT32 = "float32"
DTYPE_FLOAT16 = "float16"


def set_dim_func_(inputs, data_format="DefaultFormat", axis=1):
    """set dim info"""
    shape = get_shape(input[1])
    params = check_inputs(inputs, data_format, axis)
    axis = params.get("axis", (0,))
    is_special5d = params.get("is_special5d", False)
    hash_key1 = str((tuple(shape), input[1].dtype))
    if hash_key1 in set_dim_map_:
        return ct_util.set_dims_by_key(hash_key1, set_dim_map_), hash_key1
    hash_key = str((len(shape), input[1].dtype, axis, is_special5d))
    return ct_util.set_dims_by_key(hash_key, set_dim_map_)


def sum_data(data, axes, keepdims):
    """sum one axis data at a time"""
    for x in axes:
        data = akg.topi.sum(data, axis=x, keepdims=keepdims)
    return data


@utils.check_input_type((list, tuple), (float, type(None)), (str, type(None)),
                        (int, list, tuple, type(None)), (str, type(None)))
def fused_batch_norm_grad(inputs, eps=1e-3, data_format="DefaultFormat", axis=1):
    r"""
    Gradient for fused_batch_norm.

    .. math::
        \begin{array}{ll} \\
            \hat{x_i} = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \\
            \frac {\partial L} {\partial \gamma} =
                \sum_{i=1}^m \\frac{\partial L}{\partial y_i} \cdot \hat{x_i} \\
            \frac {\partial L} {\partial \beta} =
                \sum_{i=1}^m \\frac{\\partial L}{\partial y_i} \\
            \frac {\partial L} {\partial x_i} =
                \frac{\gamma}{\sqrt{\sigma^2+\epsilon}}
                ( \frac{\partial L}{\partial y_i}
                - \frac{1}{m} \cdot \frac{\partial L}{\partial \beta}
                - \frac{\hat{x_i}}{m} \cdot \frac{\partial L}{\partial \gamma})
        \end{array}

    Note:
        When data_format is \"NC1HWC0\", the `mean`, `var` and `gamma` should be
        5D tensors of shape `(1, C1, 1, 1, C0)`, otherwise, they should be
        1D tensors of shape `(C,)`.

    Args:
        inputs:
            dy (tvm.tensor.Tensor): Tensor of type float16, float32
                                    (:math:`\\frac{\\partial L}{\\partial y_i}`).
            data (tvm.tensor.Tensor): Tensor of type float16, float32 (:math:`x`).
            mean (tvm.tensor.Tensor): Tensor, the sample mean of `data`
                                    (:math:`\\mu`).
            var (tvm.tensor.Tensor): Tensor, the sample variance of `data`
                                    (:math:`\\sigma`).
            gamma (tvm.tensor.Tensor): Tensor for scaling (:math:`\\gamma`).
            eps (float): A small float added to variance to avoid dividing by zero
                        (:math:`epsilon`).
        data_format (str): Supported format \"DefaultFormat\", \"NCHW\",
                           \"NHWC\" or \"NC1HWC0\".
        axis (Union[int, list, tuple]): An integer to specify the channel axis
                                        when data_format is \"DefaultFormat\".
                                        List or tuple for \"NC1HWC0\".
                                        Must be in the range
                                        [-rank(data), rank(data)).

    Returns:
        dx (tvm.tensor.Tensor): Tensor of the same shape and type as `data`,
                                representing
                                :math:`\frac {\partial L} {\partial x_i}.
        dgamma (tvm.tensor.Tensor): Tensor of the same shape and type as
                                    `gamma`, representing
                                    :math:`\frac{\partial L}{\partial\gamma}`.
        dbeta (tvm.tensor.Tensor): Tensor of the same shape and type as `gamma`,
                                   representing
                                   :math:`\frac{\partial L}{\partial \beta}`.
    """
    if len(inputs) != 5:
        raise ValueError(
            "Input tensors number should be 5, but get %s." % len(inputs))
    dy = inputs[0]
    data = inputs[1]
    mean = inputs[2]
    var = inputs[3]
    gamma = inputs[4]
    attrs = get_attrs()
    attrs[DIM] = set_dim_func_(inputs, data_format, axis)

    params = check_inputs(inputs, data_format, axis)

    shape = get_shape(data)
    ori_dtype = data.dtype
    is_special5d = params.get("is_special5d", False)
    mid_shape = params.get("mid_shape", [1, ])

    if ori_dtype != DTYPE_FLOAT32:
        dy = akg.topi.cast(dy, DTYPE_FLOAT32)
        data = akg.topi.cast(data, DTYPE_FLOAT32)
        gamma = akg.topi.cast(gamma, DTYPE_FLOAT32)
        mean = akg.topi.cast(mean, DTYPE_FLOAT32)
        var = akg.topi.cast(var, DTYPE_FLOAT32)

    mean = akg.lang.ascend.broadcast(akg.topi.reshape(mean, mid_shape), shape)
    gamma = akg.lang.ascend.broadcast(
        akg.topi.reshape(gamma, mid_shape), shape)

    v = akg.topi.reshape(akg.tvm.compute(var.shape, lambda *i: var(*i) +
                         akg.tvm.const(eps, var.dtype), name="var_plus_eps"), mid_shape)
    v = akg.topi.cast(v, DTYPE_FLOAT16) if product_is_mini() else v
    rsqvar = akg.tvm.compute(mid_shape, lambda *i: akg.tvm.exp(akg.tvm.log(v(*i)) * akg.tvm.const(-0.5, v.dtype)),
                             name="rsqvar",
                             attrs={'no_inline': 1})
    if product_is_mini():
        rsqvar = akg.topi.cast(rsqvar, DTYPE_FLOAT32)

    data_minus_mean = add(data, mean, -1.0, target=utils.CCE)[0]

    dbeta = sum_data(dy, params.get("axes"), keepdims=is_special5d)
    dbeta_bc = akg.lang.ascend.broadcast(
        akg.topi.reshape(dbeta, mid_shape), shape)

    dgamma_param = akg.tvm.compute(shape,
                                   lambda *i: dy(*i) * data_minus_mean(*i),
                                   name="dgamma_param")
    dgamma_param_sum = sum_data(
        dgamma_param, params.get("axes"), keepdims=True)
    dgamma = akg.tvm.compute(dgamma_param_sum.shape,
                             lambda *i: dgamma_param_sum(*i) * rsqvar(*i),
                             name="dgamma")
    dgamma_bc = akg.lang.ascend.broadcast(dgamma, shape)
    rsqvar_bc = akg.lang.ascend.broadcast(rsqvar, shape)
    if not is_special5d:
        dgamma = akg.topi.reshape(dgamma, dbeta.shape)

    neg_m_rec = akg.tvm.const((-1.0 / reduce(lambda i, j: i * j, (shape[i] for i in params.get("axes")))),
                              dtype=DTYPE_FLOAT32)

    def cal_dx(*i):
        c = neg_m_rec * rsqvar_bc(*i) * data_minus_mean(*i) * dgamma_bc(*i)
        return rsqvar_bc(*i) * gamma(*i) * (dy(*i) + neg_m_rec * dbeta_bc(*i) + c)

    dx = akg.tvm.compute(shape, cal_dx, name="dx")

    if DTYPE_FLOAT32 != ori_dtype:
        dgamma = akg.topi.cast(dgamma, ori_dtype)
        dbeta = akg.topi.cast(dbeta, ori_dtype)
        dx = akg.topi.cast(dx, ori_dtype)
    return dx, dgamma, dbeta, attrs
