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

"""operator dsl function: smooth_l1_loss"""
import akg.tvm
import akg.topi
import akg.utils as utils
from akg.utils import custom_tiling as ct_util
from akg.dim import DIM
from akg.utils.format_transform import get_shape
from akg.ops.math import Cast
from akg.utils.kernel_exec import product_is_mini

smooth_l1_loss_set_dim_map = {
    str(((32, 8732, 4), "float16", "int32")): ((1, 1), (236, 236), (4, 4)),
}


def smooth_l1_loss_set_dim_func(prediction, _target, anchor_samples,
                                _anchor_sample_correct, _delta):
    """dim function"""
    key = get_shape(prediction)

    hash_key = str((tuple(key), prediction.dtype, anchor_samples.dtype))
    return ct_util.set_dims_by_key(hash_key, smooth_l1_loss_set_dim_map), hash_key


@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor,
                          akg.tvm.tensor.Tensor,
                          (int, type(None)), (float, type(None)))
def smooth_l1_loss(prediction, targets, anchor_samples,
                   anchor_sample_correct=0, delta=1.0):
    """
    Smooth l1 loss.

    For each value x in `error=predictions-target`, the following is calculated:

    .. math::
        y = \\left\\{
            \\begin{array}{rcl}
                0.5 x^2, & & if \\left| x \\right| <= d \\\\
                0.5 d^2 + d \\cdot (\\left| x \\right| - d), & & if \\left| x \\right| > d
            \\end{array}
        \\right.

    `anchor_samples` acts as a condition for the loss.
    if anchor_samples == anchor_sample_correct, loss = 0, else loss=loss(attention pls)

    Args:
        prediction (tvm.tensor.Tensor): A float tensor of shape
            [batch_size, num_anchors, code_size] representing the (encoded)
            predicted locations of objects.
        targets (tvm.tensor.Tensor): A float tensor of shape
            [batch_size, num_anchors, code_size]
            representing the regression targets
        anchor_samples (tvm.tensor.Tensor): A int tensor of shape [batch_size, num_anchors]
        anchor_sample_correct (int): int, the threshold of anchor_samples
        delta (float): float, the point where the loss function changes from a quadratic to linear.

    Returns:
        loss (tvm.tensor.Tensor): A float tensor of shape [batch_size, num_anchors] tensor
               representing the value of the loss function.
    """
    dim_info, _ = smooth_l1_loss_set_dim_func(
        prediction, targets, anchor_samples, anchor_sample_correct, delta)
    attrs = {DIM: dim_info}

    prediction_dtype = prediction.dtype
    target_dtype = targets.dtype
    anchor_samples_dtype = anchor_samples.dtype

    utils.elemwise_dtype_check(prediction_dtype, target_dtype,
                                 utils.DtypeForDavinci.ALL_FLOAT)
    utils.ops_dtype_check(anchor_samples_dtype,
                            [utils.DtypeForDavinci.INT8,
                             utils.DtypeForDavinci.INT32])

    if anchor_sample_correct > 5 or anchor_sample_correct < 0:
        raise ValueError("anchor_sample_correct attr only support [0,5]")

    # check shape dim
    prediction_shape = get_shape(prediction)
    if len(prediction_shape) != 3:
        raise RuntimeError("Prediction shape only support 3-dim!")

    target_shape = get_shape(targets)
    if len(target_shape) != 3:
        raise RuntimeError("Target shape only support 3-dim!")

    anchor_samples_shape = get_shape(anchor_samples)
    if len(anchor_samples_shape) != 2:
        raise RuntimeError("weights shape only support 2-dim!")

    prediction_dtype_old = prediction_dtype

    if product_is_mini() and prediction_dtype == 'float32':
        prediction = akg.topi.cast(prediction, "float16")
        targets = akg.topi.cast(targets, "float16")
        prediction_dtype = "float16"

    # cast anchor_samples to float type in order to use the vcmp instruction
    if anchor_samples.dtype.lower() != prediction_dtype.lower():
        anchor_samples = Cast(anchor_samples, prediction_dtype, target=utils.CCE)
    anchor_samples_dtype = anchor_samples.dtype.lower()

    coefficient = akg.tvm.const(0.5, dtype=prediction_dtype)
    delta = akg.tvm.const(delta, dtype=prediction_dtype)

    error = akg.topi.subtract(prediction, targets)
    abs_error = akg.topi.abs(error)
    quadratic = akg.topi.minimum(abs_error, delta)
    linear = akg.topi.subtract(abs_error, quadratic)
    loss = akg.topi.add(akg.topi.multiply(coefficient, akg.topi.multiply(
        quadratic, quadratic)), akg.topi.multiply(delta, linear))
    loss = akg.topi.sum(loss, axis=-1)
    loss = akg.tvm.compute(loss.shape,
                           lambda *i:
                           akg.tvm.expr.Select(
                               anchor_samples(*i) == anchor_sample_correct,
                               akg.tvm.const(0, loss.dtype),
                               loss(*i)),
                           name="loss")

    if product_is_mini() and prediction_dtype_old == 'float32':
        loss = akg.topi.cast(loss, prediction_dtype_old)

    return loss, attrs
