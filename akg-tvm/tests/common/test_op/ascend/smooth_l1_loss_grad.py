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

"""operator dsl function: smooth_l1_loss_grad"""
import akg.tvm
import akg.topi
from akg import dim
from akg.dim import DIM
import akg.utils as utils
from akg.utils import custom_tiling as ct_util
from akg.utils.format_transform import get_shape
from akg.utils.kernel_exec import product_is_mini

smooth_l1_loss_grad_set_dim_map = {
    str(((32, 8732, 4), "float16", "int32")): ((1, 1), (236, 236), (4, 4)),
}


def smooth_l1_loss_grad_set_dim_func(_dloss, prediction, _target, anchor_samples,
                                     _sigma, _anchor_sample_correct):
    """dim function"""
    key = get_shape(prediction)
    hash_key = str((tuple(key), prediction.dtype, anchor_samples.dtype))
    return ct_util.set_dims_by_key(hash_key, smooth_l1_loss_grad_set_dim_map), hash_key


@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor,
                          akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor,
                          float, int)
def smooth_l1_loss_grad(dloss, prediction, tar, anchor_samples,
                        sigma, anchor_sample_correct):
    """
    do backprop for smooth L1 loss (Huber loss)

    Args:
        dloss (tvm.tensor.Tensor): Tensor [x,y], derivative of loss.
        prediction (tvm.tensor.Tensor): Tensor [x,y,z], output of the forward pass.
        tar (tvm.tensor.Tensor): Tensor [x,y,z], ground truth.
        anchor_samples (tvm.tensor.Tensor): Tensor [x,y], == anchor_sample_correct indicates correct classification, otherwise no meaning.
        sigma (float): Constant parameter.
        anchor_sample_correct (int): Constant parameter.

    Returns:
        dpredirection (tvm.tensor.Tensor): output tensor [x,y,z]
    """
    if len(dloss.shape) != len(anchor_samples.shape):
        raise RuntimeError("anchor_samples shape should equal to dloss shape!")
    if len(prediction.shape) != len(tar.shape):
        raise RuntimeError("prediction shape should equal to tar shape!")
    if (len(dloss.shape) + 1) != len(prediction.shape):
        raise RuntimeError("prediction shape should be dloss shape + 1!")

    out_shape = get_shape(prediction)
    original_dtype = dloss.dtype
    utils.ops_dtype_check(original_dtype, utils.DtypeForDavinci.ALL_FLOAT)

    dim_info, _ = smooth_l1_loss_grad_set_dim_func(
        dloss, prediction, tar, anchor_samples, sigma, anchor_sample_correct)
    attrs = {DIM: dim_info}

    if product_is_mini():
        dtype = "float16"
    else:
        dtype = original_dtype

    # unify the data type of tensors
    if dloss.dtype != dtype:
        dloss = akg.topi.cast(dloss, dtype)
    if prediction.dtype != dtype:
        prediction = akg.topi.cast(prediction, dtype)
    if tar.dtype != dtype:
        tar = akg.topi.cast(tar, dtype)
    if anchor_samples.dtype != dtype:
        anchor_samples = akg.topi.cast(anchor_samples, dtype)

    def eltwise_compute_func(_prediction, _target, _dloss, _anchor_sample, dtype):
        _diff = akg.tvm.expr.Sub(_prediction, _target)
        _first_branch = akg.tvm.expr.Mul(_diff, akg.tvm.const(sigma * sigma, dtype))
        _second_branch = akg.tvm.expr.Select(
            akg.tvm.const(0, dtype) < _diff, akg.tvm.const(1, dtype), akg.tvm.const(-1, dtype))
        _abs_diff = akg.tvm.expr.Mul(_second_branch, _diff)
        _derivative = akg.tvm.expr.Select(_abs_diff <= akg.tvm.const(
            1.0 / (sigma * sigma), dtype), _first_branch, _second_branch)
        _mult_dloss = akg.tvm.expr.Mul(_derivative, _dloss)
        _output = akg.tvm.expr.Select(
            _anchor_sample == anchor_sample_correct, akg.tvm.const(0, dtype), _mult_dloss)
        return _output

    dprediction = akg.tvm.compute(out_shape, lambda *i: eltwise_compute_func(
        prediction(*i), tar(*i), dloss(*i[:-1]), anchor_samples(*i[:-1]), dtype))

    if dprediction.dtype.lower() != original_dtype:
        dprediction = akg.topi.cast(dprediction, original_dtype)

    return dprediction, attrs


def smooth_l1_loss_grad_get_dim(shape):
    """
    get dim attr for smooth L1 loss grad

    Args:
        shape: the shape of prediction tensor (e.g. [8, 4718, 4])

    Returns:
        dim string for akg.op.build(attrs=...)
    """

    # example shape: [8, 4718, 4]
    # cut dim: ((1,1), (1024,1024))
    tensor_size = 1
    for i in shape[:-1]:
        tensor_size *= i
    # if tensor_size >= threshold, cut
    ub_size = 256 * 1024
    # estimated maximum number of data copies in UB
    num_data_copies = 32
    data_size = 4
    # do not cut the last dim
    max_tensor_size = int(ub_size / data_size / num_data_copies / shape[-1])

    if tensor_size > max_tensor_size:
        # find the largest divisor of tensor_size to be the tile size
        # currently the dim size must be divisible by tile size
        tile_size = 1
        for i in range(max_tensor_size, 1, -1):
            if tensor_size % i == 0:
                tile_size = i
                break

        # generate setdim string
        info = dim.Dim()
        # do not cut last dim
        for i in range(0, len(shape) - 2):
            info.setdim(index=0, axis=i, tilel1=1, tilel0=1)
        # cut -2 dim
        info.setdim(index=0, axis=len(shape) - 2,
                    tilel1=tile_size, tilel0=tile_size)
        return str(info)
    return ''
