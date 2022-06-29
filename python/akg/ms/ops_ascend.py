# Copyright 2021-2022 Huawei Technologies Co., Ltd
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

"""ascend operators"""
import akg.utils as utils
import akg.ops.nn.ascend as nn
import akg.ops.math.ascend as math
import akg.ops.array.ascend as array
import akg.ops.state.ascend as state
import akg.ops.optimizers.ascend as optimizers
from akg.utils.format_transform import get_shape
from akg.ms.utils import reg_op


@reg_op("RealDiv", utils.CCE)
def real_div(x, y, target=utils.CCE):
    """RealDiv"""
    return math.RealDiv(x, y, target)


@reg_op("FloorDiv", utils.CCE)
def floor_div(x, y, target=utils.CCE):
    """FloorDiv"""
    return math.floor_div(x, y, target)


@reg_op("Argmax", utils.CCE)
def argmax(x, axis=-1, target=utils.CCE):
    """Argmax"""
    return math.argmax(x, axis, target=target)


@reg_op("SimpleMean", utils.CCE)
def simple_mean(x, target=utils.CCE):
    """SimpleMean"""
    return math.mean(x, axis=[2, 3], keepdims=True, target=target)


@reg_op("ReLU", utils.CCE)
def relu(x, target=utils.CCE):
    """ReLU"""
    return nn.Relu(x, target)


@reg_op("ZerosLike", utils.CCE)
def zeros_like(x, target=utils.CCE):
    """ZerosLike"""
    return nn.ZerosLike(x, target=target)


@reg_op("StridedSlice", utils.CCE)
def strided_slice(x, begin, end, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask,
                  shrink_axis_mask, target=utils.CCE):
    """StridedSlice"""
    return array.StridedSlice(x, begin, end, strides, begin_mask, end_mask, ellipsis_mask,
                              new_axis_mask, shrink_axis_mask, target=target)


@reg_op("SparseSoftmaxCrossEntropyWithLogits", utils.CCE)
def sparse_softmax_cross_entropy_with_logits(features, labels, is_grad=False, sens=1.0, target=utils.CCE):
    """sparse softmax cross entropy with logits"""
    if is_grad:
        return nn.sparse_softmax_cross_entropy_with_logits_ad(labels, features, reduction='mean', grad_scale=sens,
                                                              target=target)
    return nn.sparse_softmax_cross_entropy_with_logits(labels, features, reduction='mean', target=target)


@reg_op("Softmax", utils.CCE)
def softmax(x, axis=-1, target=utils.CCE):
    """Softmax"""
    return nn.Softmax(x, axis, target)


@reg_op("ReluGrad", utils.CCE)
def relu_grad(y_backprop, x, target=utils.CCE):
    """gradient of relu"""
    return nn.ReluAd(y_backprop, x, target)


@reg_op("ReduceMean", utils.CCE)
def reduce_mean(x, axis, keepdims, target=utils.CCE):
    """ReduceMean"""
    return math.mean(x, axis=axis, keepdims=keepdims, target=target)


@reg_op("ProdForceSeA", utils.CCE)
def prod_force_sea(net_deriv_tensor, in_deriv_tensor, nlist_tensor, natoms=192, target=utils.CCE):
    """ProdForceSeA"""
    return math.prod_force_se_a(net_deriv_tensor, in_deriv_tensor, nlist_tensor, natoms, target)


@reg_op("ProdForceSeAGrad", utils.CCE)
def prod_force_sea_grad(grad_tensor, in_deriv_tensor, nlist_tensor, natoms=192, target=utils.CCE):
    """ProdForceSeAGrad"""
    return math.prod_force_se_a_grad(grad_tensor, in_deriv_tensor, nlist_tensor, natoms, target)


@reg_op("OneHot", utils.CCE)
def one_hot(indices, depth, on_value, off_value, axis=-1, target=utils.CCE):
    """OneHot"""
    return array.OneHotV2(indices, on_value, off_value, depth, axis=axis, target=target)


@reg_op("SimpleMeanGrad", utils.CCE)
def simple_mean_grad(head, input_shape, target=utils.CCE):
    """SimpleMeanGrad"""
    return nn.MeanAd(head, input_shape, axis=[2, 3], keepdims=True, target=target)


@reg_op("MaxPoolWithArgmax", utils.CCE)
def max_pool_with_argmax(x, pad_mode="valid", window=1, pad=0, stride=1, target=utils.CCE):
    """MaxPoolWithArgmax"""
    window = int(window)
    stride = int(stride)
    pad = int(pad)
    kernel = (window, window)
    stride_ = (stride, stride)
    strategy = pad_mode.upper()
    if pad_mode.upper() == "PAD":
        strategy = [pad, pad, pad, pad]
    return nn.maxpool_with_argmax(x, kernel, stride_, strategy, target)


@reg_op("MatMul", utils.CCE)
def mat_mul(x1, x2, out_dtype, transpose_a=False, transpose_b=False, target=utils.CCE):
    """MatMul"""
    return math.matmul(x=x1, y=x2, b=None, out_dtype=out_dtype, left_format="zN", right_format="zN", out_format="zN",
                       transpose_x=transpose_a, transpose_y=transpose_b, target=target)


@reg_op("Conv2D", utils.CCE)
def conv_2d(x, x_shape, w, w_shape, pad_list, stride=1, dilation=1, target=utils.CCE):
    """Conv2D"""
    if len(pad_list) != 4:
        raise IndexError("Length of pad must be equal 4")
    pad_ = pad_list
    data = []
    data.append(x)
    data.append(w)
    fmap_shape = x_shape  # 4D
    filter_shape = w_shape  # 4D
    stride_ = [stride, stride]
    dilation_ = [dilation, dilation]
    return nn.Conv(data, fmap_shape, filter_shape, pad_, stride_, dilation_, use_bias=False, attrs=None, target=target)


@reg_op("LoadIm2Col", utils.CCE)
def load_im2col(x, ksizes, strides, target=utils.CCE):
    """LoadIm2Col"""
    import math as Math
    bs, c1, h, w, c0 = get_shape(x)
    stride_h, stride_w = strides
    k_w, k_h = ksizes
    dilation_h = 1
    dilation_w = 1
    h_out = Math.ceil(h / stride_h)
    w_out = Math.ceil(w / stride_w)
    pad_needed_h = max(0, (h_out - 1) * stride_h +
                       dilation_h * (k_h - 1) + 1 - h)
    pad_top = Math.floor(pad_needed_h / 2)
    pad_bottom = pad_needed_h - pad_top
    pad_needed_w = max(0, (w_out - 1) * stride_w +
                       dilation_w * (k_w - 1) + 1 - w)
    pad_left = Math.floor(pad_needed_w / 2)
    pad_right = pad_needed_w - pad_left
    pad_list = [pad_top, pad_bottom, pad_left, pad_right]
    return nn.LoadIm2col(x, ksizes, strides, pad=pad_list, target=target)


@reg_op("Four2Five", utils.CCE)
def four2five(x, data_format=None, dst_type="float16", target=utils.CCE):
    """from 4d(NCHW) to 5d(NC1HWC0)"""
    from akg.ms.utils import DEFAULT
    if data_format is None:
        data_format = [DEFAULT]
    if isinstance(data_format, list):
        data_format = data_format[0]
    if data_format == DEFAULT:
        data_format = "NCHW"
    return array.Four2Five(x, data_format, dst_type, target)


@reg_op("Five2Four", utils.CCE)
def five2four(x, shape4d, dst_type, output_format, target=utils.CCE):
    """from 5d(NC1HWC0) to 4d(NCHW)"""
    return array.Five2Four(x, shape4d, dst_type, output_format, target)


@reg_op("ConvBN1", utils.CCE)
def conv_bn1(x, x_shape, w, w_shape, pad_list, stride=1, dilation=1, target=utils.CCE):
    """ConvBN1"""
    if len(pad_list) != 4:
        raise IndexError("Length of pad must be equal 4")
    pad = pad_list
    data = []
    data.append(x)
    data.append(w)
    fmap_shape = x_shape  # 4D
    filter_shape = w_shape  # 4D
    stride = [stride, stride]
    dilation = [dilation, dilation]
    return nn.ConvBn1(data, fmap_shape, filter_shape, pad, stride, dilation, use_bias=False, attrs=None, target=target)


@reg_op("GatherV2", utils.CCE)
def gather_v2(params, indices, axis=0, target=utils.CCE):
    """gather version2"""
    return array.GatherV2(params, indices, axis, target)


@reg_op("LambApplyOptimizerAssign", utils.CCE)
def lamb_apply_optimizer_assign(grad, input_v, input_m, input_param, beta_1, one_minus_beta_1, beta_2, one_minus_beta_2,
                                epsilon, steps, do_use_weight,
                                weight_decay_rate, target=utils.CCE):
    """LambApplyOptimizerAssign"""
    return optimizers.LambApplyOptimizerAssign(grad, input_v, input_m, input_param, beta_1, one_minus_beta_1, beta_2,
                                               one_minus_beta_2, epsilon,
                                               steps, do_use_weight, weight_decay_rate, target)


@reg_op("FusedBN1", utils.CCE)
def fused_bn1(data, target=utils.CCE):
    """FusedBN1"""
    return nn.fused_bn1(data, target)


@reg_op("FusedBN2", utils.CCE)
def fused_bn2(mean, var_part, running_mean, running_var, momentum=0.8, target=utils.CCE):
    """FusedBN2"""
    return nn.fused_bn2(mean, var_part, running_mean, running_var, momentum, target=target)


@reg_op("FusedBN3", utils.CCE)
def fused_bn3(data, mean, variance, gamma, beta, eps=1e-3, target=utils.CCE):
    """FusedBN3"""
    return nn.fused_bn3(data, mean, variance, gamma, beta, eps, target=target)


@reg_op("ClearZero", utils.CCE)
def clear_zero(x, target=utils.CCE):
    """ClearZero"""
    return state.ClearZero(x, target)


@reg_op("BiasAdd", utils.CCE)
def bias_add(x, b, data_format=None, target=utils.CCE):
    """BiasAdd"""
    if data_format is None:
        data_format = ["NCHW"]
    return nn.BiasAdd(x, b, data_format[0], target)


@reg_op("BiasAddGrad", utils.CCE)
def bias_add_grad(dout, data_format=None, target=utils.CCE):
    """grediant of bias_add"""
    if data_format is None:
        data_format = ["NCHW"]
    dout_shape = get_shape(dout)
    return nn.bias_add_ad(dout, dout_shape, data_format[0], target)


@reg_op("BatchMatMul", utils.CCE)
def batch_matmul(x1, x2, transpose_a=False, transpose_b=False, target=utils.CCE):
    """use cube version matmul"""
    return math.matmul(x=x1, y=x2, b=None, out_dtype=x1.dtype,
                       left_format="zN", right_format="zN", out_format="zN",
                       transpose_x=transpose_a, transpose_y=transpose_b, target=target)


@reg_op("AssignAdd", utils.CCE)
def assign_add(ref, value, target=utils.CCE):
    """AssignAdd"""
    return state.AssignAdd(ref, value, target)


@reg_op("ApplyMomentum", utils.CCE)
def apply_momentum(variable, accumulation, learning_rate, gradient, momentum, use_nesterov=False,
                   gradient_scale=1.0, target=utils.CCE):
    """ApplyMomentum"""
    return optimizers.ApplyMomentum(variable, gradient, accumulation, learning_rate, momentum,
                                    use_nesterov=use_nesterov, grad_scale=gradient_scale, target=target)


@reg_op("EqualCount", utils.CCE)
def equal_count(x, y, target=utils.CCE):
    """EqualCount"""
    return math.equal_count(x, y, target)


@reg_op("BNGrad1", utils.CCE)
def bn_grad1(dy, data, mean, target=utils.CCE):
    """BNGrad1"""
    return nn.fused_bn_grad1(dy, data, mean, target)


@reg_op("BNGrad2", utils.CCE)
def bn_grad2(dgamma_red_hw, dbeta_red_hw, variance, gamma, eps=1e-3, data_shape=None, target=utils.CCE):
    """BNGrad2"""
    return nn.fused_bn_grad2(dgamma_red_hw, dbeta_red_hw, variance, gamma, eps, data_shape, target)


@reg_op("BNGrad3", utils.CCE)
def bn_grad3(dy, rs, dgamma_dx, dbeta_dx, data_minus_mean, target=utils.CCE):
    """BNGrad3"""
    return nn.fused_bn_grad3(dy, rs, dgamma_dx, dbeta_dx, data_minus_mean, target)


@reg_op("FusedBatchNorm", utils.CCE)
def fused_batch_norm(x, scale, b, mean, variance, momentum=0.99, epsilon=1e-3, data_format=None, target=utils.CCE):
    """FusedBatchNorm"""
    if data_format is None:
        from akg.ms.utils import DEFAULT
        data_format = [DEFAULT]
    if isinstance(data_format, list):
        data_format = data_format[0]
    attrs = {"momentum": momentum, "eps": epsilon, "is_training": True, "data_format": data_format,
             "axis": 1, "target": target}
    return nn.fused_batch_norm([x, scale, b, mean, variance], attrs)


@reg_op("FusedBatchNormGrad", utils.CCE)
def fused_batch_norm_grad(dy, x, scale, save_mean, save_inv_variance, data_format=None, target=utils.CCE):
    """gradient for fused batchnorm"""
    from akg.ms.utils import DEFAULT
    if data_format is None:
        data_format = [DEFAULT]
    if isinstance(data_format, list):
        data_format = data_format[0]
    eps = 1e-3
    return nn.fused_batch_norm_grad([dy, x, save_mean, save_inv_variance, scale], eps=eps, data_format=data_format,
                                    axis=1, target=target)


@reg_op("FusedBatchNormInfer", utils.CCE)
def fused_batch_norm_infer(x, scale, b, mean, variance, momentum=0.99, epsilon=1e-3, data_format=None,
                           target=utils.CCE):
    """inference mode of fuse batchnorm"""
    if data_format is None:
        from akg.ms.utils import DEFAULT
        data_format = [DEFAULT]
    if isinstance(data_format, list):
        data_format = data_format[0]
    attrs = {"momentum": momentum, "eps": epsilon, "is_training": False, "data_format": data_format,
             "axis": 1, "target": target}
    return nn.fused_batch_norm([x, scale, b, mean, variance], attrs)


@reg_op("Conv2DBackpropInput", utils.CCE)
def conv_2d_backprop_input(out_backprop, input_sizes, filter_0, filter_shape, pad_list, stride=1, dilation=1,
                           target=utils.CCE):
    """back propagation of 2d convolution on input"""
    if len(pad_list) != 4:
        raise IndexError("Length of pad must be equal 4")

    pad_ = pad_list
    data = []
    data.append(out_backprop)
    data.append(filter_0)
    fmap_shape = input_sizes
    filter_shape = filter_shape
    stride_ = [stride, stride]
    dilation_ = [dilation, dilation]

    return nn.conv_backprop_input(data, fmap_shape, filter_shape, pad_, stride_, dilation_, target=target)


@reg_op("Conv2DBackpropFilter", utils.CCE)
def conv_2d_backprop_filter(out_backprop, input_0, input_shape, filter_sizes, pad_list, stride=1, dilation=1,
                            target=utils.CCE):
    """back propagation of 2d convolution on filter"""
    if len(pad_list) != 4:
        raise IndexError("Length of pad must be equal 4")
    pad_ = pad_list
    data = []
    data.append(out_backprop)
    data.append(input_0)
    fmap_shape = input_shape
    filter_shape = filter_sizes
    stride_ = [stride, stride]
    dilation_ = [dilation, dilation]

    return nn.conv_backprop_filter(data, fmap_shape, filter_shape, pad_, stride_, dilation_, target=target)
