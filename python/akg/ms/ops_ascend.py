# Copyright 2021 Huawei Technologies Co., Ltd
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


def RealDiv(x, y, target=utils.CCE):
    """RealDiv"""
    return math.RealDiv(x, y, target)


def FloorDiv(x, y, target=utils.CCE):
    """FloorDiv"""
    return math.FloorDiv(x, y, target)


def Argmax(x, axis=-1, target=utils.CCE):
    """Argmax"""
    return math.Argmax(x, axis, target=target)


def SimpleMean(x, target=utils.CCE):
    """SimpleMean"""
    return math.Mean(x, axis=[2, 3], keepdims=True, target=target)


def ReLU(x, target=utils.CCE):
    """ReLU"""
    return nn.Relu(x, target)


def ZerosLike(x, target=utils.CCE):
    """ZerosLike"""
    return nn.ZerosLike(x, target=target)


def StridedSlice(x, begin, end, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask,
                 shrink_axis_mask, target=utils.CCE):
    """StridedSlice"""
    return array.StridedSlice(x, begin, end, strides, begin_mask, end_mask, ellipsis_mask,
                              new_axis_mask, shrink_axis_mask, target=target)


def SparseSoftmaxCrossEntropyWithLogits(features, labels, is_grad=False, sens=1.0, target=utils.CCE):
    """sparse softmax cross entropy with logits"""
    if is_grad:
        return nn.SparseSoftmaxCrossEntropyWithLogitsAd(labels, features, reduction='mean', grad_scale=sens, target=target)
    return nn.SparseSoftmaxCrossEntropyWithLogits(labels, features, reduction='mean', target=target)


def Softmax(x, axis=-1, target=utils.CCE):
    """Softmax"""
    return nn.Softmax(x, axis, target)


def ReluGrad(y_backprop, x, target=utils.CCE):
    """gradient of relu"""
    return nn.ReluAd(y_backprop, x, target)


def ReduceMean(x, axis, keepdims, target=utils.CCE):
    """ReduceMean"""
    return math.Mean(x, axis=axis, keepdims=keepdims, target=target)


def ProdForceSeA(net_deriv_tensor, in_deriv_tensor, nlist_tensor, natoms=192, target=utils.CCE):
    """ProdForceSeA"""
    return math.ProdForceSeA(net_deriv_tensor, in_deriv_tensor, nlist_tensor, natoms, target)


def ProdForceSeAGrad(grad_tensor, in_deriv_tensor, nlist_tensor, natoms=192, target=utils.CCE):
    """ProdForceSeAGrad"""
    return math.ProdForceSeAGrad(grad_tensor, in_deriv_tensor, nlist_tensor, natoms, target)


def OneHot(indices, depth, on_value, off_value, axis=-1, target=utils.CCE):
    """OneHot"""
    return array.OneHotV2(indices, on_value, off_value, depth, axis=axis, target=target)


def SimpleMeanGrad(HEAD, input_shape, target=utils.CCE):
    """SimpleMeanGrad"""
    return nn.MeanAd(HEAD, input_shape, axis=[2, 3], keepdims=True, target=target)


def MaxPoolWithArgmax(x, pad_mode="valid", window=1, pad=0, stride=1, target=utils.CCE):
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


def MatMul(x1, x2, out_dtype, transpose_a=False, transpose_b=False, target=utils.CCE):
    """MatMul"""
    return math.MatMul(x=x1, y=x2, b=None, out_dtype=out_dtype, left_format="zN", right_format="zN", out_format="zN",
                       transpose_x=transpose_a, transpose_y=transpose_b, target=target)


def Conv2D(x, x_shape, w, w_shape, pad_list, stride=1, dilation=1, target=utils.CCE):
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


def LoadIm2Col(x, ksizes, strides, target=utils.CCE):
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


def Four2Five(x, data_format=None, dst_type="float16", target=utils.CCE):
    """from 4d(NCHW) to 5d(NC1HWC0)"""
    from akg.ms.utils import DEFAULT
    if data_format is None:
        data_format = [DEFAULT]
    if isinstance(data_format, list):
        data_format = data_format[0]
    if data_format == DEFAULT:
        data_format = "NCHW"
    return array.Four2Five(x, data_format, dst_type, target)


def Five2Four(x, shape4d, dstType, output_format, target=utils.CCE):
    """from 5d(NC1HWC0) to 4d(NCHW)"""
    return array.Five2Four(x, shape4d, dstType, output_format, target)


def ConvBN1(x, x_shape, w, w_shape, pad_list, stride=1, dilation=1, target=utils.CCE):
    """ConvBN1"""
    if len(pad_list) != 4:
        raise IndexError("Length of pad must be equal 4")
    pad = pad_list
    data = []
    data.append(x)
    data.append(w)
    fmap_shape = x_shape   # 4D
    filter_shape = w_shape  # 4D
    stride = [stride, stride]
    dilation = [dilation, dilation]
    return nn.ConvBn1(data, fmap_shape, filter_shape, pad, stride, dilation, use_bias=False, attrs=None, target=target)


def GatherV2(params, indices, axis=0, target=utils.CCE):
    """gather version2"""
    return array.GatherV2(params, indices, axis, target)


def LambApplyOptimizerAssign(grad, input_v, input_m, input_param, beta_1, one_minus_beta_1, beta_2, one_minus_beta_2, epsilon, steps, do_use_weight,
                             weight_decay_rate, target=utils.CCE):
    """LambApplyOptimizerAssign"""
    return optimizers.LambApplyOptimizerAssign(grad, input_v, input_m, input_param, beta_1, one_minus_beta_1, beta_2, one_minus_beta_2, epsilon,
                                               steps, do_use_weight, weight_decay_rate, target)


def FusedBN1(data, target=utils.CCE):
    """FusedBN1"""
    return nn.FusedBn1(data, target)


def FusedBN2(mean, var_part, running_mean, running_var, momentum=0.8, target=utils.CCE):
    """FusedBN2"""
    return nn.FusedBn2(mean, var_part, running_mean, running_var, momentum, target=target)


def FusedBN3(data, mean, variance, gamma, beta, eps=1e-3, target=utils.CCE):
    """FusedBN3"""
    return nn.FusedBn3(data, mean, variance, gamma, beta, eps, target=target)


def ClearZero(x, target=utils.CCE):
    """ClearZero"""
    return state.ClearZero(x, target)


def BiasAdd(x, b, data_format=None, target=utils.CCE):
    """BiasAdd"""
    if data_format is None:
        data_format = ["NCHW"]
    return nn.BiasAdd(x, b, data_format[0], target)


def BiasAddGrad(dout, data_format=None, target=utils.CCE):
    """grediant of bias_add"""
    if data_format is None:
        data_format = ["NCHW"]
    dout_shape = get_shape(dout)
    return nn.BiasAddAd(dout, dout_shape, data_format[0], target)


def BatchMatMul(x1, x2, transpose_a=False, transpose_b=False, target=utils.CCE):
    """use cube version matmul"""
    return math.MatMul(x=x1, y=x2, b=None, out_dtype=x1.dtype,
                       left_format="zN", right_format="zN", out_format="zN",
                       transpose_x=transpose_a, transpose_y=transpose_b, target=target)


def AssignAdd(ref, value, target=utils.CCE):
    """AssignAdd"""
    return state.AssignAdd(ref, value, target)


def ApplyMomentum(variable, accumulation, learning_rate, gradient, momentum, use_nesterov=False,
                  gradient_scale=1.0, target=utils.CCE):
    """ApplyMomentum"""
    return optimizers.ApplyMomentum(variable, gradient, accumulation, learning_rate, momentum,
                                    use_nesterov=use_nesterov, grad_scale=gradient_scale, target=target)


def EqualCount(x, y, target=utils.CCE):
    """EqualCount"""
    return math.EqualCount(x, y, target)


def BNGrad1(dy, data, mean, target=utils.CCE):
    """BNGrad1"""
    return nn.FusedBnGrad1(dy, data, mean, target)


def BNGrad2(dgamma_red_hw, dbeta_red_hw, variance, gamma, eps=1e-3, data_shape=None, target=utils.CCE):
    """BNGrad2"""
    return nn.FusedBnGrad2(dgamma_red_hw, dbeta_red_hw, variance, gamma, eps, data_shape, target)


def BNGrad3(dy, rs, dgamma_dx, dbeta_dx, data_minus_mean, target=utils.CCE):
    """BNGrad3"""
    return nn.FusedBnGrad3(dy, rs, dgamma_dx, dbeta_dx, data_minus_mean, target)


def FusedBatchNorm(x, scale, b, mean, variance, momentum=0.99, epsilon=1e-3, data_format=None, target=utils.CCE):
    """FusedBatchNorm"""
    if data_format is None:
        from akg.ms.utils import DEFAULT
        data_format = [DEFAULT]
    if isinstance(data_format, list):
        data_format = data_format[0]
    outputs = nn.FusedBatchNorm(x, scale, b, mean, variance, momentum=momentum, eps=epsilon,
                                is_training=True, data_format=data_format, axis=1, target=target)
    return outputs


def FusedBatchNormGrad(dy, x, scale, save_mean, save_inv_variance, data_format=None, target=utils.CCE):
    """gradient for fused batchnorm"""
    from akg.ms.utils import DEFAULT
    if data_format is None:
        data_format = [DEFAULT]
    if isinstance(data_format, list):
        data_format = data_format[0]
    eps = 1e-3
    return nn.FusedBatchNormGrad(dy, x, save_mean, save_inv_variance, scale, eps=eps, data_format=data_format,
                                 axis=1, target=target)


def FusedBatchNormInfer(x, scale, b, mean, variance, momentum=0.99, epsilon=1e-3, data_format=None,
                        target=utils.CCE):
    """inference mode of fuse batchnorm"""
    if data_format is None:
        from akg.ms.utils import DEFAULT
        data_format = [DEFAULT]
    if isinstance(data_format, list):
        data_format = data_format[0]
    return nn.FusedBatchNorm(x, scale, b, mean, variance, momentum=momentum, eps=epsilon,
                             is_training=False, data_format=data_format, axis=1, target=target)


def Conv2DBackpropInput(out_backprop, input_sizes, filter_0, filter_shape, pad_list, stride=1, dilation=1, target=utils.CCE):
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


def Conv2DBackpropFilter(out_backprop, input_0, input_shape, filter_sizes, pad_list, stride=1, dilation=1, target=utils.CCE):
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
