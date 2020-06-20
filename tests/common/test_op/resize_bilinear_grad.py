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

"""resize_bilinear_grad"""
import akg
import akg.tvm
import akg.topi
from akg.tvm.hybrid import script
from akg.utils import validation_check as vc_util
from akg.utils.format_transform import get_shape
from .resize_bilinear import calculate_params


@script
def gradW1(inputs, outW, xs_index, lerp):
    """Grad W1"""
    batchSize, height, inW, channel = inputs.shape
    out = output_tensor((batchSize, height, outW, inW, channel), inputs.dtype)
    for b in range(batchSize):
        for h in range(height):
            for w in range(outW):
                for i in range(inW):
                    if xs_index[i] == w:
                        for c in range(channel):
                            out[b, h, w, i, c] = inputs[b, h, i, c] * lerp[i]
    return out


@script
def gradW2(inputs, outW, xs_index, lerp):
    """Grad W2"""
    batchSize, height, inW, channel = inputs.shape
    out = output_tensor((batchSize, height, outW, inW, channel), inputs.dtype)
    for b in range(batchSize):
        for h in range(height):
            for w in range(outW):
                for i in range(inW):
                    if xs_index[i] == w:
                        for c in range(channel):
                            out[b, h, w, i, c] = inputs[b, h, i, c] * lerp[i]
    return out


@script
def gradH1(inputs, outH, ys_index, lerp):
    """Grad H1"""
    batchSize, inH, width, channel = inputs.shape
    out = output_tensor((batchSize, outH, inH, width, channel), inputs.dtype)
    for b in range(batchSize):
        for h in range(outH):
            for i in range(inH):
                if ys_index[i] == h:
                    for w in range(width):
                        for c in range(channel):
                            out[b, h, i, w, c] = inputs[b, i, w, c] * lerp[i]
    return out


@script
def gradH2(inputs, outH, ys_index, lerp):
    """Grad H2"""
    batchSize, inH, width, channel = inputs.shape
    out = output_tensor((batchSize, outH, inH, width, channel), inputs.dtype)
    for b in range(batchSize):
        for h in range(outH):
            for i in range(inH):
                if ys_index[i] == h:
                    for w in range(width):
                        for c in range(channel):
                            out[b, h, i, w, c] = inputs[b, i, w, c] * lerp[i]
    return out

@vc_util.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor)
def resize_bilinear_grad(resized_data, original_data):
    """
    Gradient for resize_bilinear.
    
    Args:
        resized_data (tvm.tensor.Tensor): 4-D tensor of type float16 or float32 `("NHWC")`.
        original_data (tvm.tensor.Tensor): 4-D tensor, the image before resize.
    
    Note:
        The batch_num("N") of `resized_data` and `original_data` must be equal, channel_num("C") is also.
    
    Returns:
        tvm.tensor.Tensor, has the same type and shape as `original_data`.
    """
    resized_shape = get_shape(resized_data)
    original_shape = get_shape(original_data)
    vc_util.check_shape(resized_shape, 4, "resized_data")
    vc_util.check_shape(original_shape, 4, "original_data")
    vc_util.check_equal("input batchsize", "output batchsize", resized_shape[0], original_shape[0])
    vc_util.check_equal("input channel num", "output channel num", resized_shape[3], original_shape[3])
    vc_util.ops_dtype_check([resized_data.dtype, original_data.dtype], vc_util.DtypeForDavinci.ALL_FLOAT)

    # Get N,H,W,C from input and output shape
    resized_height, resized_width = resized_shape[1:3]
    original_height, original_width = original_shape[1:3]

    xs_lower, xs_lower_lerp, xs_upper, xs_upper_lerp, ys_lower, ys_lower_lerp, ys_upper, ys_upper_lerp = \
        calculate_params(original_height, original_width, resized_height, resized_width, resized_data.dtype)

    outH = akg.tvm.const(original_height, "int32")
    outW = akg.tvm.const(original_width, "int32")

    resW1 = gradW1(resized_data, outW, xs_lower, xs_lower_lerp)
    resW2 = gradW2(resized_data, outW, xs_upper, xs_upper_lerp)
    resW = akg.lang.cce.vadd(resW1, resW2)
    resW_sum = akg.topi.sum(resW, axis=3, keepdims=False)
    resH1 = gradH1(resW_sum, outH, ys_lower, ys_lower_lerp)
    resH2 = gradH2(resW_sum, outH, ys_upper, ys_upper_lerp)
    resH = akg.lang.cce.vadd(resH1, resH2)
    res = akg.topi.sum(resH, axis=2, keepdims=False)
    return res
