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

"""resize_bilinear"""
import akg
import akg.tvm
from akg.tvm.hybrid import script
from akg.utils import validation_check as vc_util
from akg.utils.format_transform import get_shape

def calculate_params(input_h, input_w, output_h, output_w, dtype):
    """calculate index parameters for bilinear interpolation"""
    # scale value is required to map from input space to output space
    height_scale = (input_h - 1.0) / (output_h - 1.0)
    width_scale = (input_w - 1.0) / (output_w - 1.0)
    height_scale = akg.tvm.const(height_scale, dtype=dtype)
    width_scale = akg.tvm.const(width_scale, dtype=dtype)

    # ys_lower, ys_upper and ys_lerp contain bottom index,
    # top index and interpulation factor for each row position of output matrix respectively
    float_y = akg.tvm.compute([output_h], lambda i: height_scale * i, name="float_y")
    ys_lower = akg.lang.cce.floor(float_y)
    ys_upper = akg.lang.cce.ceil(float_y)
    ys_upper_lerp = akg.tvm.compute([output_h], lambda i: float_y[i] - ys_lower[i], name="ys_upper_lerp")
    ys_lower_temp = akg.lang.cce.vmuls(ys_upper_lerp, akg.tvm.const(-1.0, dtype))
    ys_lower_lerp = akg.lang.cce.vadds(ys_lower_temp, akg.tvm.const(1.0, dtype))

    # xs_lower,xs_upper and xs_lerp contain left index,
    # right index and interpulation factor for each column position of output matrix respectively
    float_x = akg.tvm.compute([output_w], lambda i: width_scale * i, name="float_x")
    xs_lower = akg.lang.cce.floor(float_x)
    xs_upper = akg.lang.cce.ceil(float_x)
    xs_upper_lerp = akg.tvm.compute([output_w], lambda i: float_x[i] - xs_lower[i], name="xs_upper_lerp")
    xs_lower_temp = akg.lang.cce.vmuls(xs_upper_lerp, akg.tvm.const(-1.0, dtype))
    xs_lower_lerp = akg.lang.cce.vadds(xs_lower_temp, akg.tvm.const(1.0, dtype))

    return xs_lower, xs_lower_lerp, xs_upper, xs_upper_lerp, ys_lower, ys_lower_lerp, ys_upper, ys_upper_lerp


@script
def resizeH1(inputs, newH, ys_index, lerp):
    """resize H1"""
    batchSize, oldH, width, channel = inputs.shape
    out = output_tensor((batchSize, newH, width, channel), inputs.dtype)
    for b in range(batchSize):
        for h in range(newH):
            for w in range(width):
                for c in range(channel):
                    for i in range(oldH):
                        if ys_index[h] == i:
                            out[b, h, w, c] = inputs[b, i, w, c] * lerp[h]
    return out


@script
def resizeH2(inputs, newH, ys_index, lerp):
    """resize H2"""
    batchSize, oldH, width, channel = inputs.shape
    out = output_tensor((batchSize, newH, width, channel), inputs.dtype)
    for b in range(batchSize):
        for h in range(newH):
            for w in range(width):
                for c in range(channel):
                    for i in range(oldH):
                        if ys_index[h] == i:
                            out[b, h, w, c] = inputs[b, i, w, c] * lerp[h]
    return out


@script
def resizeW1(inputs, newW, xs_index, lerp):
    """resize W1"""
    batchSize, height, oldW, channel = inputs.shape
    out = output_tensor((batchSize, height, newW, channel), inputs.dtype)
    for b in range(batchSize):
        for h in range(height):
            for w in range(newW):
                for c in range(channel):
                    for i in range(oldW):
                        if xs_index[w] == i:
                            out[b, h, w, c] = inputs[b, h, i, c] * lerp[w]
    return out


@script
def resizeW2(inputs, newW, xs_index, lerp):
    """resize W2"""
    batchSize, height, oldW, channel = inputs.shape
    out = output_tensor((batchSize, height, newW, channel), inputs.dtype)
    for b in range(batchSize):
        for h in range(height):
            for w in range(newW):
                for c in range(channel):
                    for i in range(oldW):
                        if xs_index[w] == i:
                            out[b, h, w, c] = inputs[b, h, i, c] * lerp[w]
    return out

@vc_util.check_input_type(akg.tvm.tensor.Tensor, (list, tuple))
def resize_bilinear(input, output_shape):
    """
    Resize images using bilinear interpolation.
    
    Args:
        input (tvm.tensor.Tensor): 4-D tensor of type float16 or float32 `("NHWC")`.
        output_shape (Union[tuple, list]): New size of image, two integer `H` and `W`.
    
    Returns:
        tvm.tensor.Tensor, shape `(input.shape[0], output_shape[0], output_shape[1], input.shape[3])`,
            has of the same type as `input`.
    """
    vc_util.check_shape(input, 4, "input")
    vc_util.check_shape(output_shape, 2, "output_shape")
    vc_util.ops_dtype_check(input.dtype, vc_util.DtypeForDavinci.ALL_FLOAT)
    inputs_shape = get_shape(input)
    dtype = input.dtype
    if inputs_shape[1:3] == list(output_shape):
        res = akg.tvm.compute(inputs_shape, lambda *i: input(*i), name="assign")
        return res
    # Get N,H,W,C from input and output shape
    input_h, input_w = inputs_shape[1:3]
    output_h, output_w = output_shape

    xs_lower, xs_lower_lerp, xs_upper, xs_upper_lerp, ys_lower, ys_lower_lerp, ys_upper, ys_upper_lerp = \
        calculate_params(input_h, input_w, output_h, output_w, dtype)
    newH = akg.tvm.const(output_h, "int32")
    newW = akg.tvm.const(output_w, "int32")
    resH1 = resizeH1(input, newH, ys_lower, ys_lower_lerp)
    resH2 = resizeH2(input, newH, ys_upper, ys_upper_lerp)
    resH = akg.lang.cce.vadd(resH1, resH2)
    resW1 = resizeW1(resH, newW, xs_lower, xs_lower_lerp)
    resW2 = resizeW2(resH, newW, xs_upper, xs_upper_lerp)
    res = akg.lang.cce.vadd(resW1, resW2)
    return res
