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

"""resize_nearest"""
import akg
import akg.tvm
from akg.tvm.hybrid import script
from akg.utils import validation_check as vc_util
from akg.utils.format_transform import get_shape

def downsampling(inputs, output_shape):
    """downsampling"""
    scale_h, scale_w = [int(akg.tvm.truncdiv(inputs.shape[i], output_shape[i])) for i in range(1, 3)]
    res = akg.tvm.compute(output_shape, lambda b, h, w, c: inputs[b, h * scale_h, w * scale_w, c], name="downsampling")
    return res


def process_integer_scale(inputs, output_shape):
    """high performance version for integer scale"""
    inputs_shape = [x.value for x in inputs.shape]
    inputs_h, inputs_w = inputs_shape[1:3]
    output_h, output_w = output_shape[1:3]
    if inputs_h >= output_h and inputs_w >= output_w:
        if inputs_h % output_h != 0 or inputs_w % output_w != 0:
            return None
        return downsampling(inputs, output_shape)
    elif inputs_h <= output_h and inputs_w <= output_w:
        if output_h % inputs_h != 0 or output_w % inputs_w != 0:
            return None
        from .upsampling import upsampling
        return upsampling(inputs, output_shape)
    else:
        return None


def process_non_integer_scale(inputs, out_shape):
    """non integer scale"""
    in_shape = [x.value for x in inputs.shape]
    batch, height, width, channel = inputs.shape

    scale_h = akg.tvm.const(1.0 * in_shape[1] / out_shape[1], "float16")
    scale_w = akg.tvm.const(1.0 * in_shape[2] / out_shape[2], "float16")
    index_h_fp = akg.tvm.compute([out_shape[1]], lambda i: i * scale_h, name="index_h_fp")
    index_w_fp = akg.tvm.compute([out_shape[2]], lambda i: i * scale_w, name="index_w_fp")
    index_h = akg.lang.cce.floor(index_h_fp)
    index_w = akg.lang.cce.floor(index_w_fp)

    @script
    def resize(inputs, index_h, index_w, newH, newW):
        out = output_tensor((batch, newH, newW, channel), inputs.dtype)
        for i in range(batch):
            for j in range(height):
                for k in range(width):
                    for l in range(channel):
                        for m in range(newH):
                            for n in range(newW):
                                if index_h[m] == j:
                                    if index_w[n] == k:
                                        out[i, m, n, l] = inputs[i, j, k, l]
        return out

    newH = akg.tvm.const(out_shape[1], "int32")
    newW = akg.tvm.const(out_shape[2], "int32")
    res = resize(inputs, index_h, index_w, newH, newW)
    return res

@vc_util.check_input_type(akg.tvm.tensor.Tensor, (list, tuple))
def resize_nearest(input, output_shape):
    """
    Resize images using Nearest-neighbor interpolation.
    
    Args:
        input (tvm.tensor.Tensor): 4-D tensor of type float16 or float32 `("NHWC")`.
        output_shape (Union[tuple, list]): New size of image 4 integers `("NHWC")`.
    
    Note:
        The batch_num("N") of input and output must be equal, channel_num("C") is also.
    
    Returns:
        tvm.tensor.Tensor, has the same type as `input`.
    """
    input_shape = get_shape(input)
    vc_util.check_shape(input, 4, "input")
    vc_util.check_shape(output_shape, 4, "output_shape")
    vc_util.ops_dtype_check(input.dtype, vc_util.DtypeForDavinci.ALL_FLOAT)
    vc_util.check_equal("input batchsize", "output batchsize", input_shape[0], output_shape[0])
    vc_util.check_equal("input channel num", "output channel num", input_shape[3], output_shape[3])

    res = process_integer_scale(input, output_shape)
    if res == None:
        res = process_non_integer_scale(input, output_shape)
    return res
