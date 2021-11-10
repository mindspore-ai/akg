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

"""operator dsl function: upsampling"""
import akg.tvm
from akg.tvm.hybrid import script


@script
def compute_upsampling(inputs, scale_h, scale_w):
    """
    repeat the rows and columns of the data
    by `scale_h` and `scale_w` respectively.
    """
    batch_size, input_h, input_w, channel = inputs.shape
    output_h = input_h * scale_h
    output_w = input_w * scale_w
    output = output_tensor((batch_size, output_h, output_w, channel), inputs.dtype)
    for b in range(batch_size):
        for h0 in range(input_h):
            for h1 in range(scale_h):
                for w0 in range(input_w):
                    for w1 in range(scale_w):
                        for c in range(channel):
                            output[b, h0 * scale_h + h1, w0 * scale_w + w1, c] = inputs[b, h0, w0, c]
    return output


def upsampling(inputs, output_shape, target="cce"):
    """
    Upsampling for 4D inputs.

    Repeats the rows and columns of the data by height and width respectively.

    Args:
        inputs(akg.tvm.Tensor): 4D tensor.
        output_shape(list, tuple): Specifies the shape of output tensor, should be a 4D shape.

    Returns:
        akg.tvm.Tensor, has the same type as inputs and is shaped by output_shape.
    """
    inputs_shape = [x.value for x in inputs.shape]
    if len(inputs_shape) != 4:
        raise RuntimeError("Input data only support 4-dim(NHWC) shape.")
    if len(output_shape) != 4:
        raise RuntimeError("Output data only support 4-dim(NHWC) shape.")
    if inputs_shape[0] != output_shape[0]:
        raise ValueError("batch size of input and output must be equal")
    if inputs_shape[3] != output_shape[3]:
        raise ValueError("channel size of input and output must be equal")

    for i in range(1, 3):
        if output_shape[i] < inputs_shape[i]:
            raise ValueError("The length in output_shape is less than input_shape.")
        if output_shape[i] % inputs_shape[i] != 0:
            raise ValueError("The upsampling scale is not interger.")

    scale = [int(output_shape[i] / inputs_shape[i]) for i in range(1, 3)]
    scale = [akg.tvm.convert(s) for s in scale]
    res = compute_upsampling(inputs, *tuple(scale))
    return res
