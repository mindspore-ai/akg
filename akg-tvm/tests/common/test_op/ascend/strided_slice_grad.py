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

"""operator dsl function: strided_slice_grad"""

import akg.topi
import akg.tvm
from akg.tvm.hybrid import script
from akg.utils import custom_tiling as ct_util
from akg.ops.array.ascend.strided_slice import complete_args


def strided_slice_grad_tiling_strategy(tensor, begin, end, strides):
    """Custom tiling strategy for strided slice grad op."""
    strategy = list()
    for i, shp in enumerate(tensor.shape):
        length = end[i] - begin[i]
        if length <= strides[i] or int(shp) % strides[i] != 0:
            strategy += ct_util.create_constraint_on_tensor(tensor=tensor,
                                                            values="FULL",
                                                            tensor_pos=i,
                                                            constraints=ct_util.TileConstraint.MAX)
    return strategy


# stride=-1 not supported
def strided_slice_grad(grad, input_shape, begin, end, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask, target="cce"):
    # step0~4: complete begin, end, strides
    begin, end, strides, _, _ = complete_args(input_shape, begin, end, strides,
                                              begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask)
    @script
    def compute_dim1(grad, zero):
        dx = output_tensor(input_shape, grad.dtype)
        b0 = begin[0]
        s0 = strides[0]

        for i0 in range(0, input_shape[0]):
            dx[i0] = zero

        for g0 in range(0, grad.shape[0]):
            dx[g0 * s0 + b0] = grad[g0]
        return dx

    @script
    def compute_dim2(grad, zero):
        dx = output_tensor(input_shape, grad.dtype)
        b0, b1 = begin[0], begin[1]
        s0, s1 = strides[0], strides[1]

        for i0 in range(0, input_shape[0]):
            for i1 in range(0, input_shape[1]):
                dx[i0, i1] = zero

        for g0 in range(0, grad.shape[0]):
            for g1 in range(0, grad.shape[1]):
                dx[g0 * s0 + b0, g1 * s1 + b1] = grad[g0, g1]
        return dx

    @script
    def compute_dim3(grad, zero):
        dx = output_tensor(input_shape, grad.dtype)

        b0, b1, b2 = begin[0], begin[1], begin[2]
        s0, s1, s2 = strides[0], strides[1], strides[2]

        for i0 in range(0, input_shape[0]):
            for i1 in range(0, input_shape[1]):
                for i2 in range(0, input_shape[2]):
                    dx[i0, i1, i2] = zero

        for g0 in range(0, grad.shape[0]):
            for g1 in range(0, grad.shape[1]):
                for g2 in range(0, grad.shape[2]):
                    dx[g0 * s0 + b0, g1 * s1 + b1, g2 * s2 + b2] = grad[g0, g1, g2]
        return dx

    @script
    def compute_dim4(grad, zero):
        dx = output_tensor(input_shape, grad.dtype)

        b0, b1, b2, b3 = begin[0], begin[1], begin[2], begin[3]
        s0, s1, s2, s3 = strides[0], strides[1], strides[2], strides[3]

        for i0 in range(0, input_shape[0]):
            for i1 in range(0, input_shape[1]):
                for i2 in range(0, input_shape[2]):
                    for i3 in range(0, input_shape[3]):
                        dx[i0, i1, i2, i3] = zero

        for g0 in range(0, grad.shape[0]):
            for g1 in range(0, grad.shape[1]):
                for g2 in range(0, grad.shape[2]):
                    for g3 in range(0, grad.shape[3]):
                        dx[g0 * s0 + b0, g1 * s1 + b1, g2 * s2 + b2, g3 * s3 + b3] = grad[g0, g1, g2, g3]
        return dx

    @script
    def compute_dim5(grad, zero):
        dx = output_tensor(input_shape, grad.dtype)

        b0, b1, b2, b3, b4 = begin[0], begin[1], begin[2], begin[3], begin[4]
        s0, s1, s2, s3, s4 = strides[0], strides[1], strides[2], strides[3], strides[4]

        for i0 in range(0, input_shape[0]):
            for i1 in range(0, input_shape[1]):
                for i2 in range(0, input_shape[2]):
                    for i3 in range(0, input_shape[3]):
                        for i4 in range(0, input_shape[4]):
                            dx[i0, i1, i2, i3, i4] = zero

        for g0 in range(0, grad.shape[0]):
            for g1 in range(0, grad.shape[1]):
                for g2 in range(0, grad.shape[2]):
                    for g3 in range(0, grad.shape[3]):
                        for g4 in range(0, grad.shape[4]):
                            dx[g0 * s0 + b0, g1 * s1 + b1, g2 * s2 + b2, g3 * s3 + b3, g4 * s4 + b4] = grad[g0, g1, g2, g3, g4]
        return dx

    @script
    def compute_dim6(grad, zero):
        dx = output_tensor(input_shape, grad.dtype)

        b0, b1, b2, b3, b4, b5 = begin[0], begin[1], begin[2], begin[3], begin[4], begin[5]
        s0, s1, s2, s3, s4, s5 = strides[0], strides[1], strides[2], strides[3], strides[4], strides[5]

        for i0 in range(0, input_shape[0]):
            for i1 in range(0, input_shape[1]):
                for i2 in range(0, input_shape[2]):
                    for i3 in range(0, input_shape[3]):
                        for i4 in range(0, input_shape[4]):
                            for i5 in range(0, input_shape[5]):
                                dx[i0, i1, i2, i3, i4, i5] = zero

        for g0 in range(0, grad.shape[0]):
            for g1 in range(0, grad.shape[1]):
                for g2 in range(0, grad.shape[2]):
                    for g3 in range(0, grad.shape[3]):
                        for g4 in range(0, grad.shape[4]):
                            for g5 in range(0, grad.shape[5]):
                                dx[g0 * s0 + b0, g1 * s1 + b1, g2 * s2 + b2, g3 * s3 + b3, g4 * s4 + b4, g5 * s5 + b5] = grad[g0, g1, g2, g3, g4, g5]
        return dx

    @script
    def compute_dim7(grad, zero):
        dx = output_tensor(input_shape, grad.dtype)

        b0, b1, b2, b3, b4, b5, b6 = begin[0], begin[1], begin[2], begin[3], begin[4], begin[5], begin[6]
        s0, s1, s2, s3, s4, s5, s6 = strides[0], strides[1], strides[2], strides[3], strides[4], strides[5], strides[6]

        for i0 in range(0, input_shape[0]):
            for i1 in range(0, input_shape[1]):
                for i2 in range(0, input_shape[2]):
                    for i3 in range(0, input_shape[3]):
                        for i4 in range(0, input_shape[4]):
                            for i5 in range(0, input_shape[5]):
                                for i6 in range(0, input_shape[6]):
                                    dx[i0, i1, i2, i3, i4, i5, i6] = zero

        for g0 in range(0, grad.shape[0]):
            for g1 in range(0, grad.shape[1]):
                for g2 in range(0, grad.shape[2]):
                    for g3 in range(0, grad.shape[3]):
                        for g4 in range(0, grad.shape[4]):
                            for g5 in range(0, grad.shape[5]):
                                for g6 in range(0, grad.shape[6]):
                                    dx[g0 * s0 + b0, g1 * s1 + b1, g2 * s2 + b2, g3 * s3 + b3, g4 * s4 + b4, g5 * s5 + b5, g6 * s6 + b6] = grad[g0, g1, g2, g3, g4, g5, g6]
        return dx

    @script
    def compute_dim8(grad, zero):
        dx = output_tensor(input_shape, grad.dtype)

        b0, b1, b2, b3, b4, b5, b6, b7 = begin[0], begin[1], begin[2], begin[3], begin[4], begin[5], begin[6], begin[7]
        s0, s1, s2, s3, s4, s5, s6, s7 = strides[0], strides[1], strides[2], strides[3], strides[4], strides[5], strides[6], strides[7]

        for i0 in range(0, input_shape[0]):
            for i1 in range(0, input_shape[1]):
                for i2 in range(0, input_shape[2]):
                    for i3 in range(0, input_shape[3]):
                        for i4 in range(0, input_shape[4]):
                            for i5 in range(0, input_shape[5]):
                                for i6 in range(0, input_shape[6]):
                                    for i7 in range(0, input_shape[7]):
                                        dx[i0, i1, i2, i3, i4, i5, i6, i7] = zero

        for g0 in range(0, grad.shape[0]):
            for g1 in range(0, grad.shape[1]):
                for g2 in range(0, grad.shape[2]):
                    for g3 in range(0, grad.shape[3]):
                        for g4 in range(0, grad.shape[4]):
                            for g5 in range(0, grad.shape[5]):
                                for g6 in range(0, grad.shape[6]):
                                    for g7 in range(0, grad.shape[7]):
                                        dx[g0 * s0 + b0, g1 * s1 + b1, g2 * s2 + b2, g3 * s3 + b3, g4 * s4 + b4, g5 * s5 + b5, g6 * s6 + b6, g7 * s7 + b7] = grad[g0, g1, g2, g3, g4, g5, g6, g7]

        return dx

    # step5: reshape grad

    grad1_shape = [len(range(begin[i], end[i], strides[i])) for i in range(len(begin))]
    grad1 = akg.topi.reshape(grad, tuple(grad1_shape))

    # step6: use hybrid to implement strided_slice_grad's compute
    func_list = [None, compute_dim1, compute_dim2, compute_dim3, compute_dim4, compute_dim5, compute_dim6, compute_dim7, compute_dim8]
    dtype = grad.dtype
    zero = akg.tvm.const(0, dtype)
    dim = len(input_shape)
    if dim < 1 or dim > 8:
        raise AssertionError("only support 1~8 dim")
    out = func_list[dim](grad1, zero)
    attrs = {"custom_tiling": strided_slice_grad_tiling_strategy(out, begin, end, strides)}
    # step7: build
    return out, attrs
