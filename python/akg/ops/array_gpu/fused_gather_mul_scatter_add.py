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

"""operator dsl function: gather_mul_scatter_add"""
import akg.tvm
from akg.tvm.hybrid import script
from akg.utils import validation_check as vc_util
from akg.utils.format_transform import get_shape
from akg.utils.dsl_create import get_broadcast_shape


@vc_util.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, int)
def fused_gather_mul_scatter_add(input1, input2, input3, input4, axis=0):
    if axis < 0:
        axis += len(input1.shape)
    assert axis >= 0
    assert axis < len(input1.shape)

    axis_var = akg.tvm.const(0, dtype="int32")
    if len(input1.shape) == 3:
        gather_out_shape = get_shape(input1)[:axis] + [get_shape(input2)[0],] + get_shape(input1)[axis + 1:]
        broadcast_shape = get_broadcast_shape(gather_out_shape, get_shape(input3))
        dim2_size = broadcast_shape[2]
        dtype = input3.dtype

        @script(capture=locals())
        def _gather_out(input1_, input2_):
            # gather
            gather_out_ = output_tensor(broadcast_shape, input1_.dtype)
            for i in range(broadcast_shape[0]):
                for j in range(broadcast_shape[1]):
                    for k in range(broadcast_shape[2]):
                        if axis == 0:
                            gather_out_[i, j, k] = input1_[input2_[i], j, k]
                        elif axis == 1:
                            gather_out_[i, j, k] = input1_[i, input2_[j], k]
                        else:
                            gather_out_[i, j, k] = input1_[i, j, input2_[k]]
            return gather_out_

        gather_out = _gather_out(input1, input2)

        @script(capture=locals())
        def _mul_out(gather_out_, input3_):
            # mul
            mul_out_ = output_tensor(broadcast_shape, input3_.dtype)

            for i in range(input3_.shape[0]):
                i1 = i if gather_out_.shape[0] == broadcast_shape[0] else 0
                i2 = i if input3_.shape[0] == broadcast_shape[0] else 0
                for j in range(input3_.shape[1]):
                    j1 = j if gather_out_.shape[1] == broadcast_shape[1] else 0
                    j2 = j if input3_.shape[1] == broadcast_shape[1] else 0
                    for k in range(dim2_size):
                        k1 = k if gather_out_.shape[2] == broadcast_shape[2] else 0
                        k2 = k if input3_.shape[2] == broadcast_shape[2] else 0
                        mul_out_[i, j, k] = gather_out_[i1, j1, k1] * input3_[i2, j2, k2]
            return mul_out_

        mul_out = _mul_out(gather_out, input3)

        @script(capture=locals())
        def _scatter_add(input1_, mul_out_, input4_):
            # scatter_add
            scatter_add_ = output_tensor(input1_.shape, input1_.dtype)
            for i in range(broadcast_shape[0]):
                for j in range(broadcast_shape[1]):
                    for k in range(broadcast_shape[2]):
                        scatter_add_[input4_[i, 0], j, k] += mul_out_[i, j, k]

            return scatter_add_

        return _scatter_add(input1, mul_out, input4)

    raise ValueError("scatter_add only support for 3 dimensions")
