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

"""operator dsl function: fused_gather_gather_add_mul_max_exp_scatter_add"""
import akg.tvm as tvm
from akg.tvm.hybrid import script
from akg.utils import validation_check as vc_util
from akg.utils.format_transform import get_shape
from akg.utils.dsl_create import get_broadcast_shape
from .tensor_scatter_add import tensor_scatter_add
from akg.ops.math_gpu.add import add
from akg.ops.math_gpu.mul import mul
from akg.ops.math_gpu.maximum import maximum
from akg.ops.math_gpu.exp import exp

@vc_util.check_input_type(tvm.tensor.Tensor, tvm.tensor.Tensor, int, str)
def gather(data, indices, axis, flag):
    """Only support axis=0."""
    ndim = len(data.shape)
    axis = axis + ndim if axis < 0 else axis
    assert axis >= 0
    assert axis < ndim

    data_shape = list(data.shape)
    indices_shape = list(indices.shape)
    output_shape = data_shape[:axis] + indices_shape + data_shape[axis+1:]
    left_shape = output_shape[:1]
    right_shape = output_shape[1:]

    def gen_ir(data, indices, out):
        ib = tvm.ir_builder.create()
        with ib.for_range_n(left_shape, 'i') as i:
            with ib.for_range_n(right_shape, 'j') as j:
                read_idx = [ib.load(indices, i)]
                val = ib.load(data, read_idx + j)
                ib.store(out, i + j, val)
        return ib.get()

    out_buf = tvm.decl_buffer(output_shape, data.dtype, "out_buf")

    return tvm.extern(
        [output_shape],
        [data, indices],
        lambda ins, outs: gen_ir(ins[0], ins[1], outs[0]),
        dtype=data.dtype,
        out_buffers=[out_buf],
        name="fused_gather" + flag,
    )

@vc_util.check_input_type(tvm.tensor.Tensor, tvm.tensor.Tensor, tvm.tensor.Tensor)
def scatter_add(data, indices, updates):
    """
    Args:
        data: [x, y, z]
        indices: [n]
        updates: [n, y, z]
    Output:
        [x, y, z]
    """
    left_shape = list(updates.shape[:1])
    right_shape = list(updates.shape[1:])

    def gen_ir(data, indices, updates, out):
        ib = tvm.ir_builder.create()
        with ib.for_range_n(left_shape, "i") as i:
            with ib.for_range_n(right_shape, "j") as j:
                idx_updates = i + j
                idx_data = [ib.load(indices, i)] + j
                temp = ib.load(updates, idx_updates) + ib.load(out, idx_data)
                ib.store(out, idx_data, temp)
        return ib.get()

    out_buf = tvm.decl_buffer(data.shape, data.dtype, "out_buf")
    return tvm.extern(
        [data.shape],
        [data, indices, updates],
        lambda ins, outs: gen_ir(ins[0], ins[1], ins[2], outs[0]),
        dtype=data.dtype,
        out_buffers=[out_buf],
        name="fused_scatter_add",
    )

@vc_util.check_input_type(tvm.tensor.Tensor, tvm.tensor.Tensor, tvm.tensor.Tensor,
                          tvm.tensor.Tensor, int)
def fused_gather_gather_add_mul_max_exp_scatter_add(inp1, inp2, inp3, inp4, axis):
    ndim = len(inp1.shape)
    axis = axis + ndim if axis < 0 else axis
    assert axis >= 0
    assert axis < ndim

    gather_out1 = gather(inp1, inp2, axis, "1")
    gather_out2 = gather(inp1, inp2, axis, "2")

    add_out = add(gather_out1, gather_out2)
    mul_out = mul(add_out, inp3)
    max_out = maximum(add_out, mul_out)
    exp_out = exp(max_out)
    scatter_out = scatter_add(inp1, inp4, exp_out)

    return exp_out, scatter_out
