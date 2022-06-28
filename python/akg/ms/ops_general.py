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

"""general operators"""
import akg.utils as utils
import akg.ops.math as math
import akg.ops.array as array
from akg.ms.utils import reg_op


@reg_op("TensorAdd")
def tensor_add(x, y, target=utils.CUDA):
    """TensorAdd"""
    return math.add(x, y, scale=1.0, polyhedral=True, attrs={}, target=target)


@reg_op("Add")
def add(x, y, target=utils.CUDA):
    """Add"""
    return math.add(x, y, scale=1.0, polyhedral=True, attrs={}, target=target)


@reg_op("AddN")
def add_n(inputs, target=utils.CUDA):
    """AddN"""
    return math.addn(inputs, target)


@reg_op("Assign")
def assign(ref, val, target=utils.CUDA):
    """Assign"""
    return math.assign(ref, val, target)


@reg_op("Cast")
def cast(x, dst_type, target=utils.CUDA):
    """Cast"""
    return math.cast(x, dst_type, target)


@reg_op("Pow")
def pow(x, scale, target=utils.CUDA):
    """Pow"""
    return math.pow_(x, scale, target)


@reg_op("Round")
def round_(x, target=utils.CUDA):
    """Round"""
    return math.round_(x, target)


@reg_op("Rsqrt")
def rsqrt(x, target=utils.CUDA):
    """Rsqrt"""
    return math.rsqt(x, target)


@reg_op("Equal")
def equal(input1, input2, target=utils.CUDA):
    """Equal"""
    return math.Equal(input1, input2, target)


@reg_op("LessEqual")
def less_equal(input1, input2, target=utils.CUDA):
    """LessEqual"""
    return math.less_equal(input1, input2, target)


@reg_op("Mul")
def mul(x, y, target=utils.CUDA):
    """Mul"""
    return math.mul(x, y, target)


@reg_op("Sub")
def sub(x, y, target=utils.CUDA):
    """Sub"""
    return math.sub(x, y, target)


@reg_op("Div")
def div(x, y, target=utils.CUDA):
    """Div"""
    return math.divide(x, y, target)


@reg_op("Divide")
def divide(x, y, target=utils.CUDA):
    """Divide"""
    return math.divide(x, y, target)


@reg_op("Tile")
def tile(data, multiples, target=utils.CUDA):
    """Tile"""
    return array.tile(data, multiples, target)


@reg_op("LogicalOr")
def logical_or(x, y, target=utils.CUDA):
    """LogicalOr"""
    return math.logical_or(x, y, target)


@reg_op("LogicalAnd")
def logical_and(x, y, target=utils.CUDA):
    """LogicalAnd."""
    return math.logical_and(x, y, target)


@reg_op("LogicalNot")
def logical_not(data, target=utils.CUDA):
    """LogicalNot"""
    return math.logical_not(data, target)


@reg_op("NotEqual")
def not_equal(x, y, target=utils.CUDA):
    """NotEqual"""
    return math.not_equal(x, y, target)


@reg_op("GreaterEqual")
def greater_equal(x, y, target=utils.CUDA):
    """GreaterEqual"""
    return math.greater_equal(x, y, target)


@reg_op("Max")
def tensor_max(x, axis=None, keep_dims=False, target=utils.CUDA):
    """Max"""
    return math.reduce_max(x, axis=axis, keepdims=keep_dims, target=target)


@reg_op("Min")
def tensor_min(x, axis=None, keep_dims=False, target=utils.CUDA):
    """Min"""
    return math.reduce_min(x, axis=axis, keepdims=keep_dims, target=target)


@reg_op("ReduceAnd")
def tensor_and(x, axis=None, keep_dims=False, target=utils.CUDA):
    """ReduceAnd"""
    return math.reduce_and(x, axis=axis, keepdims=keep_dims, target=target)


@reg_op("ReduceOr")
def tensor_or(x, axis=None, keep_dims=False, target=utils.CUDA):
    """ReduceOr"""
    return math.reduce_or(x, axis=axis, keepdims=keep_dims, target=target)


@reg_op("ReduceSum")
def tensor_or(x, axis=None, keep_dims=False, target=utils.CUDA):
    """ReduceSum"""
    return math.reduce_sum(x, axis=axis, keepdims=keep_dims, target=target)


@reg_op("ReduceProd")
def tensor_or(x, axis=None, keep_dims=False, target=utils.CUDA):
    """ReduceProd"""
    return math.reduce_prod(x, axis=axis, keepdims=keep_dims, target=target)


@reg_op("Neg")
def neg(x, target=utils.CUDA):
    """Neg"""
    return math.neg(x, target)


@reg_op("Log")
def log(x, target=utils.CUDA):
    """Log"""
    return math.log(x, target)


@reg_op("Less")
def less(x, y, target=utils.CUDA):
    """Less"""
    return math.less(x, y, target)


@reg_op("Exp")
def exp(x, target=utils.CUDA):
    """Exp"""
    return math.exp(x, target)


@reg_op("Sum")
def tensor_sum(data, axis=None, keepdims=False, target=utils.CUDA):
    """Sum"""
    return math.sum(data, axis, keepdims, target=target)


@reg_op("Reshape")
def reshape(tensor, shape, target=utils.CUDA):
    """Reshape"""
    return array.reshape(tensor, shape, target)


@reg_op("Reciprocal")
def reciprocal(x, target=utils.CUDA):
    """Reciprocal"""
    return math.reciprocal(x, target)


@reg_op("Select")
def select(condition, x1, x2, target=utils.CUDA):
    """Select"""
    return math.select(condition, x1, x2, target)


@reg_op("Transpose")
def transpose(tensor, axes, target=utils.CUDA):
    """Transpose"""
    return array.transpose(tensor, axes, target)


@reg_op("UnsortedSegmentMax")
def unsorted_segment_max(data, segment_ids, num_segments, target=utils.CCE):
    """UnsortedSegmentMax"""
    return array.unsorted_segment_max(data, segment_ids, num_segments, target)


@reg_op("UnsortedSegmentSum")
def unsorted_segment_sum(data, indices, num, op_id=0, target=utils.CUDA):
    """UnsortedSegmentSum"""
    return array.unsorted_segment_sum(data, indices, num, op_id, target)


@reg_op("Sqrt")
def sqrt(x, target=utils.CUDA):
    """Sqrt"""
    return math.sqrt(x, target)
