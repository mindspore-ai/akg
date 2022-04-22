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

"""general operators"""
import akg.utils as utils
import akg.ops.math as math
import akg.ops.array as array
from akg.ms.utils import reg_op


@reg_op("TensorAdd")
def tensor_add(x, y, target=utils.CUDA):
    """TensorAdd"""
    return math.Add(x, y, scale=1.0, polyhedral=True, attrs={}, target=target)


@reg_op("Add")
def add(x, y, target=utils.CUDA):
    """Add"""
    return math.Add(x, y, scale=1.0, polyhedral=True, attrs={}, target=target)


@reg_op("AddN")
def add_n(inputs, target=utils.CUDA):
    """AddN"""
    return math.Addn(inputs, target)


@reg_op("Assign")
def assign(ref, val, target=utils.CUDA):
    """Assign"""
    return math.Assign(ref, val, target)


@reg_op("Cast")
def cast(x, dst_type, target=utils.CUDA):
    """Cast"""
    return math.Cast(x, dst_type, target)


@reg_op("Equal")
def equal(input1, input2, target=utils.CUDA):
    """Equal"""
    return math.Equal(input1, input2, target)


@reg_op("LessEqual")
def less_equal(input1, input2, target=utils.CUDA):
    """LessEqual"""
    return math.LessEqual(input1, input2, target)


@reg_op("Mul")
def mul(x, y, target=utils.CUDA):
    """Mul"""
    return math.Mul(x, y, target)


@reg_op("Sub")
def sub(x, y, target=utils.CUDA):
    """Sub"""
    return math.Sub(x, y, target)


@reg_op("Div")
def div(x, y, target=utils.CUDA):
    """Div"""
    return math.Divide(x, y, target)


@reg_op("Divide")
def divide(x, y, target=utils.CUDA):
    """Divide"""
    return math.Divide(x, y, target)


@reg_op("Tile")
def tile(data, multiples, target=utils.CUDA):
    """Tile"""
    return array.Tile(data, multiples, target)


@reg_op("LogicalOr")
def logical_or(x, y, target=utils.CUDA):
    """LogicalOr"""
    return math.LogicalOr(x, y, target)


@reg_op("LogicalAnd")
def logical_and(x, y, target=utils.CUDA):
    """LogicalAnd."""
    return math.LogicalAnd(x, y, target)


@reg_op("LogicalNot")
def logical_not(data, target=utils.CUDA):
    """LogicalNot"""
    return math.LogicalNot(data, target)


@reg_op("NotEqual")
def not_equal(x, y, target=utils.CUDA):
    """NotEqual"""
    return math.NotEqual(x, y, target)


@reg_op("GreaterEqual")
def greater_equal(x, y, target=utils.CUDA):
    """GreaterEqual"""
    return math.GreaterEqual(x, y, target)


@reg_op("Max")
def tensor_max(x, axis=None, keep_dims=False, target=utils.CUDA):
    """Max"""
    return math.ReduceMax(x, axis=axis, keepdims=keep_dims, target=target)


@reg_op("Neg")
def neg(x, target=utils.CUDA):
    """Neg"""
    return math.Neg(x, target)


@reg_op("Log")
def log(x, target=utils.CUDA):
    """Log"""
    return math.Log(x, target)


@reg_op("Less")
def less(x, y, target=utils.CUDA):
    """Less"""
    return math.Less(x, y, target)


@reg_op("Exp")
def exp(x, target=utils.CUDA):
    """Exp"""
    return math.Exp(x, target)


@reg_op("Sum")
def tensor_sum(data, axis=None, keepdims=False, target=utils.CUDA):
    """Sum"""
    return math.Sum(data, axis, keepdims, target=target)


@reg_op("Reshape")
def reshape(tensor, shape, target=utils.CUDA):
    """Reshape"""
    return array.Reshape(tensor, shape, target)


@reg_op("Reciprocal")
def reciprocal(x, target=utils.CUDA):
    """Reciprocal"""
    return math.Reciprocal(x, target)
