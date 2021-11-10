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

def TensorAdd(x, y, target=utils.CUDA):
    """TensorAdd"""
    return math.Add(x, y, scale=1.0, polyhedral=True, attrs={}, target=target)

def Add(x, y, target=utils.CUDA):
    """Add"""
    return math.Add(x, y, scale=1.0, polyhedral=True, attrs={}, target=target)

def AddN(inputs, target=utils.CUDA):
    """AddN"""
    return math.Addn(inputs, target)

def Assign(ref, val, target=utils.CUDA):
    """Assign"""
    return math.Assign(ref, val, target)

def Cast(x, dst_type, target=utils.CUDA):
    """Cast"""
    return math.Cast(x, dst_type, target)

def Equal(input1, input2, target=utils.CUDA):
    """Equal"""
    return math.Equal(input1, input2, target)

def LessEqual(input1, input2, target=utils.CUDA):
    """LessEqual"""
    return math.LessEqual(input1, input2, target)

def Mul(x, y, target=utils.CUDA):
    """Mul"""
    return math.Mul(x, y, target)

def Sub(x, y, target=utils.CUDA):
    """Sub"""
    return math.Sub(x, y, target)

def Div(x, y, target=utils.CUDA):
    """Div"""
    return math.Divide(x, y, target)

def Divide(x, y, target=utils.CUDA):
    """Divide"""
    return math.Divide(x, y, target)

def Tile(data, multiples, target=utils.CUDA):
    """Tile"""
    return array.Tile(data, multiples, target)

def LogicalOr(x, y, target=utils.CUDA):
    """LogicalOr"""
    return math.LogicalOr(x, y, target)

def LogicalAnd(x, y, target=utils.CUDA):
    """LogicalAnd."""
    return math.LogicalAnd(x, y, target)

def LogicalNot(data, target=utils.CUDA):
    """LogicalNot"""
    return math.LogicalNot(data, target)

def NotEqual(x, y, target=utils.CUDA):
    """NotEqual"""
    return math.NotEqual(x, y, target)

def GreaterEqual(x, y, target=utils.CUDA):
    """GreaterEqual"""
    return math.GreaterEqual(x, y, target)

def Max(x, axis=None, keep_dims=False, target=utils.CUDA):
    """Max"""
    return math.ReduceMax(x, axis=axis, keepdims=keep_dims, target=target)

def Neg(x, target=utils.CUDA):
    """Neg"""
    return math.Neg(x, target)

def Log(x, target=utils.CUDA):
    """Log"""
    return math.Log(x, target)

def Less(x, y, target=utils.CUDA):
    """Less"""
    return math.Less(x, y, target)

def Exp(x, target=utils.CUDA):
    """Exp"""
    return math.Exp(x, target)

def Sum(data, axis=None, keepdims=False, target=utils.CUDA):
    """Sum"""
    return math.Sum(data, axis, keepdims, target=target)

def Reshape(tensor, shape, target=utils.CUDA):
    """Reshape"""
    return array.Reshape(tensor, shape, target)

def Reciprocal(x, target=utils.CUDA):
    """Reciprocal"""
    return math.Reciprocal(x, target)