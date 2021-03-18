# Copyright 2020 Huawei Technologies Co., Ltd
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

"""operator dsl function:reduction_layer"""

import akg
from akg import topi
from akg import tvm
from akg.utils import validation_check as vc_util
from akg.utils import kernel_exec as utils
from akg.utils.format_transform import get_shape, refine_reduce_axis

def _get_axis_list(start_axis, shape):
    axis = list(range(start_axis, len(shape)))
    for i, _ in enumerate(axis):
        if axis[i] < 0:
            axis[i] = axis[i] + len(shape)
    axis = set(axis)
    axis = list(axis)
    axis.sort()
    return axis

def _asum(data, axis, cof):
    data_tmp_input = topi.abs(data)
    tmp = topi.multiply(data_tmp_input, cof)
    res = topi.sum(tmp, axis)
    return res

def _sumsq(data, axis, cof):
    data_tmp_input = topi.multiply(data, data)
    tmp = topi.multiply(data_tmp_input, cof)
    res = topi.sum(tmp, axis)
    return res

def _mean(data, axis, cof, shape):
    size = 1
    for i, _ in enumerate(axis):
        size = size * shape[axis[i]]
    cof = cof / tvm.const(size, "float32")
    tmp = topi.multiply(data, cof)
    res = topi.sum(tmp, axis)
    return res

def _sum(data, axis, cof):
    data_tmp_input = topi.multiply(data, cof)
    tmp = data_tmp_input
    res = topi.sum(tmp, axis)
    return res

@vc_util.check_input_type(akg.tvm.tensor.Tensor, int, str, (int, float))
def reduction_layer(data, axis, op, coeff):
    """
    Reduce data on axis and scale by coeff.

    Args:
        data (tvm.tensor.Tensor): tensor with type float16 or float32, int8, uint8.
        axis (int): the beginning axis to reduce, -1 means the last axis. if 0, reduction to scalar.
        op (str): one of "SUM", "ASUM"(abs and sum), "SUMSQ"(sqr and sum), "MEAN".
        coeff ([int, float]): scale
    Returns:
        tvm.tensor.Tensor.
    """
    dtype = data.dtype
    vc_util.ops_dtype_check(data.dtype, [vc_util.DtypeForDavinci.ALL_FLOAT, 
                                         vc_util.DtypeForDavinci.INT8,
                                         vc_util.DtypeForDavinci.UINT8])

    vc_util.check_shape(data.shape)

    if op not in ["SUM", "ASUM", "SUMSQ", "MEAN"]:
        raise RuntimeError("op can only be one of SUM, ASUM, SUMSQ, MEAN")
    
    shape = get_shape(data)
    
    vc_util.reduce_axis_check(shape, axis)
    axis = _get_axis_list(axis, shape)
    
    if dtype in ["int8", "uint8"]:
        data = topi.cast(data, "float16")
    data = topi.cast(data, "float32")
    cof = tvm.const(coeff, "float32")
   
    if op == "ASUM":
        tmp = _asum(data, axis, cof) 
    elif op == "SUMSQ":
        tmp =_sumsq(data, axis, cof) 
    elif op == "MEAN":
        tmp = _mean(data, axis, cof, shape)
    elif op == "SUM":
        tmp = _sum(data, axis, cof)
    
    if dtype in ["int8", "uint8"]:
        tmp = topi.cast(tmp, "float16")    
    res = topi.cast(tmp, dtype)
    
    return res
