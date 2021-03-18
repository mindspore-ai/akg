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

"""operator dsl function:div_no_nan"""
import akg.topi
from akg import tvm
from akg.utils import validation_check as vc_util
from akg.utils import kernel_exec as utils
from akg.utils.dsl_create import produce_shapes
from akg.ops.math import mul, div, abs
from akg.utils.format_transform import get_shape

def _refine_is_zero(is_zero, abs_clip_y_fp32):
    """
    refine is_zero for fp32 in mini.
    
    input is_zero is true if y < 2^-12.
    is_zero is true if y < 2^-38 after loop 1.
    is_zero is true if y < 2^-64 after loop 2.
   
    Args:
        is_zero (akg.tvm.tensor.Tensor): input
        abs_clip_y_fp32 (akg.tvm.tensor.Tensor): input y
    
    Return:
        is_zero, akg.tvm.tensor.Tensor
    """
    loop = [1, 2]
    for _, _ in enumerate(loop):
        is_zero_fp32 = akg.lang.cce.cast_to(is_zero, "float32")
        shift_dot = akg.lang.cce.vmuls(is_zero_fp32, 2**26)
        
        not_zero = tvm.compute(is_zero.shape,
                               lambda *i : (1 - is_zero(*i)).astype("float32"),
                               name="not_zero")
        
        shift_small = akg.lang.cce.vadd(shift_dot, not_zero)
        
        # only multiply those y < 2^-12 by 2^26, others mulitply 1
        abs_clip_y_fp32 = akg.lang.cce.vmul(abs_clip_y_fp32, shift_small)
        y_cmp = akg.lang.cce.cast_to(abs_clip_y_fp32, "float16")
        
        is_zero = tvm.compute(is_zero.shape,
                              lambda *i : tvm.expr.Select(
                                  y_cmp(*i) < tvm.const(2**(-12), dtype="float16"),
                                  tvm.const(1, dtype="float16"), 
                                  tvm.const(0, dtype="float16")),
                              name="is_zero")
    return is_zero


@vc_util.check_input_type(tvm.tensor.Tensor, tvm.tensor.Tensor)
def div_no_nan(data_x, data_y):
    """
    Returns 0 if the denominator is zero, else, like Div.

    Args:
        data_x (tvm.tensor.Tensor): tensor with type int32/int8/uint8, float16/float32.
        data_y (tvm.tensor.Tensor): tensor with type int32/int8/uint8, float16/float32.

    Returns:
        tvm.tensor.Tensor.
    """
    dtype = data_x.dtype
    if dtype != data_y.dtype:
        raise TypeError("input dtype should be the same")
    vc_util.ops_dtype_check(dtype, [vc_util.DtypeForDavinci.ALL_FLOAT, 
                                    vc_util.DtypeForDavinci.INT8,
                                    vc_util.DtypeForDavinci.UINT8, 
                                    vc_util.DtypeForDavinci.INT32])

    vc_util.check_shape(data_x.shape)
    vc_util.check_shape(data_y.shape)
    vc_util.auto_broadcast_check(data_x, data_y)

    # dtype for vsel and vcmp
    if utils.product_is_mini():
        compute_dtype = "float16"
    else:
        compute_dtype = "float32"
 
    # div fp16 y returns 0 if y < 2^-12
    # div fp32 y returns 0 if y < 2^-64
    min_val = tvm.const(2**(-12) if utils.product_is_mini() else 2**(-64),
                        dtype=compute_dtype)    

    tvm_one = tvm.const(1, dtype=compute_dtype)
    tvm_zero = tvm.const(0, dtype=compute_dtype)
    
    if not utils.product_is_mini() and dtype == "float16":
        min_val = tvm.const(2**(-12), "float32")

    data_y_fp32 = akg.lang.cce.cast_to(data_y, "float32")
    # avoid when y > 2^15 cast from fp32 to fp16 in mini
    clip_y_fp32 = akg.topi.clip(data_y_fp32, -1.0, 1.0)
    abs_clip_y_fp32 = abs.abs_value(clip_y_fp32)
    y_cmp = akg.lang.cce.cast_to(abs_clip_y_fp32, compute_dtype) 

    is_zero = tvm.compute(data_y.shape,
                          lambda *i : tvm.expr.Select(
                              y_cmp(*i) < min_val, tvm_one, tvm_zero), 
                          name="is_zero")    
    
    # if fp32 y < 2^-24, cast(y,fp16)==0. to find y in (2^-64, 2^-24): 
    if utils.product_is_mini() and dtype == "float32":
        is_zero = _refine_is_zero(is_zero, abs_clip_y_fp32)
    
    is_zero = akg.lang.cce.cast_to(is_zero, "float32")
    not_zero = tvm.compute(data_y.shape,
                           lambda *i : (1 - is_zero(*i)).astype("float32"),
                           name="not_zero")    
   
    # replace [x1 x2]/[y1 0] by [x1 0]/[y1 1] 
    data_x = mul.mul(akg.lang.cce.cast_to(data_x, "float32"), not_zero)
    data_y = akg.lang.cce.cast_to(data_y, "float32") + is_zero
    res = div.div(data_x, data_y)

    if dtype in ("int8", "uint8", "int32"):
        res = akg.lang.cce.floor(res)
        res = akg.lang.cce.cast_to(res, dtype)
    else:
        res = akg.lang.cce.cast_to(res, dtype)
    return res
