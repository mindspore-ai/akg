# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

"""operator dsl function: asinh"""

import math
import akg.utils as utils
from akg import topi
from akg import tvm
from akg.utils.kernel_exec import product_is_mini
from .sign import Sign
from ..log import log


def sqrt_mini_newton_iter_impl(x):
    """sqrt compute on mini with the Newton's Iteration"""

    # mini supports the rsqrt instruction, but not the sqrt instruction
    x_rsqrt = topi.rsqrt(x)
    x_sqrt = topi.divide(1, x_rsqrt)

    # newton_iter: x(n+1) = 1/2 *(x(n) + a/x(n))
    steps = 3
    half = tvm.const(0.5, x.dtype)
    shape = x.shape
    for i in range(steps):
        x_sqrt = tvm.compute(shape, lambda *indice: half * (x_sqrt(*indice) + x(*indice)/x_sqrt(*indice)),
                             name="x_sqrt_%s" % i)
    return x_sqrt


def log_compute_mini_impl(x, target=utils.CCE):
    """
    log compute on mini for x >= 1

    compute method:
    As vlog instruction has some precision problems when x in interval [1,2), the taylor method be used to
    calculate log value of x.
    For x in interval [1, 4/3),  calculate log value of x by the Taylor formula:
    log(1+x) = ((((0.2x - 0.25)x + 0.33333)x - 0.5)x + 1)x.
    For x in interval [4/3, 5/3) and [5/3, 2), x are mapped to in interval [1, 4/3), by the following formulas:
    [4/3, 5/3) -> log(x * 3/4) + log(4/3),
    [5/3, 2) -> log(x * 3/5) + log(5/3).
    For x in interval [2, 32768), calculate log value of x by vlog instruction directly:
    [2, 32768) -> log(x).
    As vlog instruction has overflow problems when x greater or equal to 32768, calculate log value of x
    by the following formulas:
    [32768, ) -> log(x/2.5) + log(2.5).
    """
    thresholds = [4/3, 5/3, 2, 32768]
    thresholds_rec = [3/4, 3/5]
    log_thresholds = [0.28768207245178085, 0.5108256237659907]
    overflow_div_coffient = 2.5
    log_overflow_div_coffient = 0.916290731874155

    def _log_taylor(data):
        """log algrithm is log(1+x) = ((((0.2x - 0.25)x + 0.33333)x - 0.5)x + 1)x"""
        data = topi.subtract(data, 1)
        taylor_params = [0.2, -0.25, 1/3, -0.5, 1]
        taylor_five = topi.multiply(data, taylor_params[0])
        taylor_four_1 = topi.add(taylor_five, taylor_params[1])
        taylor_four_2 = topi.multiply(taylor_four_1, data)
        taylor_three_1 = topi.add(taylor_four_2, taylor_params[2])
        taylor_three_2 = topi.multiply(taylor_three_1, data)
        taylor_two_1 = topi.add(taylor_three_2, taylor_params[3])
        taylor_two_2 = topi.multiply(taylor_two_1, data)
        taylor_one = topi.add(taylor_two_2, taylor_params[4])
        taylor = topi.multiply(taylor_one, data)
        return taylor

    # taylor
    shape = x.shape
    threshold_2 = tvm.const(thresholds[1], "float16")
    threshold_1 = tvm.const(thresholds[0], "float16")
    threshold_2_rec = tvm.const(thresholds_rec[1], "float16")
    threshold_1_rec = tvm.const(thresholds_rec[0], "float16")
    x_fp16 = topi.cast(x, "float16")
    x_1 = tvm.compute(shape, lambda *indice: tvm.expr.Select(x_fp16(*indice) >= threshold_2,
                                                             x_fp16(*indice)*threshold_2_rec, x_fp16(*indice)),
                      name="x_1")
    x_2 = tvm.compute(shape, lambda *indice: tvm.expr.Select(x_1(*indice) >= threshold_1,
                                                             x_1(*indice)*threshold_1_rec, x_1(*indice)),
                      name="x_2")
    taylor = _log_taylor(topi.cast(x_2, "float32"))
    log_threshold_1 = log_thresholds[0]
    log_threshold_2 = log_thresholds[1]
    taylor_add_log_threshold_1_fp16 = topi.cast(topi.add(taylor, log_threshold_1), "float16")
    taylor_add_log_threshold_2_fp16 = topi.cast(topi.add(taylor, log_threshold_2), "float16")
    res = tvm.compute(shape, lambda *indice: tvm.expr.Select(x_1(*indice) >= threshold_1,
                                                             taylor_add_log_threshold_1_fp16(*indice),
                                                             taylor(*indice).astype("float16")),
                      name="res_1")
    res = tvm.compute(shape, lambda *indice: tvm.expr.Select(x_fp16(*indice) >= threshold_2,
                                                             taylor_add_log_threshold_2_fp16(*indice), res(*indice)),
                      name="res_2")

    # vlog
    x_log = log(x_fp16, target)
    res = tvm.compute(shape, lambda *indice: tvm.expr.Select(x_fp16(*indice) >= thresholds[2], x_log(*indice),
                                                             res(*indice)),
                      name="res_3")

    # overflow
    overflow_threshold = tvm.const(thresholds[3], "float16")
    res_overflow = topi.cast(topi.add(log(topi.multiply(x, 1/overflow_div_coffient), target), 
                                                                                        log_overflow_div_coffient), "float16")
    res = tvm.compute(shape, lambda *indice: tvm.expr.Select(x_fp16(*indice) >= overflow_threshold,
                                                             res_overflow(*indice), res(*indice)),
                      name="res_4")
    if res.dtype != x.dtype:
        res = topi.cast(res, x.dtype)
    return res


@utils.check_input_type(tvm.tensor.Tensor, (str, type(None)))
def asinh(x, target=utils.CCE):
    r"""
    Compute asinh function.

    .. math:: asinh(x) = log(x+\sqrt{x*x+1})

    Args:
        x (tvm.tensor.Tensor): Tensor of type float16, float32. 

    Returns:
       tvm.tensor.Tensor, has the same type and shape as x.
    
    Supported Platforms:
        'Ascend'
    """
    # check shape
    utils.check_shape(x)

    # check input tensor data_type
    utils.ops_dtype_check(x.dtype, utils.DtypeForDavinci.ALL_FLOAT)
    dtype = x.dtype

    # Known that, asinh(x) = log(x + sqrt(x*x+1)), and, asinh(-x) = -asinh(x)
    # If x is a large negative number, (x + sqrt(x*x+1)) will be close to zero.
    # So, asinh(x) = sign(x) * log(|x| + sqrt(|x|*|x| + 1))
    compute_dtype = dtype
    if dtype == "float16":
        # To avoid overflow and higher accuracy, x is casted to float32
        compute_dtype = "float32"
        x = topi.cast(x, compute_dtype)

    x_abs = topi.abs(x)

    if product_is_mini():
        # sqrt(|x|*|x| + 1) = |x| * sqrt(1 + 1/(|x|*|x|))
        vsquare_add_one = topi.add(1, topi.divide(1, topi.multiply(x_abs, x_abs)))
        sqrt_compute_value = sqrt_mini_newton_iter_impl(vsquare_add_one)
        sqrt_value = topi.multiply(x_abs, sqrt_compute_value)
    else:
        x_abs_square_add_one = topi.add(topi.multiply(x_abs, x_abs), 1)
        sqrt_value = topi.sqrt(x_abs_square_add_one)

    x_add_sqrt = topi.add(x_abs, sqrt_value)

    if product_is_mini():
        log_value = log_compute_mini_impl(x_add_sqrt, target)
    else:
        log_value = topi.log(x_add_sqrt)

    res = topi.multiply(Sign(x, target), log_value)

    if res.dtype != dtype:
        res = topi.cast(res, dtype)

    if product_is_mini():
        attrs = {"enable_auto_inline": False}
        return res, attrs
    return res
