# Copyright 2019 Huawei Technologies Co., Ltd
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

"""operator dsl function: erf"""
from akg import tvm, topi
from akg.utils.format_transform import get_shape
import akg.utils.dsl_create as dc
from akg.utils import validation_check as vc_util, kernel_exec as utils

from akg.ops.math.div import div
from akg.ops.math.exp import exp
from tests.common.test_op.round import round_value

SCALER_P = 0.3275911
SCALER_A1 = 0.254829592
SCALER_A2 = -0.284496736
SCALER_A3 = 1.421413741
SCALER_A4 = -1.453152027
SCALER_A5 = 1.061405429

# define a scaler, value = 32768 = 2**15
SCALER_FP16_MAX = 32768
# define a scaler, value = 2**(-15)
SCALER_FP16_MIN = 2**(-15)


def _erf_compute(input_x):
    r"""
    Compute erf.

    .. math::
        \operatorname{erf}(x) = sign(x) \left(
            1 - (a_1t+a_2t^2+a_3t^3+a_4t^4+a_5t^5) e^{-x^2} + \epsilon(|x|)
            \right), \\
        t = \dfrac{1}{1+p|x|} \\
        \left|\epsilon(|x|)\right| \le 1.5 \times 10^{-7} \\
        where \; p=.3275911 \quad a_1=.254829592 \quad a_2=-.284496736 \\
        a_3=1.421413741 \quad a_4=-1.453152027 \quad a_5=1.061405429

    Args:
        input_x (tvm.tensor.Tensor): Input tensor.

    Returns:
        tvm.tensor.Tensor as rational approximation.
    """

    dtype = input_x.dtype
    shape = get_shape(input_x)

    cst_one = dc.one_const("float32")
    cst_neg_one = dc.neg_one_const("float32")
    cst_p = tvm.const(SCALER_P, "float32")
    cst_a1 = tvm.const(SCALER_A1, "float32")
    cst_a2 = tvm.const(SCALER_A2, "float32")
    cst_a3 = tvm.const(SCALER_A3, "float32")
    cst_a4 = tvm.const(SCALER_A4, "float32")
    cst_a5 = tvm.const(SCALER_A5, "float32")
    fp16_max = tvm.const(SCALER_FP16_MAX, "float32")
    fp16_min = tvm.const(SCALER_FP16_MIN, "float32")

    if dtype == "float16":
        input_x = topi.cast(input_x, "float32")

    # calculate: sign = floor[(x*fp16max) / (|x*fp16max| + fp16min)]
    data_sign_vmuls = topi.multiply(input_x, fp16_max)
    data_sign_abs = topi.abs(data_sign_vmuls)
    data_adds = topi.add(data_sign_abs, fp16_min)
    data_sign_div = div(data_sign_vmuls, data_adds)
    data_round = round_value(data_sign_div)
    # mini device should cast to fp16 first
    if utils.product_is_mini():
        data_round = topi.cast(data_round, "float16")
    tensor_sign = topi.cast(data_round, "float32")

    # t = 1 / (1 + px)
    tensor_abs = topi.abs(input_x)
    one_plus_px = topi.add(cst_one, topi.multiply(tensor_abs, cst_p))
    data_t = div(topi.full(shape, "float32", 1.0), one_plus_px)

    # e^{-x^2}
    abs_square = topi.multiply(tensor_abs, tensor_abs)
    neg_square = topi.multiply(abs_square, cst_neg_one)
    exp_neg_square = exp(neg_square)

    # a1t + a2t^2 + a3t^3 + a4t^4 + a5t^5 = ((((a5t + a4)t + a3)t + a2)t + a1)t
    tmp_a5 = topi.multiply(cst_a5, data_t)
    tmp_a5a4 = topi.multiply(topi.add(tmp_a5, cst_a4), data_t)
    tmp_a5a4a3 = topi.multiply(topi.add(tmp_a5a4, cst_a3), data_t)
    tmp_a5a4a3a2 = topi.multiply(topi.add(tmp_a5a4a3, cst_a2), data_t)
    data_muladd = topi.multiply(topi.add(tmp_a5a4a3a2, cst_a1), data_t)

    # erf = sign(x) * (1 - data_muladd * e^{-x^2})
    erf_res = topi.multiply(
        tensor_sign,
        topi.add(cst_one, topi.multiply(
            cst_neg_one, topi.multiply(data_muladd, exp_neg_square))))

    if dtype == "float16":
        erf_res = topi.cast(erf_res, dtype)

    return erf_res


@vc_util.check_input_type(tvm.tensor.Tensor)
def erf(input_x):
    r"""
    Computes the Gauss error function of input_x.

    .. math::
        \operatorname{erf}(x) =
            \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2} \mathrm{d} t

    Args:
        input_x (tvm.tensor.Tensor): Input tensor, only support float16, float32.

    Returns:
        tvm.tensor.Tensor with the same shape and dtype as input_x.
    """

    vc_util.ops_dtype_check(input_x.dtype, vc_util.DtypeForDavinci.ALL_FLOAT)
    vc_util.check_shape(input_x.shape)

    return _erf_compute(input_x)
