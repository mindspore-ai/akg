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

"""sigmoid_cross_entropy_with_logits"""
import akg.tvm
import akg.utils as utils
from akg.utils.format_transform import get_shape
from akg.ops.math import Exp, Log, Mul

def sigmoid_cross_entropy_with_logits(labels=None, logits=None, target="cce"):
    ##
    # \brief Computes sigmoid cross entropy given `logits`.
    #
    # \f[
    #   cost = lables * -log(sigmoid(logits)) + (1 - lables) * -log(1 - sigmoid(logits))
    # \f]
    # \param labels akg.tvm.Tensor of the same type and shape as `logits`.
    # \param  logits akg.tvm.Tensor of type float16, float32
    #
    # \return akg.tvm.Tensor of the same shape as `logits` with the componentwise logistic losses.
    ##

    if get_shape(logits) != get_shape(labels):
        raise ValueError("logits and labels must have the same shape  (%s vs %s)" %
                         (get_shape(logits), get_shape(labels)))
    if logits.dtype != labels.dtype:
        raise ValueError("logits and labels must have the same dtype  (%s vs %s)" %
                         (logits.dtype, labels.dtype))

    shape = logits.shape
    dtype = logits.dtype

    check_list = ["float16", "float32"]
    if not (dtype.lower() in check_list):
        raise RuntimeError("sigmoid_cross_entropy_with_logits only support %s while dtype is %s" % (",".join(check_list), dtype))

    #    z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
    # =  z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
    # =  max(x, 0) - x * z + log(1 + exp(-abs(x)))

    zero = akg.tvm.const(0, dtype=dtype)
    relu_logits = akg.tvm.compute(shape, lambda *indice: akg.tvm.expr.Select(logits(*indice) < zero, zero, logits(*indice)), name="relu_logits")
    neg_abs_logits = akg.tvm.compute(shape, lambda *indice: akg.tvm.expr.Select(logits(*indice) < zero, logits(*indice), logits(*indice) * -1), name="neg_abs_logits")
    sigmoid_logits = Exp(neg_abs_logits, target=target) + akg.tvm.const(1, dtype=dtype)
    ln_sigmoid_logits = Log(sigmoid_logits, target=target)
    logits_mul_lables = Mul(logits, labels, target=target)
    res = relu_logits - logits_mul_lables + ln_sigmoid_logits
    return res
