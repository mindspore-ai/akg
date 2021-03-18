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
# limitations under the License

"""fused operator dsl function"""
from __future__ import absolute_import
import akg.tvm as tvm
import akg.topi as topi
from topi import tag

def relu_grad(head, in_data):
    shape = head.shape
    dtype = head.dtype

    zero = tvm.const(0, dtype)
    relugrad = tvm.compute(shape, lambda *i: tvm.if_then_else(in_data(*i) >= zero, head(*i), zero), tag=tag.INJECTIVE)
    return relugrad

def bn_beta_grad(head, layout='NHWC'):
    if layout == "NCHW":
        head = topi.tranpose(head, (0, 2, 3, 1))
    
    n, h, w, c = head.shape
    n = n.value
    h = h.value
    w = w.value
    c = c.value
    bn_beta_grad = topi.sum(head, axis=(0, 1, 2))
    return bn_beta_grad

def bn_gamma_grad(head, in_data, data_sum, layout="NHWC"):
    if layout == "NCHW":
        head = topi.tranpose(head, (0, 2, 3, 1))
    
    n, h, w, c = head.shape
    n = n.value
    h = h.value
    w = w.value
    c = c.value
    scale = tvm.const(n * h * w, head.dtype)
    mean = topi.divide(data_sum, scale)
    x_hat = topi.subtract(in_data, mean)
    x_hat_mul = topi.multiply(x_hat, head)
    bn_gamma_grad = topi.sum(x_hat_mul, axis=(0, 1, 2))
    return bn_gamma_grad
