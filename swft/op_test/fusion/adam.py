#!/usr/bin/env python3
# coding: utf-8
# Copyright 2025 Huawei Technologies Co., Ltd
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

import numpy as np
import os
import sys
from swft.core import *
from swft.api import *

OP_NAME = 'adam'
os.system(f"mkdir -p temp/{OP_NAME}")
os.system(f"mkdir -p temp/{OP_NAME}/input")
os.system(f"mkdir -p temp/{OP_NAME}/output")

INF = 60000
S = 512

# Numpy Test
# ===============================================================================


def gen_data():
    weight = np.random.uniform(-0.5, 0.5, [S]).astype(np.float32)
    m = np.random.uniform(0, 0.5, [S]).astype(np.float32)
    v = np.random.uniform(0, 0.5, [S]).astype(np.float32)
    grad = np.random.uniform(-1, 1, [S]).astype(np.float32)
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-5
    learning_rate = 1
    t = 1
    beta_pow2 = 1 / (1 - beta2 ** t)
    beta_pow1 = 1 / (1 - beta1 ** t)
    eps = epsilon / np.sqrt(beta_pow2)

    m_new = beta1 * m + (1-beta1) * grad
    v_new = beta2 * v + (1-beta2) * (grad ** 2)
    m_hat = m_new * beta_pow1
    v_hat = v_new * beta_pow2
    out = weight - learning_rate * m_hat / (np.sqrt(v_hat) + eps)

    weight.tofile(f"./temp/{OP_NAME}/input/weight.bin")
    m.tofile(f"./temp/{OP_NAME}/input/m.bin")
    v.tofile(f"./temp/{OP_NAME}/input/v.bin")
    beta1 = np.array([beta1], dtype=np.float32)
    beta1.tofile(f"./temp/{OP_NAME}/input/beta1.bin")
    beta2 = np.array([beta2], dtype=np.float32)
    beta2.tofile(f"./temp/{OP_NAME}/input/beta2.bin")
    beta_pow1 = np.array([beta_pow1], dtype=np.float32)
    beta_pow1.tofile(f"./temp/{OP_NAME}/input/beta1_pow.bin")
    beta_pow2 = np.array([beta_pow2], dtype=np.float32)
    beta_pow2.tofile(f"./temp/{OP_NAME}/input/beta2_pow.bin")
    learning_rate = np.array([learning_rate], dtype=np.float32)
    learning_rate.tofile(f"./temp/{OP_NAME}/input/learning_rate.bin")
    eps = np.array([eps], dtype=np.float32)
    eps.tofile(f"./temp/{OP_NAME}/input/eps.bin")
    grad.tofile(f"./temp/{OP_NAME}/input/gradient.bin")
    out.tofile(f"./temp/{OP_NAME}/output/weight_golden.bin")
    m_new.tofile(f"./temp/{OP_NAME}/output/m_golden.bin")
    v_new.tofile(f"./temp/{OP_NAME}/output/v_golden.bin")

# OP Impl
# ===============================================================================


@sub_kernel(core_num=8)
def adam(gradient, weight, m, v, beta1, beta2, beta1_pow, beta2_pow, eps, learning_rate):
    '''
    m = beta1 * m + (1-beta1) * gradient
    v = beta2 * v + (1-beta2) * (gradient**2)
    m_hat = m / (1 - beta1 ** t)
    v_hat = m / (1 - beta2 ** t)
    theta = theta - learningrate * m_hat / (np.sqrt(v_hat)+epsilon)
    '''
    ub_grad = move_to_ub(gradient)
    ub_weight = move_to_ub(weight)
    ub_m = move_to_ub(m)
    ub_v = move_to_ub(v)
    ub_beta1m = vmuls(ub_m, beta1)
    ub_beta2v = vmuls(ub_v, beta2)
    ub_beta1g = vmuls(ub_grad, beta1)
    ub_g2 = vmul(ub_grad, ub_grad)
    ub_beta2g2 = vmuls(ub_g2, beta2)
    ub_beta1g = vsub(ub_grad, ub_beta1g)
    ub_beta2g2 = vsub(ub_g2, ub_beta2g2)
    ub_m_new = vadd(ub_beta1m, ub_beta1g)
    ub_v_new = vadd(ub_beta2v, ub_beta2g2)
    ub_m_hat = vmuls(ub_m_new, beta1_pow)
    ub_v_hat = vmuls(ub_v_new, beta2_pow)
    ub_m_hat_learning_rate = vmuls(ub_m_hat, learning_rate)
    ub_v_hat_sqrt = vsqrt(ub_v_hat)
    ub_v_hat_sqrt_epsilon = vadds(ub_v_hat_sqrt, eps)
    ub_m_hat_div = vdiv(ub_m_hat_learning_rate, ub_v_hat_sqrt_epsilon)
    ub_weight_new = vsub(ub_weight, ub_m_hat_div)
    weight.load(ub_weight_new)
    m.load(ub_m_new)
    v.load(ub_v_new)


if __name__ == '__main__':
    set_context("310P")
    gen_data()
    gradient = Tensor(
        "GM", "FP32", [S], format="ND", multi_core=True)
    weight = Tensor("GM", "FP32", [S], format="ND", multi_core=True)
    m = Tensor("GM", "FP32", [S], format="ND", multi_core=True)
    v = Tensor("GM", "FP32", [S], format="ND", multi_core=True)
    eps = Scalar("FP32")
    learning_rate = Scalar("FP32")
    beta1 = Scalar("FP32")
    beta2 = Scalar("FP32")
    beta1_pow = Scalar("FP32")
    beta2_pow = Scalar("FP32")
    adam(gradient, weight, m, v, beta1, beta2,
         beta1_pow, beta2_pow, eps, learning_rate)
    compile_kernel(f"./temp/{OP_NAME}/{OP_NAME}.cce", OP_NAME)
    exec_kernel(OP_NAME, locals(), prefix_path="temp", inputs=['gradient', 'weight', 'm', 'v', 'beta1', 'beta2',
                'beta1_pow', 'beta2_pow', 'eps', 'learning_rate'], outputs=['m', 'v', 'weight'])
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return_code_1 = os.system(
        f'python3 {script_dir}/../verify_result.py ./temp/{OP_NAME}/output/m_actual.bin ./temp/{OP_NAME}/output/m_golden.bin float32 4e-2 1e-2 4e-3')
    return_code_2 = os.system(
        f'python3 {script_dir}/../verify_result.py ./temp/{OP_NAME}/output/v_actual.bin ./temp/{OP_NAME}/output/v_golden.bin float32 4e-2 1e-2 4e-3')
    return_code_3 = os.system(
        f'python3 {script_dir}/../verify_result.py ./temp/{OP_NAME}/output/weight_actual.bin ./temp/{OP_NAME}/output/weight_golden.bin float32 4e-2 1e-2 4e-3')
    sys.exit(return_code_1 >> 8 or return_code_2 >> 8 or return_code_3 >> 8)
