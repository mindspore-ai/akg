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

import numpy as np
from tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from test_op.prob_program import distr_normal_diag_KLdiv_ad
from gen_random import random_gaussian
from base import get_rtol_atol


def KLdiv_ad_run(shape, dtype, kernel_name="", attrs=None):
    expects, head, mean, scale, outputs = gen_data(dtype, shape)

    mod = utils.op_build_test(distr_normal_diag_KLdiv_ad.normal_diag_KLdiv_ad, [head.shape, mean.shape, scale.shape],
                                         [dtype, dtype, dtype], kernel_name=kernel_name,
                                         op_attrs=None, attrs=None, log_cce=True, dump_cce=True, polyhedral=True)
    outputs = utils.mod_launch(mod, [head, mean, scale, *outputs], outputs=tuple(range(-len(outputs), 0)), expect=expects)
    outputs = list(outputs)

    result = True
    rtol, atol = get_rtol_atol("KL_div_ad", dtype)
    for i in range(len(outputs)):
        result &= compare_tensor(outputs[i], expects[i], rtol=rtol, atol=atol, equal_nan=True)

    return (head, mean, scale), outputs, expects, result

def gen_data(dtype, shape):
    support_list = {"float16": np.float16, "float32": np.float32}

    m, k = shape

    mean  = random_gaussian((m, k), miu=1, sigma=0.1).astype(support_list[dtype])
    scale = random_gaussian((m, k), miu=1, sigma=0.1).astype(support_list[dtype])
    head = random_gaussian((m, ), miu=1, sigma=0.1).astype(support_list[dtype])

    output1 = np.full((m, k), 0.0, dtype)
    output2 = np.full((m, k), 0.0, dtype)

    expect_mean = mean * head.reshape(-1, 1)
    expect_sigma = (scale - 1/scale) * head.reshape(-1, 1)
    expects = (expect_mean, expect_sigma)

    return expects, head, mean, scale, (output1, output2)
