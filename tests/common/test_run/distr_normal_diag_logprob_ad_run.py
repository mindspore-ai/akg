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
from tests.common.tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from tests.common.test_op.prob_program import distr_normal_diag_logprob_ad
from tests.common.gen_random import random_gaussian, gen_epsilon


def logprob_ad_run(shape, dtype, kernel_name="", attrs=None):
    expects, head, x, mean, scale, outputs = gen_data(dtype, shape)

    mod = utils.op_build_test(
        distr_normal_diag_logprob_ad.normal_diag_logprob_ad,
        [head.shape, x.shape, mean.shape, scale.shape],
        [dtype, dtype, dtype, dtype],
        kernel_name=kernel_name,
        op_attrs=None,
        attrs=None,
        log_cce=True,
        dump_code=True,
        polyhedral=True,
    )
    outputs = utils.mod_launch(
        mod,
        [head, x, mean, scale, *outputs],
        outputs=tuple(range(-len(outputs), 0)),
        expect=expects
    )
    outputs = list(outputs)
    result = True
    for i in range(len(outputs)):
        result &= compare_tensor(outputs[i], expects[i], rtol=5e-03, equal_nan=True)

    return (head, x, mean, scale), outputs, expects, result


def gen_data(dtype, shape):
    support_list = {"float16": np.float16, "float32": np.float32}

    m, k = shape
    x = random_gaussian((m, k), miu=1, sigma=0.1).astype(support_list[dtype])
    mean = random_gaussian((k,), miu=1, sigma=0.1).astype(support_list[dtype])
    scale = random_gaussian((k,), miu=1, sigma=0.1, epsilon=1e-3).astype(support_list[dtype])

    output1 = np.full((m, k), 0.0).astype(support_list[dtype])
    output2 = np.full((k,), 0.0).astype(support_list[dtype])
    output3 = np.full((k,), 0.0).astype(support_list[dtype])
    head = random_gaussian((m,), miu=1, sigma=0.1).astype(support_list[dtype])

    expect_x = -(x - mean) / (scale * scale) * head.reshape(-1, 1)
    expect_mean = np.sum((x - mean) / (scale * scale) * head.reshape(-1, 1), axis=0)
    expect_sigma = np.sum(
        -0.5 * (2.0 / scale - (x - mean) * (x - mean) / (scale * scale * scale) * 2) * head.reshape(-1, 1),
        axis=0)
    expects = (expect_x, expect_mean, expect_sigma)

    return expects, head, x, mean, scale, (output1, output2, output3)
