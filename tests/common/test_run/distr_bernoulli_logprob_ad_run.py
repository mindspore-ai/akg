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
from test_op.prob_program import distr_bernoulli_logprob_ad
from scipy.stats import bernoulli, uniform
from gen_random import random_gaussian
from numpy.random import seed


def logprob_ad_run(shape, dtype, kernel_name="", attrs=None):
    expects, head, x, probs, outputs = gen_data(dtype, shape)

    mod = utils.op_build_test(distr_bernoulli_logprob_ad.bernoulli_logprob_ad, [head.shape, x.shape, probs.shape],
                                         [dtype, dtype, dtype], kernel_name=kernel_name,
                                         op_attrs=None, attrs=None, log_cce=True, dump_code=True, polyhedral=True)
    outputs = utils.mod_launch(mod, [head, x, probs, *outputs], outputs=tuple(range(-len(outputs), 0)), expect=expects)
    outputs = list(outputs)

    return (head, x, probs), outputs, expects, compare_tensor(outputs, expects, rtol=5e-03, atol=1e-03, equal_nan=True)

def gen_data(dtype, shape):
    support_list = {"float16": np.float16, "float32": np.float32}
    seed(0)

    m, k = shape

    x = bernoulli.rvs(0.5, size=(m, k)).astype(support_list[dtype])
    eps = 1e-3
    # generate probabilities in the range [eps, 1 - eps], to avoid mismatch between np.inf and computed
    # inf = -65500.0, due to taking log
    probs  = uniform(eps, 1.0 - 2.0 * eps).rvs(size=(m, k)).astype(support_list[dtype])
    head = random_gaussian((m, k), miu=1, sigma=0.1).astype(support_list[dtype])

    output1 = np.full((m, k), 0.0, dtype)
    output2 = np.full((m, k), 0.0, dtype)

    expect_x = (-np.log(1 - probs) + np.log(probs)) * head
    expect_prob = (x / probs - (1 - x) / (1 - probs)) * head
    expects = (expect_x, expect_prob)

    return expects, head, x, probs, (output1, output2)
