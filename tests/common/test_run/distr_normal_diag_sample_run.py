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
from tests.common.test_op.prob_program.distribution import normal_diag
from tests.common.gen_random import random_gaussian

def sample_op(mean, scale, eps):
  return normal_diag.normal_diag(mean, scale).sample(eps)

def sample_run(shape, dtype, kernel_name="", attrs=None):
    expect, mean, scale, eps, output = gen_data(dtype, shape)

    mod = utils.op_build_test(sample_op, [mean.shape, scale.shape, eps.shape],
                                         [dtype, dtype, dtype], kernel_name=kernel_name,
                                         op_attrs=None, attrs=None, log_code=True, dump_code=True, polyhedral=True)
    output = utils.mod_launch(mod, [mean, scale, eps, output], expect=expect)

    return (mean, scale, eps), output, expect, compare_tensor(output, expect, rtol=5e-03, atol=0.1, equal_nan=True)

def gen_data(dtype, shape):
    support_list = {"float16": np.float16, "float32": np.float32}

    m, k = shape

    mean  = random_gaussian((k,), miu=1, sigma=0.1).astype(support_list[dtype])
    scale = random_gaussian((k,), miu=1, sigma=0.1).astype(support_list[dtype])
    eps  = random_gaussian((m, k), miu=0, sigma=1).astype(support_list[dtype])

    output = np.full((m, k), 0.0, dtype)

    expect = mean + scale * eps

    return expect, mean, scale, eps, output
