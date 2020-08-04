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
from test_op.prob_program import distr_normal_prob_regr_train
from gen_random import random_gaussian


def prob_regression_run(shape, dtype, kernel_name, attrs):
    
    expect, x, w, y, output = gen_data(dtype, shape)

    mod = utils.op_build_test(distr_normal_prob_regr_train.prob_regression_train, [x.shape, w.shape, y.shape],
                                         [dtype, dtype, dtype], kernel_name=kernel_name,
                                         op_attrs=[], attrs=None, log_cce=True, dump_code=True, polyhedral=True)

    output = utils.mod_launch(mod, [x, w, y, output], expect=expect)
    return (x, w, y), output, expect, compare_tensor(output, expect, rtol=5e-03, equal_nan=True)

def gen_data(dtype, shape):
    support_list = {"float16": np.float16, "float32": np.float32}

    m, k = shape

    x = random_gaussian((m, k), miu=1, sigma=0.1).astype(support_list[dtype])
    w = random_gaussian((1, k), miu=1, sigma=0.1).astype(support_list[dtype])
    y = random_gaussian((m, 1), miu=1, sigma=0.1).astype(support_list[dtype])

    output = np.full((1, k), 0.0, dtype)

    pred = np.dot(x, w.transpose())
    expect = np.dot((pred - y).transpose(), x) * 2

    return expect, x, w, y, output
