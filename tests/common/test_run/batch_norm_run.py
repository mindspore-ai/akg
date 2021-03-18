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
from akg.utils import kernel_exec as utils
from tests.common.test_op import batch_norm
from tests.common.tensorio import compare_tensor
from tests.common.gen_random import random_gaussian

def batch_norm_run(shape, dtype, eps, kernel_name, attrs=None,  polyhedral=True):
    if len(shape) == 5 :
        (n, c, h, w, c0) = shape
    else :
        (n, c, h, w) = shape
        c0 = None

    support_list = {"float16": np.float16, "float32": np.float32}
    op_attrs = [eps, polyhedral, attrs]

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        np_beta, np_gamma, np_mean, np_var = get_input_data(c, c0, dtype, support_list)
        mod = utils.op_build_test(batch_norm.batch_norm,
                                  [shape, np_mean.shape, np_var.shape, np_gamma.shape, np_beta.shape],
                                  [dtype, dtype, dtype, dtype, dtype], op_attrs, kernel_name=kernel_name, attrs=attrs,
                                  polyhedral=polyhedral, tuning=t)
        if t:
            expect, np_data, output = gen_data(c, c0, dtype, eps, np_beta, np_gamma, np_mean, np_var, shape, support_list)
            return mod, expect, (np_data, np_mean, np_var, np_gamma, np_beta, output)
        else:
            return mod
    else:
        np_beta, np_gamma, np_mean, np_var = get_input_data(c, c0, dtype, support_list)
        mod = utils.op_build_test(batch_norm.batch_norm,
                                  [shape, np_mean.shape, np_var.shape, np_gamma.shape, np_beta.shape],
                                  [dtype, dtype, dtype, dtype, dtype], op_attrs, kernel_name=kernel_name, attrs=attrs,
                                  polyhedral=polyhedral)
        expect, np_data, output = gen_data(c, c0, dtype, eps, np_beta, np_gamma, np_mean, np_var, shape, support_list)
        output = utils.mod_launch(mod, (np_data, np_mean, np_var, np_gamma, np_beta, output), expect=expect)
        return (np_data, np_mean, np_var, np_gamma, np_beta), output, expect, compare_tensor(output, expect, atol=0.01)


def gen_data(c, c0, dtype, eps, np_beta, np_gamma, np_mean, np_var, shape, support_list):
    np_data = random_gaussian(shape, miu=1, sigma=0.1).astype(support_list[dtype])
    var = np.abs(np_var)
    if c0 is None :
        chan_shape = (1, c, 1, 1)
    else :
        chan_shape = (1, c, 1, 1, c0)

    mean_bc = np.broadcast_to(np_mean, shape)
    var_bc = np.broadcast_to(var, shape)
    gamma_bc = np.broadcast_to(np_gamma, shape)
    beta_bc = np.broadcast_to(np_beta, shape)
    normalize_data = (np_data - mean_bc) / np.sqrt(var_bc + eps)
    expect = gamma_bc * normalize_data + beta_bc
    output = np.full(shape, np.nan, dtype)

    return expect, np_data, output


def get_input_data(c, c0, dtype, support_list) :
    if c0 is None :
        chan_shape = (1, c, 1, 1)
    else :
        chan_shape = (1, c, 1, 1, c0)

    np_mean = random_gaussian(chan_shape, miu=1, sigma=0.1).astype(support_list[dtype])
    np_var = random_gaussian(chan_shape, miu=1, sigma=0.1).astype(support_list[dtype])
    np_gamma = random_gaussian(chan_shape, miu=1, sigma=0.1).astype(support_list[dtype])
    np_beta = random_gaussian(chan_shape, miu=1, sigma=0.1).astype(support_list[dtype])
    return np_beta, np_gamma, np_mean, np_var
