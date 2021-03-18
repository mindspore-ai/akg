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

import akg
from akg.utils import custom_tiling as ct_util
from tests.common.test_op.prob_program.distribution import normal_unit_var

dist_normal_ad_set_dim_map = {
}


def dist_normal_ad_set_dim_func(x, w, y):
    """
    Lookup the dist_normal_prob_regr_set_dim_map and return hash_value and hash_key
    """
    key = []
    key.append(tuple(x.shape))
    key.append(tuple(w.shape))
    key.append(tuple(y.shape))
    key.append(x.dtype)
    hash_key = str(tuple(key))

    if hash_key in dist_normal_ad_set_dim_map.keys():
        return ct_util.set_dims(dist_normal_ad_set_dim_map[hash_key]), hash_key
    return "", hash_key

def prob_regression(x, w):
    """
    Create probabilistic regression model. Potentially, make distribution one of the parameters, and allow variable
    arguments
    """
    assert x.shape[1].value == w.shape[1].value

    pred = akg.topi.nn.dense(x, w)

    model = normal_unit_var.normal_unit_var(pred)

    return model

@ct_util.reg_set_dim_func(dist_normal_ad_set_dim_func)
def prob_regression_train(x, w, y):
    """
    One step of training probabilistic regression

    Args:
        x: input
        w: trained weight
        y: output
    Returns:
        dw: change in weight
    """

    model = prob_regression(x, w)

    log_prob = model.log_prob(y)

    neglik = akg.topi.sum(-log_prob, [0, 1])

    head = akg.tvm.compute(neglik.shape,
                           lambda *indices:
                           akg.tvm.const(1.0, dtype = y.dtype))

    dw = list(akg.differentiate(neglik, [w], head))

    return dw[0]
