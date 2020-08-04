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
from test_op.prob_program.distribution import bernoulli as akg_bernoulli
from scipy.stats import bernoulli, uniform
from numpy.random import seed
import sys
#np.set_printoptions(threshold=sys.maxsize)

def log_prob_op(x, probs):
  return akg_bernoulli.bernoulli(probs).log_prob(x)

def log_prob_run(shape, dtype, kernelname="", attrs = None):
  expect, x, probs, output = gen_data(dtype, shape)

  mod = utils.op_build_test(log_prob_op, [x.shape, probs.shape],
                            [dtype, dtype], kernel_name=kernelname,
                            op_attrs=[], attrs=None, log_cce=True, dump_code=True, polyhedral=True)
  output = utils.mod_launch(mod, [x, probs, output], expect=expect)
  return (x, probs), output, expect, compare_tensor(output, expect, rtol=1e-03, atol=1e-03, equal_nan=True)

def gen_data(dtype, shape):
  support_list = {"float16": np.float16, "float32": np.float32}
  seed(0)
  m, k = shape
  x = bernoulli.rvs(0.5, size=(m, k)).astype(support_list[dtype])
  eps = 1e-3
  # generate probabilities in the range [eps, 1 - eps], to avoid mismatch between np.inf and computed
  # inf = -65500.0, due to taking log
  probs  = uniform(eps, 1.0 - 2.0 * eps).rvs(size=(m, k)).astype(support_list[dtype])
  expect = bernoulli.logpmf(x, probs)
  output = np.full((m, k), 0.0, dtype)

  return expect, x, probs, output
