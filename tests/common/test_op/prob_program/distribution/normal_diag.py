# Copyright 2019-2021 Huawei Technologies Co., Ltd
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

import akg.tvm
import akg.topi
import akg
import numpy as np


class normal_diag:
    """
    Multivariate normal distribution with diagonal covariance matrix
    tensor shape is the same as mean and scale

    Args:
        log_prob (x):
            logarithm of probability density function
    """
    def __init__(self, mean, scale, dtype="float16"):
        self.mean = mean
        self.scale = scale
        self.cov = self.scale * self.scale
        self.dtype = dtype

    def log_prob(self, x):
        # logpdf(x) = -1/2( SUM_i [ (x_i-mu_i)^2 * sigma_i^(-1)  + ln(sigma_i) + ln(2pi) ]  )
        diff = x - self.mean
        temp = (diff * diff / self.cov + akg.topi.log(self.cov) + akg.tvm.const(np.log(2 * np.pi), self.dtype))
        result = akg.topi.sum(temp, list(range(-len(self.mean.shape), 0))) * akg.tvm.const(-0.5, self.dtype)
        return result

    def sample(self, eps):
        sample_data = self.mean + self.scale * eps
        return sample_data

    def KL_divergence(self):
        temp = akg.topi.log(self.cov) - (self.mean * self.mean) - self.cov + akg.tvm.const(1.0, self.dtype)
        result = akg.topi.sum(temp, -1) * akg.tvm.const(-0.5, self.dtype)
        return result