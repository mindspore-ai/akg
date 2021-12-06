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

import akg.topi
import akg.tvm
import akg
import akg.lang.ascend


class normal_unit_var:
    """
    A tensor of normal distributions with variance equal to 1

    Args:
        log_prob (x):
            logarithm of probability density function for each tensor element
    """

    def __init__(self, inputs):
        self.mean = inputs

    def log_prob(self, outputs):
        return akg.tvm.compute(
            outputs.shape,
            lambda *indices: -(outputs(*indices) - self.mean(*indices)) * (outputs(*indices) - self.mean(*indices))
        )
