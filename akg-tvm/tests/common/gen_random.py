# Copyright 2019-2022 Huawei Technologies Co., Ltd
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

"""generate gaussian random array"""


def random_gaussian(size, miu=0, sigma=8, epsilon=0, seed=None):
    """Generate random array with absolution value obeys gaussian distribution."""
    from akg.utils import gen_random as random
    return random.random_gaussian(size, miu, sigma, epsilon, seed)


def gen_epsilon(dtype):
    """Generate suggested epsilon according to data type."""
    from akg.utils import gen_random as random
    return random.gen_epsilon(dtype)
