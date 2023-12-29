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

"""operator dsl function: softplus_ad"""

import akg
from tests.common.test_op.ascend import softplus
from akg.utils import custom_tiling as ct_util

def softplus_ad(head, a, target="cce"):
    b = softplus.softplus(a)
    _jacs = list(akg.differentiate(b, [a], head))
    return _jacs[0]
