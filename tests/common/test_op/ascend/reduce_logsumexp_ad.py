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

"""operator dsl function: reduce_logsumexp_ad"""

import akg
from tests.common.test_op.ascend import reduce_logsumexp

def reduce_logsumexp_ad(head, a, axis, keepdims, target="cce"):
    b = reduce_logsumexp.reduce_logsumexp(a, axis, keepdims)
    _jacs = list(akg.differentiate(b, [a], head))
    return _jacs[0]
