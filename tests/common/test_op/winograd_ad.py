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

"""operator dsl function: winograd_ad"""

import akg
import akg.topi

def winograd_ad(head, a):
    b = akg.topi.nn.conv2d_winograd_weight_transform(a, 2)
    _jacs = list(akg.differentiate(b, [a], head))
    return _jacs[0]
