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

"""operator dsl function:add_ad"""
import akg
import akg.utils as utils
from akg.ops.math import Add


def add_ad(head, a, b, scale, target="cce"):
    """Compute gradient of add operator using automatic differentiate."""
    output = Add(a, b, scale, target=target)
    jacs = list(akg.differentiate(output, [a], head))
    return jacs[0]

