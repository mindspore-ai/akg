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

"""operator dsl function:invert"""

import akg.tvm
from akg.lang import ascend as dav
import akg.utils as utils

def invert(data, target="cce"):
    check_list = ["uint16"]
    dtype = data.dtype

    if not (dtype.lower() in check_list):
        raise RuntimeError("invert only support %s while dtype is %s" % (",".join(check_list), dtype))
    shape = [x.value for x in data.shape]
    utils.check_shape(shape)

    res = akg.tvm.compute(shape, lambda *i: dav.fnot(data[i]), name="res")

    return res
