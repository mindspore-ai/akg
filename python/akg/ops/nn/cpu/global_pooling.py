# Copyright 2022 Huawei Technologies Co., Ltd
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

"""operator dsl function: global pooling"""
import akg.topi as topi
from akg.topi.util import get_const_tuple
import akg.tvm as tvm


def global_pooling(data, pool_type,
                   data_layout="NCHW"):
    """Global pooling op impl"""
    if data_layout == "NHWC":
        red_axis = (1, 2)
    else:
        # data_layout is NCHW or NCHWc
        red_axis = (2, 3)

    if pool_type == "max":
        out = topi.max(data, axis=red_axis, keepdims=True)
    elif pool_type == "avg":
        out = topi.sum(data, axis=red_axis, keepdims=True)

        count = 1
        for i in red_axis:
            count *= data.shape[i]
        out = topi.divide(out, count)
    else:
        raise ValueError(
            "pool_type should be max/avg, current pool_type is {}".format(pool_type))

    return out
