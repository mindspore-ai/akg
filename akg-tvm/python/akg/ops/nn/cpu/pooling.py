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

"""operator dsl function: pooling"""
import akg.topi as topi
from akg.topi.util import get_const_tuple
import akg.tvm as tvm

def pooling(data, kernel, stride, padding, pool_type, 
                ceil_mode, count_include_pad=True,
                data_layout="NCHW"):
    """Pooling op impl"""
    out = topi.nn.pool(data, kernel=kernel, stride=stride, padding=padding,
                pool_type=pool_type, ceil_mode=ceil_mode,
                layout=data_layout, count_include_pad=count_include_pad)
    return out