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

"""operator dsl function:ceil"""
import akg
import akg.utils as utils


def ceil(inputs, target=utils.CCE):
    """
    Supported Platforms:
        'Ascend'
    """
    dtype = inputs.dtype

    check_list = ["float16", "float32"]
    if dtype.lower() not in check_list:
        raise RuntimeError("tile_cce only support %s while dtype is %s" % (",".join(check_list), dtype))

    shape = inputs.shape
    utils.check_shape(shape)

    res = akg.lang.ascend.ceil(inputs)

    return res
