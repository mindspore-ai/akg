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

"""operator dsl function:clip"""
import akg.topi

from akg.utils import validation_check as vc_util

def clip(data, min_val, max_val):
    """
    Clip the data in range(min_val, max_val).

    Change values less than min_val in data to min_val, and change values greater than max_val to max_val.

    Note:
        min_val should be smaller or equal to max_val.

    Args:
        data: Tensor.
        min_val: Float. When data < min_val, set data to min_val.
        max_val: Float. When data > max_val, set data to max_val.

    Returns:
        Tensor, has the same type and shape as data.
    """

    dtype = data.dtype
    check_list = ["float16", "float32"]
    if not dtype.lower() in check_list:
        raise RuntimeError("clip only support %s while dtype is %s" % (",".join(check_list), dtype))

    shape = data.shape
    vc_util.check_shape(shape)

    res = akg.topi.clip(data, min_val, max_val)

    return res
