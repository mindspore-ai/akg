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

"""operator dsl function: reduce_logsumexp"""

import akg
from akg.lang.ascend import vexp, vlog, sum
import akg.utils as utils


def reduce_logsumexp(data, axis=None, keepdims=False, target="cce"):
    """
    Compute `log(sum(exp(elements across dimensions of a tensor)))`
        of elements over a give axis or a list of axes of a tensor

    Args:
        data: (tvm.tensor.Tensor): Tensor of type float16
        axis: The dimensions to reduce. Could be None(by default), int, list or tuple.
              If None, all dimenstions will be reduced.
              If int or list, must be in the range of [-len(date.shape), len(date.shape)-1]
        keepdims: Boolean. If true, remians reduced dimensions with lengthe 1. False by default

    Returns:
        tvm.tensor.Tensor, has the same shape and type as data.
    """

    check_list = ["float16"]
    dtype = data.dtype
    if not dtype in check_list:
        raise RuntimeError("reduce_logsumexp_cce only support %s while dtype is %s" % (",".join(check_list), dtype))
    shape = [x.value for x in data.shape]
    utils.check_shape(shape)

    exp_ = vexp(data)
    sum_ = sum(exp_, axis=axis, keepdims=keepdims)
    res = vlog(sum_)
    return res
