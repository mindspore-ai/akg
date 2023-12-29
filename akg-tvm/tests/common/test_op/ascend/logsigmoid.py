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

"""operator dsl function: logsigmoid"""

from akg.lang.ascend import vrec, vadds, vexp, vmuls, vlog
import akg.utils as utils


def logsigmoid(data, target="cce"):
    """
    Compute logsigmoid of x element-wise.
    \f[
        y = \flog({1}/{e^{x} + 1})
    \f]

    Args:
        data: (tvm.tensor.Tensor): Tensor of type float16

    Returns:
        tvm.tensor.Tensor, has the same shape and type as data.
    """

    check_list = ["float16"]
    dtype = data.dtype
    if not dtype in check_list:
        raise RuntimeError("logsigmoid_cce only support %s while dtype is %s" % (",".join(check_list), dtype))
    shape = [x.value for x in data.shape]
    utils.check_shape(shape)

    res = vlog(vrec(vadds(vexp(vmuls(data, -1.0)), 1.0)))
    return res
