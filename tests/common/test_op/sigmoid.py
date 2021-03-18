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

"""operator dsl function: sigmoid"""
from akg.lang.cce import vrec, vadds, vexp, vmuls
from akg.utils import validation_check as vc_util


def sigmoid(data):
    """
    Computes sigmoid of x element-wise.
    \f[
        y = \frac{1}{e^{-x} + 1}
    \f]

    Args:
        data (tvm.tensor.Tensor): Tensor of type float16, float32.

    Returns:
        tvm.tensor.Tensor, has same type and shape as data.

    """

    check_list = ["float16", "float32"]
    dtype = data.dtype
    if not dtype in check_list:
        raise RuntimeError("sigmoid_cce only support %s while dtype is %s" % (",".join(check_list), dtype))
    shape = data.shape
    vc_util.check_shape(shape)

    res = vrec(vadds(vexp(vmuls(data, -1.0)), 1.0))

    return res
