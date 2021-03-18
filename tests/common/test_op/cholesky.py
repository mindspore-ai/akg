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

"""dsl: cholesky"""
import akg.tvm
from akg.utils import validation_check as vc_util


@akg.tvm.hybrid.script
def chol(inputs):
    dim = inputs.shape[0]

    for i in range(dim):
        for j in range(dim):
            res[i, j] = inputs[i, j] * exp(log(inputs[i, i]) * (-0.5))
        if i != (dim - 1):
            for j in range(dim - i - 1):
                for k in range(dim):
                    res[j + i + 1, k] = res[j + i + 1, k] - res[i, j + i + 1] * res[i, k]
    return res


def cholesky(inputs):
    dtype = inputs.dtype
    check_list = ["float32"]
    if not (dtype.lower() in check_list):
        raise RuntimeError("cholesky only support %s while dtype is %s" % (",".join(check_list), dtype))
    shape = inputs.shape
    vc_util.check_shape(shape)
    l_res = chol(inputs)

    return l_res
