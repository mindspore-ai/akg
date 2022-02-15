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

from akg.tvm.hybrid import script
import akg.utils as utils
from akg.utils.dsl_create import TensorUtils


def hpl_trsm(a, b):
    attrs = {"RewriteVarTensorIdx": True}

    @script
    def func(a, b):
        inverse_0 = allocate(b.shape, b.dtype, "local")
        row = b.shape[0]
        col = b.shape[1]
        for l in range(col // 16):
            for i in range(row):
                for j in range(i):
                    for k in range(16):
                        inverse_0[i, l*16+k] = a[i, j] * b[j, l*16+k]
                        b[i, l*16+k] = b[i, l*16+k] - inverse_0[i, l*16+k]
                for k in range(16):
                    b[i, l*16+k] = b[i, l*16+k] / a[i, i]
        return b

    out = func(a, b)
    out, binds_info = TensorUtils.inplace_set(b, out)
    attrs[utils.BINDS] = binds_info
    return out, attrs
