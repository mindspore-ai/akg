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
import numpy as np


@script(capture=locals())
def hpl_lu(a):
    out_0 = allocate(a.shape, a.dtype, "local")
    out_1 = allocate(a.shape, a.dtype, "local")
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if j > i:
                a[j, i] = a[j, i] / a[i, i]
        for k in range(a.shape[0]):
            for l in range(a.shape[1]):
                if k > i and l > i:
                    out_0[k, l] = a[k, i]
                    out_1[k, l] = out_0[k, l] * a[i, l]
                    a[k, l] = a[k, l] - out_1[k, l]
    return a
