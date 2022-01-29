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

@script(capture=locals())
def hpl_cholesky(a):
    w = a.shape[0]
    out = output_tensor(a.shape, a.dtype)
    tmp = allocate((a.shape[0],), a.dtype, "local")
    tmp_0 = allocate((a.shape[0],), a.dtype, "local")
    tmp_1 = allocate((a.shape[0],), a.dtype, "local")
    out_0 = allocate(a.shape, a.dtype, "local")
    out_1 = allocate(a.shape, a.dtype, "local")
    for i in range(w):
        for j in range(w):
            tmp_0[j] = out[i, i]
            tmp_1[j] = sqrt(tmp_0[j])
            tmp[j] = out[i, j] / tmp_1[j]
        for j in range(w):
            if j >= i:
                out[i, j] = tmp[j]
            else:
                out[i, j] = float16(0.0)
        for k in range(a.shape[0]):
            for l in range(a.shape[1]):
                if k > i and l > i:
                    out_0[k, l] = out[i, k]
                    out_1[k, l] = out_0[k, l] * out[i, l]
                    out[k, l] = out[k, l] - out_1[k, l]
    return out
