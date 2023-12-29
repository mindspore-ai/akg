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
from tvm.hybrid import script
    
@script(capture=locals())
def lu(a, b):
    out = output_tensor(b.shape, b.dtype)
    len = a.shape[0]
    for n in range(len):
        i = len - n - 1
        out[i, 0] = b[i, 0] / a[i, i]
        for j in range(i):
            b[j, 0] = b[j, 0] - a[j, i] * out[i, 0]
    return out