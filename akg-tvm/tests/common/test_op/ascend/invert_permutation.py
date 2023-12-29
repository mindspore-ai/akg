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

"""invert_permutation"""

import akg.tvm

def invert_permutation(A, target="cce"):
    h = A.shape[0].value
    k = akg.tvm.reduce_axis((0, h))
    k2 = akg.tvm.reduce_axis((0, h))
    # return akg.tvm.compute([h], lambda i: akg.tvm.sum(akg.tvm.expr.Select(A[k] == i, k, 0), axis=k))
    # tmp = akg.tvm.compute([h,h],lambda i,j:tvm.expr.Select(j == A[i],1,0), name="tmp")
    tmp = akg.tvm.compute([h, h, h], lambda k, i, j: akg.tvm.expr.Select(akg.tvm.all(j == A[i], k < i), 1, 0), name="tmp")
    tmp2 = akg.tvm.compute([h, h], lambda i, j: akg.tvm.sum(tmp[k][i][j], axis=k), name="tmp2")
    return akg.tvm.compute([h], lambda i: akg.tvm.sum(tmp2[k2][i], axis=k2))
