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

"""operator dsl function: where"""

import akg.tvm

def where(A, B, C, D):

    shape_con = [x.value for x in A.shape]
    shape = [x.value for x in C.shape]

    if shape == shape_con:
        res = akg.tvm.compute(shape, lambda *indice: akg.tvm.expr.Select(A[indice] >= B[indice], C[indice], D[indice]))
    else:
        tmpA = akg.tvm.compute(shape, lambda *indice: A[indice[0]], name='tmpA')
        tmpB = akg.tvm.compute(shape, lambda *indice: B[indice[0]], name='tmpB')
        res = akg.tvm.compute(shape, lambda *indice: akg.tvm.expr.Select(tmpA[indice] >= tmpB[indice], C[indice], D[indice]), name='res')

    return res
