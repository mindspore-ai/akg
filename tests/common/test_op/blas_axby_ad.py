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

"""operator dsl function: blas_axby_ad"""
import akg.tvm
import akg
from test_op import blas_axby

def blas_axby_ad(head, alpha, beta):
    """Compute gradient of blas_axby operator using automatic differentiate."""
    x = akg.tvm.placeholder(head.shape, head.dtype, "inputx")
    y = akg.tvm.placeholder(head.shape, head.dtype, "inputy")
    op = blas_axby.blas_axby(x, y, alpha, beta)
    jacs = list(akg.differentiate(op, [x, y], head))
    return jacs[0], jacs[1]
