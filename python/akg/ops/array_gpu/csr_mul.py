# Copyright 2021 Huawei Technologies Co., Ltd
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

"""operator dsl function: CSRMul"""
import akg.tvm
from akg.utils import validation_check as vc_util
from ...composite import csr_mul as cuda_csr_mul

@vc_util.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor,
                          akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, tuple)
def csr_mul(dense, sparse_data, col_idx, row_idx, shape):
    attrs = {"dense_shape": shape}
    return cuda_csr_mul((row_idx, col_idx, sparse_data, dense), attrs)
