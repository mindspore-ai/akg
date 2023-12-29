# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

"""batch matmul"""
import akg
import numpy as np
import akg.topi as topi
import akg.tvm as tvm
import akg.utils as utils
from .batch_matmul_orig import batch_matmul
from .tensorcore_batch_matmul import batch_matmul as tc_batch_matmul

def BatchMatMul(x, y, bias, out_dtype="float32", layout1="NHDT", layout2="NHDT", layout_out="NHDT", tensor_core=True,
                add_bias=False):
    """
    BatchMatmul with auto poly.

    Supported Platforms:
        'GPU'
    """

    if add_bias == False:
        bias = None
    if tensor_core == True:
        attrs = [bias, out_dtype, layout1, layout2, layout_out]
        return tc_batch_matmul(x, y, attrs)
    else:
        return batch_matmul(x, y, bias, layout1, layout2, layout_out)