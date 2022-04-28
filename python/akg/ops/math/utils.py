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

"""dsl common functions"""

import akg.tvm
import akg.topi
import akg.utils as utils
from akg.utils.dsl_create import produce_shapes

def make_input_and_value(data1, data2):
    shape1 = [x.value for x in data1.shape]
    shape2 = [x.value for x in data2.shape]
    utils.check_shape(shape1)
    utils.check_shape(shape2)

    shape1, shape2, shape = produce_shapes(shape1, shape2)

    utils.elemwise_dtype_check(data1.dtype, data2.dtype)
    dtype = data1.dtype

    t_value = akg.tvm.compute(shape, lambda *indice: akg.tvm.const(1, dtype), "T")
    f_value = akg.tvm.compute(shape, lambda *indice: akg.tvm.const(0, dtype), "F")
    
    input1_bro = akg.topi.broadcast_to(data1, shape)
    input2_bro = akg.topi.broadcast_to(data2, shape)
    res = (t_value, f_value, input1_bro, input2_bro, shape)
    return res