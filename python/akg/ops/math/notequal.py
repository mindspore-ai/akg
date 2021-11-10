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

"""operator dsl function: notequal"""
import akg.tvm
import akg.topi
import akg.utils as utils
from akg.utils.dsl_create import produce_shapes

@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor)
def _not_equal(data1, data2):
    shape1 = [x.value for x in data1.shape]
    shape2 = [x.value for x in data2.shape]
    utils.check_shape(shape1)
    utils.check_shape(shape2)

    shape1, shape2, shape = produce_shapes(shape1, shape2)

    utils.elemwise_dtype_check(data1.dtype, data2.dtype)
    dtype = data1.dtype

    # get notequal compute
    t_value = akg.tvm.compute(shape, lambda *indice: akg.tvm.const(1, dtype), "T")
    f_value = akg.tvm.compute(shape, lambda *indice: akg.tvm.const(0, dtype), "F")

    input1_bro = akg.topi.broadcast_to(data1, shape)
    input2_bro = akg.topi.broadcast_to(data2, shape)
    c_out = akg.tvm.compute(shape, lambda *indice: akg.tvm.expr.Select(input1_bro[indice] != input2_bro[indice],
                                                                         t_value[indice], f_value[indice]), name="C")
    res = akg.tvm.compute(shape, lambda *indice: c_out(*indice).astype("bool"), name="res")

    return res

def _not_equal_ascend(data1, data2):
    # check shapes
    utils.check_shape(data1.shape)
    utils.check_shape(data2.shape)

    # check types
    check_list = ["float16"]
    if not (data1.dtype.lower() in check_list):
        raise RuntimeError("not_equal only support %s while dtype is %s" % (",".join(check_list), data1.dtype))
    if not (data2.dtype.lower() in check_list):
        raise RuntimeError("not_equal only support %s while dtype is %s" % (",".join(check_list), data2.dtype))

    res = akg.topi.not_equal(data1, data2)
    return res

def NotEqual(data1, data2, target=utils.CCE):
    """
    check whether data1 notequals to data2.

    Args:
        data1 (tvm.tensor.Tensor): Tensor.
        data2 (tvm.tensor.Tensor): Tensor.

    Returns:
        tvm.tensor.Tensor. If data1 notequal to data2 return True, else return False.
    
    Supported Platforms:
        'Ascend', 'GPU', 'CPU'
    """
    utils.check_supported_target(target)
    if target == utils.CCE:
        return _not_equal_ascend(data1, data2)
    else:
        return _not_equal(data1, data2)