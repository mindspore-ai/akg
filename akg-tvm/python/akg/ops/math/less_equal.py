# Copyright 2020-2022 Huawei Technologies Co., Ltd
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

"""operator dsl function: lessequal"""
import akg.tvm
import akg.topi
import akg.utils as utils
from akg.utils.kernel_exec import product_is_mini
from akg.utils.dsl_create import produce_shapes
from .sub import sub
from .cast import cast
from .utils import make_input_and_value


@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor)
def _less_equal(data1, data2):
    t_value, f_value, input1_bro, input2_bro, shape = make_input_and_value(data1, data2)
    c_out = akg.tvm.compute(shape, lambda *indice: akg.tvm.expr.Select(input1_bro[indice] <= input2_bro[indice],
                                                                       t_value[indice], f_value[indice]), name="C")
    res = akg.tvm.compute(shape, lambda *indice: c_out(*indice).astype("bool"), name="res")

    return res


def _less_equal_ascend(data1, data2, target=utils.CCE):
    # check shapes
    shape1 = [x.value for x in data1.shape]
    utils.check_shape(shape1)
    shape2 = [x.value for x in data2.shape]
    utils.check_shape(shape2)

    # check types
    if data1.dtype != data2.dtype:
        raise TypeError("data1 is of type %s, data2 is of type %s, which are different." % (data1.dtype, data2.dtype))

    check_list = ["float16", "float32", "int32"]
    dtype = data1.dtype
    orig_dtype = data1.dtype
    if not dtype in check_list:
        raise TypeError("less_equal only support %s while dtype is %s" % (",".join(check_list), dtype))

    if product_is_mini():
        if dtype is not "float16":
            dtype = "float16"
    else:
        if dtype not in ["float32", "float16"]:
            dtype = "float32"

    if orig_dtype == "float32" and dtype == "float16":
        data_sub = sub(data1, data2, target)
        data_sub = akg.topi.cast(data_sub, dtype)
        zero = akg.tvm.const(0.0, dtype)
        res = akg.topi.less_equal(data_sub, zero)
    else:
        data1 = akg.topi.cast(data1, dtype)
        data2 = akg.topi.cast(data2, dtype)
        res = akg.topi.less_equal(data1, data2)

    return res


def less_equal(data1, data2, target=utils.CCE):
    """
    Check whether input1 lessequals to input2.

    Args:
        input1 (tvm.tensor.Tensor): Tensor.
        input2 (tvm.tensor.Tensor): Tensor.

    Returns:
        tvm.tensor.Tensor. If input1 lessequal to input2 return True, else return False.

    Supported Platforms:
        'Ascend', 'GPU', 'CPU'
    """
    utils.check_supported_target(target)
    if target == utils.CCE:
        return _less_equal_ascend(data1, data2)
    else:
        return _less_equal(data1, data2)
