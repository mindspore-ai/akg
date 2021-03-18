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

import akg
import akg.lang.cce
import akg.tvm
from akg.utils import kernel_exec as utils
from akg.utils import custom_tiling as ct_util
from akg.utils import validation_check as vc_util

elemwise_sum_ad_set_dim_map = {
    str(([3, 3], "float16")): ([(1, 0)]),
    str(([3, 3], "float32")): ([(1, 0)]),
}


def elemwise_sum_ad_set_dim_func(a, b):
    """setdim function"""
    key = []
    key.append(tuple(a.shape))
    key.append(a.dtype)
    hash_key = str(tuple(key))

    if hash_key in elemwise_sum_ad_set_dim_map.keys():
        return ct_util.set_dims(elemwise_sum_ad_set_dim_map[hash_key]), hash_key
    else:
        return "", hash_key

@vc_util.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor)
def elemwise_sum(a, b):
    """
    Element-wise sum data.

    Args:
        a (tvm.tensor.Tensor): Input `a` of type float16 or float32.
        b (tvm.tensor.Tensor): Input `b` of type float16 or float32.

    Returns:
        tvm.tensor.Tensor, has the same shape and type as inputs.
    """
    vc_util.check_shape(a)
    vc_util.check_shape(b)

    dim_info, _ = elemwise_sum_ad_set_dim_func(a, b)
    attrs = {"dim": dim_info}

    shape = a.shape

    c = akg.tvm.compute(shape, lambda *indices: a(*indices) + b(*indices), name="b")
    return c, attrs


def elemwise_sum_manual_schedule(input_shape, polyhedral=False, attrs=None):
    """manually schedule"""
    b = akg.tvm.placeholder(input_shape, dtype='float16', name="b")
    c = akg.tvm.placeholder(input_shape, dtype='float16', name="c")
    a = akg.tvm.compute(input_shape, lambda *indices: b(*indices) + c(*indices))
    ss = akg.tvm.create_schedule([a.op])
    ss.cache_read(b, "local.UB", [a])
    ss.cache_read(c, "local.UB", [a])
    ss.cache_write(a, "local.UB")
    ss[a].set_scope("local.UB")
    with akg.build_config(add_lower_pass=utils.debug_mode(0), dump_pass_ir=True):
        mod = akg.build(ss,
                    [b, c, a],
                    "cce",
                    name="test_manual_schedule",
                    attrs=attrs,
                    polyhedral=polyhedral)
    return mod