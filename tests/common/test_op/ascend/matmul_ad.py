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

"""operator dsl function: matmul_ad"""
import akg.tvm
import akg
from akg.utils.kernel_exec import debug_mode

def matmul_ad(data_shape, weight_shape, dtype, attrs=None):
    check_list = ["float16"]
    if not (dtype.lower() in check_list):
        raise RuntimeError("matmul test only support %s while dtype is %s" % (",".join(check_list), dtype))
    # check_shape(shape)
    assert(len(data_shape) == 2)
    assert(len(weight_shape) == 2)
    assert(data_shape[1] == weight_shape[0])

    m, k = data_shape
    _, n = weight_shape

    a = akg.tvm.placeholder((m, k), name='a', dtype=dtype)
    b = akg.tvm.placeholder((k, n), name='b', dtype=dtype)
    kk = akg.tvm.reduce_axis((0, k), name='kk')
    c = akg.tvm.compute((m, n), lambda i, j: akg.lang.ascend.mmad(a[i, kk] * b[kk, j], axis=kk), name="c")


    head = akg.tvm.placeholder(c.shape, name="Head", dtype='float16')
    _jacs = list(akg.differentiate(c, [a], head))
    sjac = akg.tvm.create_schedule([_jacs[0].op])
    op_vars = [head, b, _jacs[0]]

    with akg.build_config(add_lower_pass=debug_mode(0), dump_pass_ir=True):
        mod = akg.build(sjac, op_vars, "cce", name="test2", attrs=attrs, polyhedral=True)
        return mod
