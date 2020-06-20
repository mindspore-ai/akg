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

"""operator dsl function:gelu"""

from akg import backend as cce
import akg
import akg.tvm


def test_vmadd():
    shape = (10, 256)
    dtype = 'float16'

    x = akg.tvm.placeholder(shape, name="x", dtype=dtype)

    def compute_func(*indices):
        y = x(*indices) + akg.tvm.const(2.0, dtype)
        return y * x(*indices) + x(*indices) + akg.tvm.const(1.0, dtype)
    res = akg.tvm.compute(shape, compute_func)

    s = akg.tvm.create_schedule(res.op)

    # build the cce kernel
    with akg.build_config(add_lower_pass=cce.debug_mode(0), dump_pass_ir=True):
        mod = akg.build(s, [x, res], "cce", polyhedral=True)

    assert "vmadd" in mod.imported_modules[0].get_source()


def test_vmaddrelu():
    shape = (10, 256)
    dtype = 'float16'

    x = akg.tvm.placeholder(shape, name="x", dtype=dtype)

    def compute_func(*indices):
        y = x(*indices) + akg.tvm.const(2.0, dtype)
        return akg.tvm.max(akg.tvm.const(0.0, dtype), y * x(*indices) + x(*indices)) + akg.tvm.const(1.0, dtype)
    res = akg.tvm.compute(shape, compute_func)

    s = akg.tvm.create_schedule(res.op)

    # build the cce kernel
    with akg.build_config(add_lower_pass=cce.debug_mode(0), dump_pass_ir=True):
        mod = akg.build(s, [x, res], "cce", polyhedral=True)

    assert "vmaddrelu" in mod.imported_modules[0].get_source()


def test_vaxpy():
    shape = (10, 256)
    dtype = 'float16'

    x = akg.tvm.placeholder(shape, name="x", dtype=dtype)

    def compute_func(*indices):
        y = x(*indices) + akg.tvm.const(2.0, dtype)
        return y + akg.tvm.const(3.0, dtype) * x(*indices) + akg.tvm.const(1.0, dtype)
    res = akg.tvm.compute(shape, compute_func)

    s = akg.tvm.create_schedule(res.op)

    # build the cce kernel
    with akg.build_config(add_lower_pass=cce.debug_mode(0), dump_pass_ir=True):
        mod = akg.build(s, [x, res], "cce", polyhedral=True)

    assert "vaxpy" in mod.imported_modules[0].get_source()


def test_select():
    N = 128

    actual = akg.tvm.placeholder((N,), name='actual', dtype='int32')
    predict = akg.tvm.placeholder((N,), name='predict', dtype='int32')
    k = akg.tvm.reduce_axis((0, N), name='k')
    output = akg.tvm.compute((N, N), lambda i, j: akg.tvm.sum(akg.tvm.expr.Select(akg.tvm.all(i == actual[k], j == predict[k]), 1.0, 0.0), axis=k))

    s = akg.tvm.create_schedule(output.op)

    # build the cce kernel
    with akg.build_config(add_lower_pass=cce.debug_mode(0), dump_pass_ir=True):
        mod = akg.build(s, [actual, predict, output], "cce", polyhedral=True)


if __name__ == "__main__":
    test_vmadd()
    test_vmaddrelu()
    test_vaxpy()
    test_select()
