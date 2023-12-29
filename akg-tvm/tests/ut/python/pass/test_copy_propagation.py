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

import akg.tvm
from akg.tvm import build_module


def test_copy_prop_case0():
    '''
     B = A
     C = B * 2

     ==>

     C = A * 2
    '''

    shape = (8, 128)
    dtype = "float32"
    A = akg.tvm.placeholder(shape, name="input", dtype=dtype)

    B = akg.tvm.compute(shape, lambda *indices: A(*indices), name="B")
    B1 = akg.tvm.compute(shape, lambda *indices: B(*indices), name="B1")
    C = akg.tvm.compute(shape, lambda *indices: B1(*indices) * akg.tvm.const(2.0, dtype), name="C")

    s = akg.tvm.create_schedule(C.op)

    args = [A, C]
    binds, _ = build_module.get_binds(args)
    bounds = akg.tvm.schedule.InferBound(s)
    stmt = akg.tvm.schedule.ScheduleOps(s, bounds)

    stmt = akg.tvm.ir_pass.CopyPropagation(stmt, binds)

    success = [True, True, True]

    def verify(n):
        if isinstance(n, akg.tvm.stmt.Realize):
            if n.func.name == "B1":
                success[0] = False
        if isinstance(n, akg.tvm.stmt.AttrStmt):
            if n.node.name == "B1":
                success[1] = False
        if isinstance(n, akg.tvm.expr.Call):
            if n.name == "B1":
                success[2] = False

    akg.tvm.ir_pass.PostOrderVisit(stmt, verify)

    assert(success[0] == True)
    assert(success[1] == True)
    assert(success[2] == True)


def test_copy_prop_case1():
    '''
     B = A
     C = B * 2
     B = C

     == >

     C = A * 2
     B = C
    '''

    shape = (8, 128)
    dtype = "float32"
    A = akg.tvm.placeholder(shape, name="input", dtype=dtype)

    B = akg.tvm.compute(shape, lambda *indices: A(*indices), name="B")
    B1 = akg.tvm.compute(shape, lambda *indices: B(*indices), name="B1")
    C = akg.tvm.compute(shape, lambda *indices: B1(*indices) * akg.tvm.const(2.0, dtype), name="C")
    B = akg.tvm.compute(shape, lambda *indices: C(*indices), name='B')

    s = akg.tvm.create_schedule(B.op)
    args = [A, C]
    binds, _ = build_module.get_binds(args)
    bounds = akg.tvm.schedule.InferBound(s)
    stmt = akg.tvm.schedule.ScheduleOps(s, bounds)

    stmt = akg.tvm.ir_pass.CopyPropagation(stmt, binds)

    success = [True, True, True]

    def verify(n):
        if isinstance(n, akg.tvm.stmt.Realize):
            if n.func.name == "B1":
                success[0] = False
        if isinstance(n, akg.tvm.stmt.AttrStmt):
            if n.node.name == "B1":
                success[1] = False
        if isinstance(n, akg.tvm.expr.Call):
            if n.name == "B1":
                success[2] = False

    akg.tvm.ir_pass.PostOrderVisit(stmt, verify)

    assert(success[0] == True)
    assert(success[1] == True)
    assert(success[2] == True)


if __name__ == '__main__':
    test_copy_prop_case0()
    test_copy_prop_case1()
