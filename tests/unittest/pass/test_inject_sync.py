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
import akg.backend as cce


def check_sync(stmt, from_pip, to_pip, events):
    data = [0, 0, 0, 0]  # next_push_idx, push_pass_cnt, next_pop_idx, push_pass_cnt

    def verify(n):
        if isinstance(n, akg.tvm.expr.Call):
            if n.name == "cce.coproc_dep_push":
                p1 = n.args[0].value
                p2 = n.args[1].value
                e = n.args[2].value
                if p1 == from_pip and p2 == to_pip:
                    if events[data[0]] == e:
                        data[1] += 1
                    data[0] += 1
            elif n.name == "cce.coproc_dep_pop":
                p1 = n.args[0].value
                p2 = n.args[1].value
                e = n.args[2].value
                if p1 == from_pip and p2 == to_pip:
                    if events[data[2]] == e:
                        data[3] += 1
                    data[2] += 1
    akg.tvm.ir_pass.PostOrderVisit(stmt, verify)
    assert(data[1] == len(events))
    assert(data[3] == len(events))


def check_bar(stmt, pip, num):
    data = [0]

    def verify(n):
        if isinstance(n, akg.tvm.expr.Call) and n.name == "cce.coproc_sync":
            if n.args[0].value == pip:
                data[0] += 1
    akg.tvm.ir_pass.PostOrderVisit(stmt, verify)
    assert(data[0] == num)


def test_basic():
    ib = akg.tvm.ir_builder.create()
    cp = akg.tvm.thread_axis("cce")
    A = ib.allocate("float32", 128, name="A")
    B = ib.allocate("float32", 128, name="B")
    C = ib.allocate("float32", 128, name="C")
    D = ib.allocate("float32", 128, name="D")
    with ib.new_scope():
        ib.scope_attr(cp, "coproc_scope", 1)
        A[0] = 10.0
    with ib.new_scope():
        ib.scope_attr(cp, "coproc_scope", 1)
        C[0] = 10.0
    with ib.new_scope():
        ib.scope_attr(cp, "coproc_scope", 2)
        B[0] = A[0]
    with ib.new_scope():
        ib.scope_attr(cp, "coproc_scope", 2)
        D[0] = C[0]
    stmt = ib.get()
    # print(stmt)
    stmt = akg.tvm.ir_pass.InjectSync(stmt)
    # print(stmt)
    check_sync(stmt, 1, 2, (0, 1))


def test_dep_v_to_s():
    ib = akg.tvm.ir_builder.create()
    cp = akg.tvm.thread_axis("cce")
    A = ib.allocate("int8", 32, name="A")
    zero = akg.tvm.const(0, 'int8')
    with ib.new_scope():
        ib.scope_attr(cp, "coproc_scope", 2)
        ib.emit(akg.tvm.call_pure_intrin("float32", "tvm_access_ptr", zero, A, 0, 26, 2))
    with ib.new_scope():
        ib.scope_attr(cp, "coproc_scope", 1)
        A[28] = zero
    stmt = ib.get()
    # print(stmt)
    stmt = akg.tvm.ir_pass.InjectSync(stmt)
    # print(stmt)
    check_sync(stmt, 2, 1, (0,))

def test_innate_sclar_s():
    ib = akg.tvm.ir_builder.create()
    cp = akg.tvm.thread_axis("cce")
    A = ib.allocate("float32", 32, name="A")
    with ib.new_scope():
        ib.scope_attr(cp, "coproc_scope", 1)
        A[0] = 1.0
    with ib.new_scope():
        ib.scope_attr(cp, "coproc_scope", 1)
        A[0] = 1.0
    stmt = ib.get()
    #print(stmt)
    stmt = akg.tvm.ir_pass.InjectSync(stmt)
    #print(stmt)
    check_bar(stmt, 1, 0)

def test_innate_spr():
    ib = akg.tvm.ir_builder.create()
    cp = akg.tvm.thread_axis("cce")
    A = ib.allocate("float32", 32, name="A")
    zero = akg.tvm.const(0, 'int8')
    with ib.new_scope():
        ib.scope_attr(cp, "coproc_scope", 2)
        ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", 0, 2))
    with ib.new_scope():
        ib.scope_attr(cp, "coproc_scope", 2)
        ib.emit(akg.tvm.call_pure_intrin("float32", "tvm_access_ptr", zero, A, 0, 26, 2))
    with ib.new_scope():
        ib.scope_attr(cp, "coproc_scope", 2)
        ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", 0, 3))
    stmt = ib.get()
    #print(stmt)
    stmt = akg.tvm.ir_pass.InjectSync(stmt)
    #print(stmt)
    check_bar(stmt, 2, 0)

def test_innate_mad():
    ib = akg.tvm.ir_builder.create()
    cp = akg.tvm.thread_axis("cce")
    A = ib.allocate("float32", 32, name="A")
    zero = akg.tvm.const(0, 'int8')
    with ib.new_scope():
        ib.scope_attr(cp, "coproc_scope", 3)
        a = akg.tvm.call_pure_intrin("float32", "tvm_access_ptr", zero, A, 0, 26, 2)
        ib.emit(akg.tvm.call_extern("float32", "mad", a, 0, 0, 128, 0, 32, 0))
    with ib.new_scope():
        ib.scope_attr(cp, "coproc_scope", 3)
        a = akg.tvm.call_pure_intrin("float32", "tvm_access_ptr", zero, A, 0, 26, 2)
        ib.emit(akg.tvm.call_extern("float32", "mad", a, 0, 0, 32, 0, 32, 0))
    with ib.new_scope():
        ib.scope_attr(cp, "coproc_scope", 3)
        a = akg.tvm.call_pure_intrin("float32", "tvm_access_ptr", zero, A, 0, 26, 2)
        ib.emit(akg.tvm.call_extern("float32", "mad", a, 0, 0, 64, 0, 64, 0))
    stmt = ib.get()
    #print(stmt)
    stmt = akg.tvm.ir_pass.InjectSync(stmt)
    #print(stmt)
    check_bar(stmt, 3, 1)

if __name__ == '__main__':
    test_basic()
    test_dep_v_to_s()
    test_innate_sclar_s()
    test_innate_spr()
    test_innate_mad()
