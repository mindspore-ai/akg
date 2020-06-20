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

from akg.backend import cce_runtime
import akg.tvm


def check_result(stmt, seq):
    ''' seq example : [1, 2, 3, "s", 4, 5, "e", 6] '''
    loop_stack = []
    seq_index = [0]

    def verify(op):
        i = seq_index[0]
        while i < len(seq) and seq[i] == 's':
            seq_index[0] += 1
            i = seq_index[0]
            assert i < len(seq)
            first_v, loop_cnt = None, 1
            for v in range(i, len(seq)):
                if seq[v] == 's':
                    loop_cnt += 1
                elif seq[v] == 'e':
                    loop_cnt -= 1
                    if loop_cnt == 0:
                        break
                else:
                    first_v = seq[v]
                    break
            loop_stack.append(first_v)
        if isinstance(op, akg.tvm.expr.Call) and op.name == 'set_vector_mask':
            assert op.args[1].value == seq[i]
            seq_index[0] += 1
        elif isinstance(op, akg.tvm.stmt.For):
            assert seq[i] == 'e'
            v = loop_stack.pop()
            first = [v, None]

            def verify_first(op):
                if first[1] == None and isinstance(op, akg.tvm.expr.Call) and op.name == 'set_vector_mask':
                    first[1] = op.args[1].value
            akg.tvm.ir_pass.PostOrderVisit(op, verify_first)
            assert first[0] == first[1]
            seq_index[0] += 1
    akg.tvm.ir_pass.PostOrderVisit(stmt, verify)
    assert seq_index[0] == len(seq)


def test_elim_1():
    ''' elim reduplicated mask after elim pend '''
    ib = akg.tvm.ir_builder.create()
    cp = akg.tvm.thread_axis("cce")
    A = ib.allocate("float32", 128, name="A", scope="local")
    ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x1, "uint64")))
    ib.emit(akg.tvm.call_extern("float32", "vadd", A, A, 0))
    ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x2, "uint64")))
    ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x1, "uint64")))
    ib.emit(akg.tvm.call_extern("float32", "vadd", A, A, 0))
    stmt = ib.get()
    # print(stmt)
    stmt = akg.tvm.ir_pass.ElimVectorMask(stmt)
    # print(stmt)
    check_result(stmt, [1])


def test_elim_2():
    ''' elim repeat mask  '''
    ib = akg.tvm.ir_builder.create()
    cp = akg.tvm.thread_axis("cce")
    A = ib.allocate("float32", 128, name="A", scope="local")
    ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x1, "uint64")))
    ib.emit(akg.tvm.call_extern("float32", "vadd", A, A, 0))
    ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x1, "uint64")))
    ib.emit(akg.tvm.call_extern("float32", "vadd", A, A, 0))
    stmt = ib.get()
    # print(stmt)
    stmt = akg.tvm.ir_pass.ElimVectorMask(stmt)
    # print(stmt)
    check_result(stmt, [1])


def test_hoist_1():
    ''' vec + vm in loop '''
    ib = akg.tvm.ir_builder.create()
    cp = akg.tvm.thread_axis("cce")
    A = ib.allocate("float32", 128, name="A", scope="local")
    with ib.for_range(0, 5, 'i') as i:
        ib.emit(akg.tvm.call_extern("float32", "vadd", A, A, i))
        ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x2, "uint64")))
    stmt = ib.get()
    # print(stmt)
    stmt = akg.tvm.ir_pass.ElimVectorMask(stmt)
    # print(stmt)
    check_result(stmt, ['s', 'e'])


def test_hoist_2():
    ''' vec + vm + vec in loop '''
    ib = akg.tvm.ir_builder.create()
    cp = akg.tvm.thread_axis("cce")
    ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x1, "uint64")))
    A = ib.allocate("float32", 128, name="A", scope="local")
    with ib.for_range(0, 5, 'i') as i:
        ib.emit(akg.tvm.call_extern("float32", "vadd", A, A, i))
        ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x2, "uint64")))
        ib.emit(akg.tvm.call_extern("float32", "vadd", A, A, i))
    stmt = ib.get()
    # print(stmt)
    stmt = akg.tvm.ir_pass.ElimVectorMask(stmt)
    # print(stmt)
    check_result(stmt, [1, 's', 2, 'e'])


def test_hoist_3():
    ''' vm + vec in loop '''
    ib = akg.tvm.ir_builder.create()
    cp = akg.tvm.thread_axis("cce")
    ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x1, "uint64")))
    A = ib.allocate("float32", 128, name="A", scope="local")
    with ib.for_range(0, 5, 'i') as i:
        ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x2, "uint64")))
        ib.emit(akg.tvm.call_extern("float32", "vadd", A, A, i))
    stmt = ib.get()
    # print(stmt)
    stmt = akg.tvm.ir_pass.ElimVectorMask(stmt)
    # print(stmt)
    check_result(stmt, [2, 's', 'e'])


def test_hoist_4():
    ''' only vm in loop '''
    ib = akg.tvm.ir_builder.create()
    cp = akg.tvm.thread_axis("cce")
    ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x1, "uint64")))
    A = ib.allocate("float32", 128, name="A", scope="local")
    with ib.for_range(0, 5, 'i') as i:
        ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x2, "uint64")))
    stmt = ib.get()
    # print(stmt)
    stmt = akg.tvm.ir_pass.ElimVectorMask(stmt)
    # print(stmt)
    check_result(stmt, [])


def test_hoist_5():
    ''' vm + vec + vm in loop '''
    ib = akg.tvm.ir_builder.create()
    cp = akg.tvm.thread_axis("cce")
    ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x1, "uint64")))
    A = ib.allocate("float32", 128, name="A", scope="local")
    with ib.for_range(0, 5, 'i') as i:
        ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x2, "uint64")))
        ib.emit(akg.tvm.call_extern("float32", "vadd", A, A, i))
        ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x3, "uint64")))
    stmt = ib.get()
    # print(stmt)
    stmt = akg.tvm.ir_pass.ElimVectorMask(stmt)
    # print(stmt)
    check_result(stmt, [2, 's', 'e'])


def test_hoist_6():
    ''' if + vm in loop, prevent prev-hoist'''
    ib = akg.tvm.ir_builder.create()
    cp = akg.tvm.thread_axis("cce")
    ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x1, "uint64")))
    A = ib.allocate("float32", 128, name="A", scope="local")
    with ib.for_range(0, 5, 'i') as i:
        with ib.if_scope(0):
            ib.emit(akg.tvm.call_extern("float32", "vadd", A, A, i))
        ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x2, "uint64")))
        ib.emit(akg.tvm.call_extern("float32", "vadd", A, A, i))
    stmt = ib.get()
    # print(stmt)
    stmt = akg.tvm.ir_pass.ElimVectorMask(stmt)
    # print(stmt)
    check_result(stmt, [1, 's', 2, 'e'])


def test_hoist_7():
    ''' vm + if  in loop, prevent post-hoist'''
    ib = akg.tvm.ir_builder.create()
    cp = akg.tvm.thread_axis("cce")
    ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x1, "uint64")))
    A = ib.allocate("float32", 128, name="A", scope="local")
    with ib.for_range(0, 5, 'i') as i:
        ib.emit(akg.tvm.call_extern("float32", "vadd", A, A, i))
        ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x2, "uint64")))
        with ib.if_scope(0):
            ib.emit(akg.tvm.call_extern("float32", "vadd", A, A, i))
    stmt = ib.get()
    # print(stmt)
    stmt = akg.tvm.ir_pass.ElimVectorMask(stmt)
    # print(stmt)
    check_result(stmt, [1, 's', 2, 'e'])


def test_hoist_8():
    ''' mulit-loop hoist '''
    ib = akg.tvm.ir_builder.create()
    cp = akg.tvm.thread_axis("cce")
    ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x1, "uint64")))
    A = ib.allocate("float32", 128, name="A", scope="local")
    with ib.for_range(0, 5, 'i') as i:
        with ib.for_range(0, 5, 'j') as i:
            ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x2, "uint64")))
            ib.emit(akg.tvm.call_extern("float32", "vadd", A, A, i))
            ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x3, "uint64")))
    stmt = ib.get()
    # print(stmt)
    stmt = akg.tvm.ir_pass.ElimVectorMask(stmt)
    # print(stmt)
    check_result(stmt, [2, 's', 's', 'e', 'e'])


def test_hoist_9():
    ''' post hoist, rm-pend '''
    ib = akg.tvm.ir_builder.create()
    cp = akg.tvm.thread_axis("cce")
    ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x1, "uint64")))
    A = ib.allocate("float32", 128, name="A", scope="local")
    with ib.for_range(0, 5, 'i') as i:
        ib.emit(akg.tvm.call_extern("float32", "vadd", A, A, i))
        ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x2, "uint64")))
    ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x3, "uint64")))
    ib.emit(akg.tvm.call_extern("float32", "vadd", A, A, 0))
    stmt = ib.get()
    # print(stmt)
    stmt = akg.tvm.ir_pass.ElimVectorMask(stmt)
    # print(stmt)
    check_result(stmt, [1, 's', 'e', 3])


def test_hoist_10():
    ''' state same after hoist '''
    ib = akg.tvm.ir_builder.create()
    cp = akg.tvm.thread_axis("cce")
    ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x2, "uint64")))
    A = ib.allocate("float32", 128, name="A", scope="local")
    ib.emit(akg.tvm.call_extern("float32", "vadd", A, A, 0))
    with ib.for_range(0, 5, 'i') as i:
        ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x2, "uint64")))
        ib.emit(akg.tvm.call_extern("float32", "vadd", A, A, i))
    stmt = ib.get()
    # print(stmt)
    stmt = akg.tvm.ir_pass.ElimVectorMask(stmt)
    # print(stmt)
    check_result(stmt, [2, 's', 'e'])


def test_hoist_11():
    ''' recover dup for coherent '''
    ib = akg.tvm.ir_builder.create()
    cp = akg.tvm.thread_axis("cce")
    ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x2, "uint64")))
    A = ib.allocate("float32", 128, name="A", scope="local")
    ib.emit(akg.tvm.call_extern("float32", "vadd", A, A, 0))
    with ib.for_range(0, 5, 'i') as i:
        ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x2, "uint64")))
        ib.emit(akg.tvm.call_extern("float32", "vadd", A, A, i))
        ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x3, "uint64")))
        ib.emit(akg.tvm.call_extern("float32", "vadd", A, A, i))
    stmt = ib.get()
    # print(stmt)
    stmt = akg.tvm.ir_pass.ElimVectorMask(stmt)
    # print(stmt)
    check_result(stmt, [2, 's', 2, 3, 'e'])


def test_hoist_12():
    ''' covert pend to mask for coherent '''
    ib = akg.tvm.ir_builder.create()
    cp = akg.tvm.thread_axis("cce")
    ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x2, "uint64")))
    A = ib.allocate("float32", 128, name="A", scope="local")
    ib.emit(akg.tvm.call_extern("float32", "vadd", A, A, 0))
    with ib.for_range(0, 5, 'i') as i:
        ib.emit(akg.tvm.call_extern("float32", "vadd", A, A, i))
        ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x3, "uint64")))
        ib.emit(akg.tvm.call_extern("float32", "vadd", A, A, i))
        ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x2, "uint64")))
    stmt = ib.get()
    # print(stmt)
    stmt = akg.tvm.ir_pass.ElimVectorMask(stmt)
    # print(stmt)
    check_result(stmt, [2, 's', 3, 2, 'e'])


def test_hoist_13():
    ''' cur_mask coherent to entry after pend hoist '''
    ib = akg.tvm.ir_builder.create()
    cp = akg.tvm.thread_axis("cce")
    ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x2, "uint64")))
    A = ib.allocate("float32", 128, name="A", scope="local")
    ib.emit(akg.tvm.call_extern("float32", "vadd", A, A, 0))
    with ib.for_range(0, 5, 'i') as i:
        ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x2, "uint64")))
        ib.emit(akg.tvm.call_extern("float32", "vadd", A))
        ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x3, "uint64")))
        ib.emit(akg.tvm.call_extern("float32", "vadd", A))
        ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x2, "uint64")))
        ib.emit(akg.tvm.call_extern("float32", "vadd", A))
        ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x3, "uint64")))
    ib.emit(akg.tvm.call_extern("float32", "vadd", A))
    stmt = ib.get()
    # print(stmt)
    stmt = akg.tvm.ir_pass.ElimVectorMask(stmt)
    # print(stmt)
    check_result(stmt, [2, 's', 3, 2, 'e', 3])


def test_hoist_14():
    ''' mask coherent to entry '''
    ib = akg.tvm.ir_builder.create()
    cp = akg.tvm.thread_axis("cce")
    ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x2, "uint64")))
    with ib.for_range(0, 5, 'i') as i:
        ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x2, "uint64")))
        ib.emit(akg.tvm.call_extern("float32", "vadd", 1))
        ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x3, "uint64")))
        ib.emit(akg.tvm.call_extern("float32", "vadd", 2))
        ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x2, "uint64")))
        ib.emit(akg.tvm.call_extern("float32", "vadd", 2))
    stmt = ib.get()
    # print(stmt)
    stmt = akg.tvm.ir_pass.ElimVectorMask(stmt)
    # print(stmt)
    check_result(stmt, [2, 's', 3, 2, 'e'])


def test_hoist_15():
    ''' if as first '''
    ib = akg.tvm.ir_builder.create()
    cp = akg.tvm.thread_axis("cce")
    ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x1, "uint64")))
    ib.emit(akg.tvm.call_extern("float32", "vadd", 0))
    with ib.for_range(0, 5, 'i') as i:
        with ib.if_scope(0):
            ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x1, "uint64")))
            ib.emit(akg.tvm.call_extern("float32", "vadd", 0))
            ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x2, "uint64")))
        ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x2, "uint64")))
        ib.emit(akg.tvm.call_extern("float32", "vadd", 0))
    stmt = ib.get()
    # print(stmt)
    stmt = akg.tvm.ir_pass.ElimVectorMask(stmt)
    # print(stmt)
    check_result(stmt, [1, 's', 1, 2, 'e'])


def test_hoist_16():
    ''' if as first '''
    ib = akg.tvm.ir_builder.create()
    cp = akg.tvm.thread_axis("cce")
    with ib.for_range(0, 5, 'i') as i:
        with ib.for_range(0, 5, 'j') as j:
            ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x1, "uint64")))
            ib.emit(akg.tvm.call_extern("float32", "vadd", 0))
            ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x1, "uint64")))

    with ib.for_range(0, 5, 'i') as i:
        with ib.for_range(0, 5, 'j') as j:
            ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x1, "uint64")))
            ib.emit(akg.tvm.call_extern("float32", "vadd", 0))
            ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x1, "uint64")))
        with ib.for_range(0, 55, 'j') as j:
            ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x2, "uint64")))
            ib.emit(akg.tvm.call_extern("float32", "vadd", 0))
            ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x1, "uint64")))

    stmt = ib.get()
    # print(stmt)
    stmt = akg.tvm.ir_pass.ElimVectorMask(stmt)
    # print(stmt)
    check_result(stmt, [1, 's', 's', 'e', 'e', 's', 's', 'e', 2, 's', 'e', 1, 'e'])

def test_hoist_17():
    ib = akg.tvm.ir_builder.create()
    cp = akg.tvm.thread_axis("cce")
    ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x0, "uint64")))
    with ib.for_range(0, 5, 'i') as i:
        ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x0, "uint64")))
        ib.emit(akg.tvm.call_extern("float32", "vadd", 0))
        ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x1, "uint64")))
        ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x0, "uint64")))
        with ib.for_range(0, 55, 'j') as j:
            ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x1, "uint64")))
            ib.emit(akg.tvm.call_extern("float32", "vadd", 0))
            ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x0, "uint64")))
    ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x1, "uint64")))
    ib.emit(akg.tvm.call_extern("float32", "vadd", 0))

    stmt = ib.get()
    # print(stmt)
    stmt = akg.tvm.ir_pass.ElimVectorMask(stmt)
    # print(stmt)
    check_result(stmt, [0, 's', 1, 's', 'e', 0, 'e', 1])

def test_hoist_18():
    ib = akg.tvm.ir_builder.create()
    cp = akg.tvm.thread_axis("cce")
    ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x1, "uint64")))
    ib.emit(akg.tvm.call_extern("float32", "vadd", 1))
    with ib.for_range(0, 5, 'i') as i:
        with ib.for_range(0, 10, 'j') as j:
            ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x1, "uint64")))
            ib.emit(akg.tvm.call_extern("float32", "vadd", 1))
        ib.emit(akg.tvm.call_extern("float32", "set_vector_mask", akg.tvm.const(0, "uint64"), akg.tvm.const(0x2, "uint64")))
        ib.emit(akg.tvm.call_extern("float32", "vadd", 2))

    stmt = ib.get()
    #print(stmt)
    stmt = akg.tvm.ir_pass.ElimVectorMask(stmt)
    #print(stmt)
    check_result(stmt, [1, 's', 's', 1, 'e', 2, 'e'])

if __name__ == '__main__':
    test_elim_1()
    test_elim_2()
    test_hoist_1()
    test_hoist_2()
    test_hoist_3()
    test_hoist_4()
    test_hoist_5()
    test_hoist_6()
    test_hoist_7()
    test_hoist_8()
    test_hoist_9()
    test_hoist_10()
    test_hoist_11()
    test_hoist_12()
    test_hoist_13()
    test_hoist_14()
    test_hoist_15()
    test_hoist_16()
    test_hoist_17()
    test_hoist_18()
