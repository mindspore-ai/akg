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


def test_single_axis_001():
    ''' integer split by blocks '''
    ib = akg.tvm.ir_builder.create()
    zero = akg.tvm.const(0, "int32")
    A = ib.pointer("float32", name='A')
    with ib.for_range(0, 100, 'i') as i:
        with ib.new_scope():
            ib.scope_attr(zero, "pragma_emit_insn", "dma_copy")
            with ib.for_range(0, 1024, 'j') as j:
                A[i * 1024 + j] = akg.tvm.const(1, A.dtype)
    stmt = ib.get()
    # print(stmt)
    stmt = akg.tvm.ir_pass.InjectMultiCore(stmt, 4, 0, False)
    # print(stmt)
    assert(stmt.attr_key == "thread_extent")
    assert(stmt.value.value == 4)
    assert(stmt.body.extent.value == 25)


def test_single_axis_002():
    ''' non-integer split by blocks '''
    ib = akg.tvm.ir_builder.create()
    zero = akg.tvm.const(0, "int32")
    A = ib.pointer("float32", name='A')
    with ib.for_range(0, 99, 'i') as i:
        with ib.new_scope():
            ib.scope_attr(zero, "pragma_emit_insn", "dma_copy")
            with ib.for_range(0, 1024, 'j') as j:
                A[i * 1024 + j] = akg.tvm.const(1, A.dtype)
    stmt = ib.get()
    # print(stmt)
    stmt = akg.tvm.ir_pass.InjectMultiCore(stmt, 4, 0, False)
    # print(stmt)
    assert(stmt.attr_key == "thread_extent")
    assert(stmt.value.value == 4)
    assert(stmt.body.extent.value == 25)


def test_single_axis_003():
    ''' core number less that max blocks '''
    ib = akg.tvm.ir_builder.create()
    zero = akg.tvm.const(0, "int32")
    A = ib.pointer("float32", name='A')
    with ib.for_range(0, 7, 'i') as i:
        with ib.new_scope():
            ib.scope_attr(zero, "pragma_emit_insn", "dma_copy")
            with ib.for_range(0, 1024, 'j') as j:
                A[i * 1024 + j] = akg.tvm.const(1, A.dtype)
    stmt = ib.get()
    # print(stmt)
    stmt = akg.tvm.ir_pass.InjectMultiCore(stmt, 8, 0, False)
    # print(stmt)
    assert(stmt.attr_key == "thread_extent")
    assert(stmt.value.value == 7)


def test_multi_axis_001():
    ''' integer split by blocks '''
    ib = akg.tvm.ir_builder.create()
    zero = akg.tvm.const(0, "int32")
    A = ib.pointer("float32", name='A')
    with ib.for_range(0, 2, 'i') as i:
        with ib.for_range(0, 8, 'k') as k:
            with ib.new_scope():
                ib.scope_attr(zero, "pragma_emit_insn", "dma_copy")
                with ib.for_range(0, 1024, 'j') as j:
                    A[i * 8192 + k * 1024 + j] = akg.tvm.const(1, A.dtype)
    stmt = ib.get()
    # print(stmt)
    stmt = akg.tvm.ir_pass.InjectMultiCore(stmt, 8, 0, False)
    # print(stmt)
    assert(stmt.attr_key == "thread_extent")
    assert(stmt.value.value == 8)
    assert(stmt.body.extent.value == 2)


def test_multi_axis_002():
    ''' no-integer split by blocks '''
    ib = akg.tvm.ir_builder.create()
    zero = akg.tvm.const(0, "int32")
    A = ib.pointer("float32", name='A')
    with ib.for_range(0, 3, 'i') as i:
        with ib.for_range(0, 6, 'k') as k:
            with ib.new_scope():
                ib.scope_attr(zero, "pragma_emit_insn", "dma_copy")
                with ib.for_range(0, 1024, 'j') as j:
                    A[(i * 6 + k) * 1024 + j] = akg.tvm.const(1, A.dtype)
    stmt = ib.get()
    # print(stmt)
    stmt = akg.tvm.ir_pass.InjectMultiCore(stmt, 8, 0, False)
    # print(stmt)
    assert(stmt.attr_key == "thread_extent")
    assert(stmt.value.value == 6)
    assert(stmt.body.extent.value == 3)


def test_dep_between_core_001():
    ''' w-w dep '''
    ib = akg.tvm.ir_builder.create()
    zero = akg.tvm.const(0, "int32")
    A = ib.pointer("float32", name='A')
    with ib.for_range(0, 2, 'i') as i:
        with ib.for_range(0, 8, 'k') as k:
            with ib.new_scope():
                ib.scope_attr(zero, "pragma_emit_insn", "dma_copy")
                with ib.for_range(0, 7, 'j') as j:
                    A[(i * 8 + k) * 7 + j] = akg.tvm.const(1, A.dtype)
    stmt = ib.get()
    # print(stmt)
    stmt = akg.tvm.ir_pass.InjectMultiCore(stmt, 8, 0, False)
    # print(stmt)
    assert(not isinstance(stmt, akg.tvm.stmt.AttrStmt))


def test_dep_between_core_002():
    ''' r-w dep '''
    ib = akg.tvm.ir_builder.create()
    zero = akg.tvm.const(0, "int32")
    A = ib.pointer("float32", name='A')
    with ib.for_range(0, 2, 'i') as i:
        with ib.for_range(0, 8, 'k') as k:
            AL = ib.allocate("float32", 32, name="AL", scope="local")
            with ib.new_scope():
                ib.scope_attr(zero, "pragma_emit_insn", "dma_copy")
                with ib.for_range(0, 32, 'j') as j:
                    AL[j] = A[(i * 8) * 32 + j]
            with ib.new_scope():
                ib.scope_attr(zero, "pragma_emit_insn", "dma_copy")
                with ib.for_range(0, 32, 'j') as j:
                    A[(i * 8 + k) * 32 + j] = AL[j]
    stmt = ib.get()
    # print(stmt)
    stmt = akg.tvm.ir_pass.InjectMultiCore(stmt, 8, 0, False)
    # print(stmt)
    assert(stmt.attr_key == "thread_extent")
    assert(stmt.value.value == 2)
    assert(stmt.body.extent.value == 8)


def test_dep_between_core_003():
    ''' tail_align dep '''
    ib = akg.tvm.ir_builder.create()
    zero = akg.tvm.const(0, "int32")
    A = ib.pointer("float32", name='A')
    with ib.for_range(0, 2, 'i') as i:
        with ib.for_range(0, 8, 'k') as k:
            with ib.new_scope():
                ib.scope_attr(zero, "pragma_emit_insn", "dma_copy")
                with ib.for_range(0, 11, 'j') as j:
                    A[(i * 8 + k) * 11 + j] = akg.tvm.const(1, A.dtype)
    stmt = ib.get()
    # print(stmt)
    stmt = akg.tvm.ir_pass.InjectMultiCore(stmt, 8, 0, False)
    # print(stmt)
    assert(stmt.attr_key == "thread_extent")
    assert(stmt.value.value == 8)
    assert(stmt.body.extent.value == 2)


def test_plan_sibling_loop():
    ''' sibling loop prevent mc'''
    ib = akg.tvm.ir_builder.create()
    zero = akg.tvm.const(0, "int32")
    A = ib.pointer("float32", name='A')
    B = ib.pointer("float32", name='B')
    with ib.for_range(0, 100, 'i') as i:
        with ib.new_scope():
            ib.scope_attr(zero, "pragma_emit_insn", "dma_copy")
            with ib.for_range(0, 1024, 'j') as j:
                A[i * 1024 + j] = akg.tvm.const(1, A.dtype)
    with ib.for_range(0, 100, 'i') as i:
        with ib.new_scope():
            ib.scope_attr(zero, "pragma_emit_insn", "dma_copy")
            with ib.for_range(0, 1024, 'j') as j:
                B[i * 1024 + j] = akg.tvm.const(1, B.dtype)
    stmt = ib.get()
    # print(stmt)
    stmt = akg.tvm.ir_pass.InjectMultiCore(stmt, 4, 0, False)
    # print(stmt)
    assert(not isinstance(stmt, akg.tvm.stmt.AttrStmt))


def test_plan_sibling_attrstmt():
    ''' sibling attrstmt prevent mc'''
    ib = akg.tvm.ir_builder.create()
    zero = akg.tvm.const(0, "int32")
    A = ib.pointer("float32", name='A')
    B = ib.pointer("float32", name='B')
    with ib.for_range(0, 100, 'i') as i:
        with ib.new_scope():
            ib.scope_attr(zero, "pragma_emit_insn", "dma_copy")
            with ib.for_range(0, 1024, 'j') as j:
                A[i * 1024 + j] = akg.tvm.const(1, A.dtype)
    with ib.new_scope():
        ib.scope_attr(zero, "pragma_emit_insn", "dma_copy")
        ib.emit(akg.tvm.call_pure_intrin("float32", "foo", zero))
    stmt = ib.get()
    # print(stmt)
    stmt = akg.tvm.ir_pass.InjectMultiCore(stmt, 4, 0, False)
    # print(stmt)
    assert(not isinstance(stmt, akg.tvm.stmt.AttrStmt))


def test_storage_scope_overtop_tolerant():
    ''' tolerant to storage scope overtop. pending to fix later '''
    ib = akg.tvm.ir_builder.create()
    zero = akg.tvm.const(0, "int32")
    A = ib.pointer("float32", name='A')
    AL = ib.allocate("float32", 32, name="AL", scope="local")
    with ib.for_range(0, 8, 'i') as i:
        with ib.new_scope():
            ib.scope_attr(zero, "pragma_emit_insn", "dma_copy")
            AL[0] = A[i * 32]
    stmt = ib.get()
    # print(stmt)
    stmt = akg.tvm.ir_pass.InjectMultiCore(stmt, 8, 0, False)
    # print(stmt)
    assert(stmt.attr_key == "thread_extent")
    assert(stmt.value.value == 8)


if __name__ == '__main__':
    test_single_axis_001()
    test_single_axis_002()
    test_single_axis_003()
    test_multi_axis_001()
    test_multi_axis_002()
    test_dep_between_core_001()
    test_dep_between_core_002()
    test_dep_between_core_003()
    test_plan_sibling_loop()
    test_plan_sibling_attrstmt()
    test_storage_scope_overtop_tolerant()
