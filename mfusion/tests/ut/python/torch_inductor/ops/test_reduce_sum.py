# Copyright 2026 Huawei Technologies Co., Ltd
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
"""UT for reduce_sum operator."""

import textwrap
from mfusion.passmanager import PassManager
from ut_utils.mlir_checker import MlirChecker


def test_ut_create_reduce_sum_pass_reduce_dim0_keepdim_false():
    """Verify test reduce_sum pass inserts mfuse.reduce_sum with dim=0, keepdim=False."""
    text = textwrap.dedent("""
        module {
          func.func @main(%arg0: tensor<?x2xf32, #mfuse.symshape<["s0", "2"]>>) attributes {mfuse.ut_reduce_sum_dims = [0], mfuse.ut_reduce_sum_keepdim = false} {
            return
          }
        }
    """)
    checker = MlirChecker.parse_torch_module(text)
    module = checker.module

    with module.context:
        pm = PassManager.parse("builtin.module(func.func(mfuse-ut-create-reduce-sum))")
        pm.run(module.operation)

    assert checker.check_has_op("mfuse.reduce_sum", count=1), checker.error
    assert checker.check_text_contains("#mfuse.symshape<"), checker.error


def test_ut_create_reduce_sum_pass_reduce_dim0_keepdim_true():
    """Verify test reduce_sum pass inserts mfuse.reduce_sum with dim=0, keepdim=True."""
    text = textwrap.dedent("""
        module {
          func.func @main(%arg0: tensor<?x2xf32, #mfuse.symshape<["s0", "2"]>>) attributes {mfuse.ut_reduce_sum_dims = [0], mfuse.ut_reduce_sum_keepdim = true} {
            return
          }
        }
    """)
    checker = MlirChecker.parse_torch_module(text)
    module = checker.module

    with module.context:
        pm = PassManager.parse("builtin.module(func.func(mfuse-ut-create-reduce-sum))")
        pm.run(module.operation)

    assert checker.check_has_op("mfuse.reduce_sum", count=1), checker.error
    assert checker.check_text_contains("#mfuse.symshape<"), checker.error


def test_ut_create_reduce_sum_pass_reduce_dim1_keepdim_false():
    """Verify test reduce_sum pass inserts mfuse.reduce_sum with dim=1, keepdim=False."""
    text = textwrap.dedent("""
        module {
          func.func @main(%arg0: tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>>) attributes {mfuse.ut_reduce_sum_dims = [1], mfuse.ut_reduce_sum_keepdim = false} {
            return
          }
        }
    """)
    checker = MlirChecker.parse_torch_module(text)
    module = checker.module

    with module.context:
        pm = PassManager.parse("builtin.module(func.func(mfuse-ut-create-reduce-sum))")
        pm.run(module.operation)

    assert checker.check_has_op("mfuse.reduce_sum", count=1), checker.error
    assert checker.check_text_contains("#mfuse.symshape<"), checker.error


def test_ut_create_reduce_sum_pass_reduce_dim1_keepdim_true():
    """Verify test reduce_sum pass inserts mfuse.reduce_sum with dim=1, keepdim=True."""
    text = textwrap.dedent("""
        module {
          func.func @main(%arg0: tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>>) attributes {mfuse.ut_reduce_sum_dims = [1], mfuse.ut_reduce_sum_keepdim = true} {
            return
          }
        }
    """)
    checker = MlirChecker.parse_torch_module(text)
    module = checker.module

    with module.context:
        pm = PassManager.parse("builtin.module(func.func(mfuse-ut-create-reduce-sum))")
        pm.run(module.operation)

    assert checker.check_has_op("mfuse.reduce_sum", count=1), checker.error
    assert checker.check_text_contains("#mfuse.symshape<"), checker.error


def test_ut_create_reduce_sum_pass_reduce_all_dims_keepdim_false():
    """Verify test reduce_sum pass inserts mfuse.reduce_sum with all dims, keepdim=False."""
    text = textwrap.dedent("""
        module {
          func.func @main(%arg0: tensor<?x?xf32, #mfuse.symshape<["s0", "s1"]>>) attributes {mfuse.ut_reduce_sum_dims = [0, 1], mfuse.ut_reduce_sum_keepdim = false} {
            return
          }
        }
    """)
    checker = MlirChecker.parse_torch_module(text)
    module = checker.module

    with module.context:
        pm = PassManager.parse("builtin.module(func.func(mfuse-ut-create-reduce-sum))")
        pm.run(module.operation)

    assert checker.check_has_op("mfuse.reduce_sum", count=1), checker.error
    assert checker.check_text_contains("#mfuse.symshape<"), checker.error


def test_ut_create_reduce_sum_pass_reduce_all_dims_keepdim_true():
    """Verify test reduce_sum pass inserts mfuse.reduce_sum with all dims, keepdim=True."""
    text = textwrap.dedent("""
        module {
          func.func @main(%arg0: tensor<?x?xf32, #mfuse.symshape<["s0", "s1"]>>) attributes {mfuse.ut_reduce_sum_dims = [0, 1], mfuse.ut_reduce_sum_keepdim = true} {
            return
          }
        }
    """)
    checker = MlirChecker.parse_torch_module(text)
    module = checker.module

    with module.context:
        pm = PassManager.parse("builtin.module(func.func(mfuse-ut-create-reduce-sum))")
        pm.run(module.operation)

    assert checker.check_has_op("mfuse.reduce_sum", count=1), checker.error
    assert checker.check_text_contains("#mfuse.symshape<"), checker.error


def test_ut_create_reduce_sum_pass_3d_reduce_dim1_keepdim_false():
    """Verify test reduce_sum pass inserts mfuse.reduce_sum with 3D tensor, dim=1, keepdim=False."""
    text = textwrap.dedent("""
        module {
          func.func @main(%arg0: tensor<?x?x?xf32, #mfuse.symshape<["s0", "s1", "s2"]>>) attributes {mfuse.ut_reduce_sum_dims = [1], mfuse.ut_reduce_sum_keepdim = false} {
            return
          }
        }
    """)
    checker = MlirChecker.parse_torch_module(text)
    module = checker.module

    with module.context:
        pm = PassManager.parse("builtin.module(func.func(mfuse-ut-create-reduce-sum))")
        pm.run(module.operation)

    assert checker.check_has_op("mfuse.reduce_sum", count=1), checker.error
    assert checker.check_text_contains("#mfuse.symshape<"), checker.error


def test_ut_create_reduce_sum_pass_3d_reduce_dim1_keepdim_true():
    """Verify test reduce_sum pass inserts mfuse.reduce_sum with 3D tensor, dim=1, keepdim=True."""
    text = textwrap.dedent("""
        module {
          func.func @main(%arg0: tensor<?x?x?xf32, #mfuse.symshape<["s0", "s1", "s2"]>>) attributes {mfuse.ut_reduce_sum_dims = [1], mfuse.ut_reduce_sum_keepdim = true} {
            return
          }
        }
    """)
    checker = MlirChecker.parse_torch_module(text)
    module = checker.module

    with module.context:
        pm = PassManager.parse("builtin.module(func.func(mfuse-ut-create-reduce-sum))")
        pm.run(module.operation)

    assert checker.check_has_op("mfuse.reduce_sum", count=1), checker.error
    assert checker.check_text_contains("#mfuse.symshape<"), checker.error


def test_ut_create_reduce_sum_pass_3d_reduce_multiple_dims_keepdim_false():
    """Verify test reduce_sum pass inserts mfuse.reduce_sum with 3D tensor, multiple dims, keepdim=False."""
    text = textwrap.dedent("""
        module {
          func.func @main(%arg0: tensor<?x?x?xf32, #mfuse.symshape<["s0", "s1", "s2"]>>) attributes {mfuse.ut_reduce_sum_dims = [0, 2], mfuse.ut_reduce_sum_keepdim = false} {
            return
          }
        }
    """)
    checker = MlirChecker.parse_torch_module(text)
    module = checker.module

    with module.context:
        pm = PassManager.parse("builtin.module(func.func(mfuse-ut-create-reduce-sum))")
        pm.run(module.operation)

    assert checker.check_has_op("mfuse.reduce_sum", count=1), checker.error
    assert checker.check_text_contains("#mfuse.symshape<"), checker.error


def test_ut_create_reduce_sum_pass_3d_reduce_multiple_dims_keepdim_true():
    """Verify test reduce_sum pass inserts mfuse.reduce_sum with 3D tensor, multiple dims, keepdim=True."""
    text = textwrap.dedent("""
        module {
          func.func @main(%arg0: tensor<?x?x?xf32, #mfuse.symshape<["s0", "s1", "s2"]>>) attributes {mfuse.ut_reduce_sum_dims = [0, 2], mfuse.ut_reduce_sum_keepdim = true} {
            return
          }
        }
    """)
    checker = MlirChecker.parse_torch_module(text)
    module = checker.module

    with module.context:
        pm = PassManager.parse("builtin.module(func.func(mfuse-ut-create-reduce-sum))")
        pm.run(module.operation)

    assert checker.check_has_op("mfuse.reduce_sum", count=1), checker.error
    assert checker.check_text_contains("#mfuse.symshape<"), checker.error
