#!/usr/bin/env python3
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
"""UT for broadcast_to operator."""

import textwrap

from mfusion.passmanager import PassManager
from ut_utils.mlir_checker import MlirChecker


def test_ut_create_broadcast_to_static_leading_dim():
    """Verify UT pass inserts mfuse.broadcast_to and attaches symbolic shape (static leading dim)."""
    text = textwrap.dedent(
        """
        module {
          func.func @main(%arg0: tensor<?x4xf32, #mfuse.symshape<["s0", "4"]>>)
              attributes {mfuse.ut_broadcast_to_outshape = [2, -1, 4]} {
            return
          }
        }
        """
    )
    checker = MlirChecker.parse_torch_module(text)
    module = checker.module

    with module.context:
        pm = PassManager.parse("builtin.module(func.func(mfuse-ut-create-broadcast-to))")
        pm.run(module.operation)

    assert checker.check_has_op("mfuse.broadcast_to", count=1), checker.error
    # Output tensor should be annotated with symbolic shape after construction.
    assert checker.check_text_contains("-> tensor<2x?x4xf32, #mfuse.symshape<"), checker.error


def test_ut_create_broadcast_to_dynamic_trailing_dim():
    """Verify UT pass inserts mfuse.broadcast_to and attaches symbolic shape (dynamic trailing dim)."""
    text = textwrap.dedent(
        """
        module {
          func.func @main(%arg0: tensor<1x2x?xf32, #mfuse.symshape<["1", "2", "s0"]>>)
              attributes {mfuse.ut_broadcast_to_outshape = [4, 2, -1]} {
            return
          }
        }
        """
    )
    checker = MlirChecker.parse_torch_module(text)
    module = checker.module

    with module.context:
        pm = PassManager.parse("builtin.module(func.func(mfuse-ut-create-broadcast-to))")
        pm.run(module.operation)

    assert checker.check_has_op("mfuse.broadcast_to", count=1), checker.error
    # Output tensor should be annotated with symbolic shape after construction.
    assert checker.check_text_contains("-> tensor<4x2x?xf32, #mfuse.symshape<"), checker.error
