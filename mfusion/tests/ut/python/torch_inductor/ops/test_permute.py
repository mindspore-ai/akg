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
"""UT for permute operator."""

import textwrap

from mfusion.passmanager import PassManager
from ut_utils.mlir_checker import MlirChecker


def test_ut_create_permute_refines_static_result_dim():
    """Verify mfuse.permute builder refines static dims before attaching symshape."""
    text = textwrap.dedent(
        """
        module {
          func.func @main(%arg0: tensor<?x16xf32, #mfuse.symshape<["s0", "16"]>>)
              attributes {mfuse.ut_permute = [1, 0]} {
            return
          }
        }
        """
    )
    checker = MlirChecker.parse_torch_module(text)
    module = checker.module

    with module.context:
        pm = PassManager.parse("builtin.module(func.func(mfuse-ut-create-permute))")
        pm.run(module.operation)

    assert checker.check_has_op("mfuse.permute", count=1), checker.error
    assert checker.check_text_contains(
        '-> tensor<16x?xf32, #mfuse.symshape<["16", "s0"]>>'
    ), checker.error
