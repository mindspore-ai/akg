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
"""UT for reshape operator."""

from mfusion.passmanager import PassManager
from ut_utils.mlir_checker import MlirChecker


def test_ut_create_reshape_pass_from_python_passmanager_parse():
    """Verify test reshape pass inserts mfuse.reshape."""
    text = r"""
module {
  func.func @main(%arg0: tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>>) attributes {mfuse.ut_create_reshape = true} {
    return
  }
}
"""
    checker = MlirChecker.parse_torch_module(text)
    module = checker.module

    with module.context:
        pm = PassManager.parse("builtin.module(func.func(mfuse-ut-create-reshape))")
        pm.run(module.operation)

    assert checker.check_has_op("mfuse.reshape", count=1), checker.error
    assert checker.check_text_contains("#mfuse.symshape<"), checker.error
