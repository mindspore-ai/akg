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

"""UT for clone operator."""

from mfusion.torch.inductor import fuse_and_optimize

from ut_utils.mlir_checker import MlirChecker

MLIR_CLONE = r"""
module {
  func.func @main(%arg0: !torch.vtensor<[2,2],f32>) -> !torch.vtensor<[2,2],f32> {
    %int0 = torch.constant.int 0
    %0 = torch.aten.clone %arg0, %int0 : !torch.vtensor<[2,2],f32>, !torch.int -> !torch.vtensor<[2,2],f32>
    return %0 : !torch.vtensor<[2,2],f32>
  }
}
"""

MLIR_CLONE_CONTIGUOUS = r"""
module {
  func.func @main(%arg0: !torch.vtensor<[2,2],f32>) -> !torch.vtensor<[2,2],f32> {
    %int1 = torch.constant.int 1
    %0 = torch.aten.clone %arg0, %int1 : !torch.vtensor<[2,2],f32>, !torch.int -> !torch.vtensor<[2,2],f32>
    return %0 : !torch.vtensor<[2,2],f32>
  }
}
"""

MLIR_CLONE_CHANNELS_LAST = r"""
module {
  func.func @main(%arg0: !torch.vtensor<[1,2,2,4],f32>) -> !torch.vtensor<[1,2,2,4],f32> {
    %int2 = torch.constant.int 2
    %0 = torch.aten.clone %arg0, %int2 : !torch.vtensor<[1,2,2,4],f32>, !torch.int -> !torch.vtensor<[1,2,2,4],f32>
    return %0 : !torch.vtensor<[1,2,2,4],f32>
  }
}
"""

def test_clone():
    """Verify identity clone (memory_format=0) is pruned after fuse and optimize."""
    result = fuse_and_optimize(MLIR_CLONE)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_no_op("torch.aten.clone"), checker.error


def test_clone_contiguous_kept():
    """ContiguousFormat (1) requests layout semantics and must be preserved."""
    result = fuse_and_optimize(MLIR_CLONE_CONTIGUOUS)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_has_op("torch.aten.clone", count=1), checker.error


def test_clone_channels_last_kept():
    """ChannelsLast (2) requests layout change and must be preserved."""
    result = fuse_and_optimize(MLIR_CLONE_CHANNELS_LAST)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_has_op("torch.aten.clone", count=1), checker.error
