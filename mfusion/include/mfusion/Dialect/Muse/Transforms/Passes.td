/**
 * Copyright 2026 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MFUSION_DIALECT_MUSE_TRANSFORMS_PASSES
#define MFUSION_DIALECT_MUSE_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// SetTensorDevice pass
//===----------------------------------------------------------------------===//

def SetTensorDevice : Pass<"set-tensor-device", "::mlir::ModuleOp"> {
  let summary = "Set device information for tensors in MUSE dialect";

  let description = [{
    This pass sets device information for all tensors in MUSE dialect operations.
    It propagates device information from network inputs (function arguments) to
    all operation outputs. The device information is set from the first example_input
    and applied to all function arguments.
  }];

  let options = [
    Option<"deviceType", "device-type", "std::string", "\"cpu\"",
           "Device type (e.g., 'cpu', 'npu') for all input tensors">,
    Option<"deviceIndex", "device-index", "int64_t", "-1",
           "Device index for all input tensors">
  ];

  let constructor = "mlir::createSetTensorDevice()";

  let dependentDialects = [
    "::mlir::muse::MuseDialect",
    "::mlir::func::FuncDialect"
  ];
}

#endif // MFUSION_DIALECT_MUSE_TRANSFORMS_PASSES
