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

#ifndef MFUSION_CONVERSION_PASSES
#define MFUSION_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTorchToMfuse : Pass<"convert-torch-to-mfuse", "::mlir::ModuleOp"> {
  let summary = "Convert Torch operations to Mfuse dialect operations";

  let description = [{
    This pass converts Torch dialect operations to the Mfuse dialect.
  }];

  let constructor = "mlir::createConvertTorchToMfusePass()";

  let dependentDialects = [
    "::mlir::torch::TorchDialect",
    "::mlir::mfuse::MfuseDialect"
  ];
}

def ConvertTorchSymbolToMfuse : Pass<"convert-torch-symbol-to-mfuse", "::mlir::ModuleOp"> {
  let summary = "Attach symbolic shape attributes after Torch-to-Mfuse conversion";

  let description = [{
    This pass consumes torch.bind_symbolic_shape ops and attaches symbolic shape
    attributes to ranked tensor types produced by Torch-to-Mfuse conversion.
  }];

  let constructor = "mlir::createConvertTorchSymbolToMfusePass()";

  let dependentDialects = [
    "::mlir::torch::TorchDialect",
    "::mlir::mfuse::MfuseDialect"
  ];
}

def ConvertMfuseToTorch : Pass<"convert-mfuse-to-torch", "::mlir::ModuleOp"> {
  let summary = "Convert Mfuse operations to Torch dialect operations";

  let description = [{
    This pass converts Mfuse dialect operations to the Torch dialect.
  }];

  let constructor = "mlir::createConvertMfuseToTorchPass()";

  let dependentDialects = [
    "::mlir::torch::TorchDialect",
    "::mlir::mfuse::MfuseDialect"
  ];
}

#endif // MFUSION_CONVERSION_PASSES
