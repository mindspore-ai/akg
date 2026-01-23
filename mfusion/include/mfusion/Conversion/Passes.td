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

def ConvertTorchToMuse : Pass<"convert-torch-to-muse", "::mlir::ModuleOp"> {
  let summary = "Convert Torch operations to Muse dialect operations";

  let description = [{
    This pass converts Torch dialect operations to the Muse dialect.
  }];

  let constructor = "mlir::createConvertTorchToMusePass()";

  let dependentDialects = [
    "::mlir::torch::TorchDialect",
    "::mlir::muse::MuseDialect"
  ];
}

def ConvertMuseToTorch : Pass<"convert-muse-to-torch", "::mlir::ModuleOp"> {
  let summary = "Convert Muse operations to Torch dialect operations";

  let description = [{
    This pass converts Muse dialect operations to the Torch dialect.
  }];

  let constructor = "mlir::createConvertMuseToTorchPass()";

  let dependentDialects = [
    "::mlir::torch::TorchDialect",
    "::mlir::muse::MuseDialect"
  ];
}

#endif // MFUSION_CONVERSION_PASSES
