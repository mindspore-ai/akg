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

//===----------------------------------------------------------------------===//
// Torch dialect fusion passes (run before convert-torch-to-mfuse)
//===----------------------------------------------------------------------===//

def TorchFusion : Pass<"torch-fusion", "::mlir::ModuleOp"> {
  let summary = "Torch dialect fusion pipeline (RoPE etc.) before Convert Torch to Mfuse";
  let description = [{
    Runs a sequence of fusion passes on Torch IR before convert-torch-to-mfuse.
    Currently includes torch-fuse-rope; more passes can be added for future extension.
  }];
  let constructor = "mlir::createTorchFusionPass()";
  let dependentDialects = [
    "::mlir::torch::Torch::TorchDialect"
  ];
}

def TorchFuseRoPE : Pass<"torch-fuse-rope", "::mlir::ModuleOp"> {
  let summary = "Fuse decomposed RoPE on Torch dialect into torch.npu.npu_rotary_mul";
  let description = [{
    Matches the RoPE pattern on Torch IR (slice/neg/cat + mul.Tensor + add.Tensor)
    and replaces it with a single torch.operator "torch.npu.npu_rotary_mul"(x, cos, sin).
    Runs before convert-torch-to-mfuse.
  }];
  let constructor = "mlir::createTorchFuseRoPEPass()";
  let dependentDialects = [
    "::mlir::torch::Torch::TorchDialect"
  ];
}

def TorchFuseRmsNorm : Pass<"torch-fuse-rms-norm", "::mlir::ModuleOp"> {
  let summary = "Fuse decomposed RmsNorm on Torch dialect into torch.npu.npu_rms_norm";
  let description = [{
    Matches decomposed RmsNorm patterns (add+rsqrt+mul+mul, with optional npu_dtype_cast)
    and replaces them with torch.operator "torch.npu.npu_rms_norm"(x, gamma, eps).
    Runs before convert-torch-to-mfuse.
  }];
  let constructor = "mlir::createTorchFuseRmsNormPass()";
  let dependentDialects = [
    "::mlir::torch::Torch::TorchDialect"
  ];
}

def ConvertTorchToMfuse : Pass<"convert-torch-to-mfuse", "::mlir::ModuleOp"> {
  let summary = "Convert Torch operations to Mfuse dialect operations";

  let description = [{
    This pass converts Torch dialect operations to the Mfuse dialect.
  }];

  let constructor = "mlir::createConvertTorchToMfusePass()";

  let dependentDialects = [
    "::mlir::torch::Torch::TorchDialect",
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
    "::mlir::torch::Torch::TorchDialect",
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
    "::mlir::torch::Torch::TorchDialect",
    "::mlir::mfuse::MfuseDialect"
  ];
}

def ConvertMfuseToDvm : Pass<"convert-mfuse-to-dvm", "::mlir::ModuleOp"> {
  let summary = "Convert outlined Mfuse subgraphs to DVM dialect operations";

  let description = [{
    This pass converts Mfuse operations inside functions tagged with
    'mfusion.outlined' and fusion_type == "dvm" into DVM dialect operations,
    while leaving the main graph in Mfuse dialect. It also lifts constants to
    arguments and inserts dvm.load/dvm.store for graph IO semantics.
  }];

  let constructor = "mlir::createConvertMfuseToDvmPass()";

  let dependentDialects = [
    "::mlir::mfuse::MfuseDialect",
    "::mlir::dvm::DvmDialect",
    "::mlir::func::FuncDialect",
    "::mlir::arith::ArithDialect"
  ];
}

#endif // MFUSION_CONVERSION_PASSES
