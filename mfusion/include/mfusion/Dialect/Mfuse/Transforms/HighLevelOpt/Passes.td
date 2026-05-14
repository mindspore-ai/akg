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

#ifndef MFUSION_DIALECT_MFUSE_TRANSFORMS_HIGH_LEVEL_OPT_PASSES_TD
#define MFUSION_DIALECT_MFUSE_TRANSFORMS_HIGH_LEVEL_OPT_PASSES_TD

include "mlir/Pass/PassBase.td"


def ReorderOpsPass : Pass<"mfuse-reorder-ops", "::mlir::ModuleOp"> {
  let summary = "Reorder Cast and type-insensitive operations";
  let description = [{
    This pass reorders Cast operations and type-insensitive operations (like Reshape,
    Transpose, etc.) to reduce unnecessary type conversions and improve performance.
  }];
  let constructor = "mlir::mfuse::createReorderOpsPass()";
}


def ConvertBFloat16Pass : Pass<"mfuse-convert-bfloat16", "::mlir::ModuleOp"> {
  let summary = "Convert bfloat16 inputs to float32 for operations that don't support bfloat16";
  let description = [{
    This pass adds Cast operations to convert bfloat16 inputs to float32 for operations
    that don't support bfloat16, and converts the results back to bfloat16 if needed.
  }];
  let constructor = "mlir::mfuse::createConvertBFloat16Pass()";
}


def RaiseReductionPrecisionPass : Pass<"mfuse-raise-reduction-precision", "::mlir::ModuleOp"> {
  let summary = "Raise reduction precision from float16 to float32";
  let description = [{
    This pass increases the precision of reduction operations (like ReduceSum) by
    converting inputs from float16 to float32 before the reduction, and converting
    results back to float16 if needed.
  }];
  let constructor = "mlir::mfuse::createRaiseReductionPrecisionPass()";
}

#endif // MFUSION_DIALECT_MFUSE_TRANSFORMS_HIGH_LEVEL_OPT_PASSES_TD
