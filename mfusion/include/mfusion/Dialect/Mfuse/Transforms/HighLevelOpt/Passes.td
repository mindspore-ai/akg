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

def FoldBnsdFlattenSumPass : Pass<"mfuse-fold-bnsd-flatten-sum", "::mlir::func::FuncOp"> {
  let summary = "Fold permute-clone-flatten reductions into direct rank-4 reductions";
  let description = [{
    Rewrites reductions over a flattened contiguous clone of a rank-4
    [B, N, S, D] -> [B, S, N, D] permutation into direct reductions over the
    original rank-4 tensor. The materialized flattened tensor is preserved for
    other users such as matmul; only the reduce branch is redirected.

    In guarded mode (default) the rewrite additionally requires that the
    [B, N, S, D] source has a DVM-clusterable ELEMWISE producer (per the Split
    OpRegistry), so the new producer -> reduce_sum edge is consumable by the
    subsequent DVM Cluster/Split pass. This is a DVM-specific profitability
    guard, not full backend legality. Use guard=false for tests that need
    unconditional semantic rewrite coverage.
  }];
  let options = [
    Option<"guard", "guard", "bool", "true",
           "Require a DVM-clusterable ELEMWISE producer before rewriting (guarded DVM-profitability mode). When false, every semantic match is rewritten unconditionally (legacy behavior, for testing only).">
  ];
  let constructor = "mlir::mfuse::createFoldBnsdFlattenSumPass()";
  let dependentDialects = [
    "::mlir::mfuse::MfuseDialect",
    "::mlir::func::FuncDialect"
  ];
}


def PromoteBinaryOpsPass : Pass<"mfuse-promote-binary-ops", "::mlir::ModuleOp"> {
  let summary = "Promote binary operations to ensure consistent input types";
  let description = [{
    This pass ensures that binary operations have consistent input types:
    - For arithmetic operations (Add, Sub, Mul, Div, etc.), inputs are promoted to match the result type
    - For comparison operations (Eq, Ne, Gt, Ge, Lt, Le, etc.), inputs are promoted to the same type
  }];
  let constructor = "mlir::mfuse::createPromoteBinaryOpsPass()";
}

#endif // MFUSION_DIALECT_MFUSE_TRANSFORMS_HIGH_LEVEL_OPT_PASSES_TD
