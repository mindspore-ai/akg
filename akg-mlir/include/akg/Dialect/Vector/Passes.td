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

#ifndef AKG_MLIR_DIALECT_VECTOR_PASSES
#define AKG_MLIR_DIALECT_VECTOR_PASSES

include "mlir/Pass/PassBase.td"

def VectorLegalizeType : Pass<"vector-legalize-type", "mlir::ModuleOp"> {
  let summary = "Legalize i8 and float types for SIMD (vector + arith)";
  let description = [{
    This pass legalizes certain type conversions that are not directly supported by SIMD
    instructions. Currently, it handles the following pattern:

    - `arith.uitofp` from `i8` to `f32`:
      The operation is rewritten as `arith.extui` from `i8` to `i16` followed by
      `arith.uitofp` from `i16` to `f32`. 

    The pass only applies to functions that carry the `kVectorFunctionAttr` attribute.
  }];
  let constructor = "mlir::vector::createVectorLegalizeTypePass()";
  let dependentDialects = [
    "arith::ArithDialect",
    "vector::VectorDialect"
  ];
}

#endif // AKG_MLIR_DIALECT_VECTOR_PASSES
