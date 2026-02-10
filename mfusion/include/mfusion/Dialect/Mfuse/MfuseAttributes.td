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

#ifndef MFUSE_DIALECT_MFUSE_ATTRIBUTES_TD
#define MFUSE_DIALECT_MFUSE_ATTRIBUTES_TD

include "mfusion/Dialect/Mfuse/MfuseDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributes.td"

// Symbolic shape attribute for persisting SymEngine expressions per dimension.
def Mfuse_SymbolicShapeAttr : AttrDef<Mfuse_Dialect, "SymbolicShape"> {
  let mnemonic = "symshape";
  let summary = "Symbolic shape attribute for MFUSE tensors";
  let description = [{
    Stores a list of symbolic expressions, one per tensor dimension.
    Each expression is serialized as a string for persistence across passes.
  }];

  let parameters = (ins "mlir::ArrayAttr":$exprs);
  let assemblyFormat = "`<` $exprs `>`";

  let extraClassDeclaration = [{
    std::vector<SymEngine::RCP<const SymEngine::Basic>> getSymEngineExprs() const;
  }];
}

#endif // MFUSE_DIALECT_MFUSE_ATTRIBUTES_TD
