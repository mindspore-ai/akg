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

#ifndef AKG_MLIR_DIALECT_NPUVECTOR_PASSES
#define AKG_MLIR_DIALECT_NPUVECTOR_PASSES

include "mlir/Pass/PassBase.td"

def OutlineVectorFunction : Pass<"outline-vector-function", "mlir::ModuleOp"> {
  let summary = "This pass is designed to support SMID Reg";
  let description = [{
    This pass scans each function in a module and outlines contiguous regions of
    NPUVector-based computation into separate private functions. It is designed
    to separate pure NPUVector compute logic from surrounding control flow and
    memory transfer operations.
  }];
  let constructor = "mlir::npuvector::createOutlineVectorFunctionPass()";
}

#endif // AKG_MLIR_DIALECT_NPUVECTOR_PASSES
