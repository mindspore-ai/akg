/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef AKG_MLIR_DIALECT_TOSA_PASSES
#define AKG_MLIR_DIALECT_TOSA_PASSES

include "mlir/Pass/PassBase.td"

def AKGTosaOperatorIdentify : Pass<"akg-tosa-operator-identify", "func::FuncOp"> {
  let summary = "Identify all operator type templates.";
  let constructor = "mlir::createAKGOperatorIdentifyPass()";
}

def FoldDimension : Pass<"fold-dimension", "func::FuncOp"> {
  let summary = "Fold consecutive dimensions together if requirements satisfied.";
  let constructor = "mlir::createFoldDimensionPass()";
}

#endif // AKG_MLIR_DIALECT_TOSA_PASSES
