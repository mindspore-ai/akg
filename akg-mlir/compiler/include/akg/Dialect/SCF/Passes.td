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

#ifndef AKG_MLIR_DIALECT_SCF_PASSES
#define AKG_MLIR_DIALECT_SCF_PASSES

include "mlir/Pass/PassBase.td"

def RewriteReduceInMultiLevelMemory
    : Pass<"rewrite-reduce-in-multi-level-memory", "mlir::func::FuncOp"> {
  let summary = "rewrite reduce ops in multi-level memory.";
  let constructor = "mlir::createRewriteReduceInMultiLevelMemoryPass()";
  let description = "rewrite reduce ops in multi-level memory; currently used in gpu backendfor SIMT reduction.";
}

def PrimeNumReplaceForDynamicShape
    : Pass<"prime-num-replace-for-dynamic-shape", "mlir::func::FuncOp"> {
  let summary = "Replace unknown mapping axes to prime and restore them later.";
  let constructor = "mlir::createPrimeNumReplaceForDynamicShapePass()";
  let options = [
    Option<"action", "action", "std::string", /*default=*/"\"replace\"",
           "The action, replace(default) or restore.">,
  ];  
  let dependentDialects = [
  ];   
  let description = "Replace unknown mapping axes to prime and restore them later.";
}

def NPUVectorVectorizePass : Pass<"npuvector-vectorize", "mlir::func::FuncOp"> {
  let summary = "SCF vectorization pass using NPUVector dialect with dynamic shape support";
  let description = [{
    Vectorizes scf.for loops using NPUVector dialect, supporting dynamic shapes.
    
    Key features:
      - Processes scf.for loops (not affine.for)
      - Handles memref.load/store (not affine.load/store)
      - Uses npuvector.transfer_read/write
      - Supports dynamic shapes (vector<?xT>)
    
    Reads attributes from loops:
      - "vector" attribute → Elementwise vectorization
      - "reduction" attribute → Reduction vectorization
    
    Loop bounds can be:
      - Static (constant index) → generates vector<VF×T>
      - Dynamic (SSA value) → generates vector<?×T>
  }];
  let constructor = "mlir::scf::createNPUVectorVectorizePass()";
  let options = [
    Option<"defaultVF", "default-vf", "int64_t", /*default=*/"512",
           "Vector factor used for dynamic bounds or when trip count equals to 1">
  ];
}

#endif // AKG_MLIR_DIALECT_SCF_PASSES
