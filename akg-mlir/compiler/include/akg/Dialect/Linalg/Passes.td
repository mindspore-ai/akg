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

#ifndef AKG_MLIR_DIALECT_LINAGLG_PASSES
#define AKG_MLIR_DIALECT_LINAGLG_PASSES

include "mlir/Pass/PassBase.td"

def LinalgSimplify : Pass<"linalg-simplify-ops", "func::FuncOp"> {
  let summary = "simplify tensor.empty/tensor.dim Ops in linalg";
  let constructor = "mlir::createLinalgSimplifyPass()";
  let dependentDialects = [
    "linalg::LinalgDialect",
    "memref::MemRefDialect"
  ];
}

def LinalgExtBufferize : Pass<"linalg-ext-bufferize", "func::FuncOp"> {
  let summary = "Bufferize the linalg-ext and linalg dialect";
  let constructor = "mlir::createLinalgExtBufferizePass()";
  let dependentDialects = [
    "affine::AffineDialect",
    "bufferization::BufferizationDialect",
    "linalg::LinalgDialect",
    "linalgExt::LinalgExtDialect",
    "memref::MemRefDialect"
  ];
}

def LinalgTemplated: Pass<"linalg-template-named-ops", "ModuleOp"> {
  let summary = "Convert named ops to templated implementation.";
  let constructor = "mlir::createLinalgTemplatedPass()";
  let options = [
    Option<"templatePath","template-path", "std::string",
           /*default=*/"",
           "the path of operation templated implementation">
  ];
  let dependentDialects = [
    "affine::AffineDialect",
    "scf::SCFDialect",
    "bufferization::BufferizationDialect",
    "linalg::LinalgDialect",
    "linalgExt::LinalgExtDialect",
    "memref::MemRefDialect",
    "fusion::FusionDialect",
    "vector::VectorDialect"
  ];
}

def LinalgLowerTemplateOp : Pass<"linalg-lower-template-ops", "func::FuncOp"> {
  let summary = "Convert templated ops to function call";
  let constructor = "mlir::createLinalgLowerTemplateOpPass()";
  let dependentDialects = [];
}

def MatchAndMarkReductionOps : Pass<"match-and-mark-reduction-ops", "func::FuncOp"> {
  let summary = "Match and mark reduction Ops in linalg";
  let constructor = "mlir::createMatchAndMarkReductionOpsPass()";
  let options = [
    Option<"dialect", "dialect", "std::string", /*default=*/"\"linalg\"",
           "The dialect op need to match & mark."
           "Currently only support linalg & affine dialects.">,
  ];  
  let dependentDialects = [
    "linalg::LinalgDialect",
    "arith::ArithDialect",
    "affine::AffineDialect"
  ];
}

def LinalgCopyBufferize : Pass<"linalg-copy-bufferize", "func::FuncOp"> {
  let summary = "Bufferize linalg copy op";
  let constructor = "mlir::createLinalgCopyBufferizePass()";
  let options = [
    Option<"keepOuts", "keep-outs", "bool", "false",
           "Keep outputs of linalg.copy or not, default is False.">,
  ];
  let dependentDialects = [
    "memref::MemRefDialect",
    "bufferization::BufferizationDialect"
  ];
}

def LinalgElementwiseFusionExt : Pass<"linalg-fuse-elementwise-ext"> {
  let summary = "Extension of elementwise fusion on tensors";
  let constructor = "mlir::createLinalgElementwiseFusionExtPass()";
  let dependentDialects = [
    "affine::AffineDialect",
    "linalg::LinalgDialect",
    "linalgExt::LinalgExtDialect",
    "memref::MemRefDialect"
  ];
}

#endif // AKG_MLIR_DIALECT_LINAGLG_PASSES
