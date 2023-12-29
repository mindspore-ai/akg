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

#ifndef AKG_CONVERSION_PASSES
#define AKG_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"


//===----------------------------------------------------------------------===//
// TosaToLinalgUpdate
//===----------------------------------------------------------------------===//

def TosaMultiReduceToLinalg : Pass<"tosa-multi-reduce-to-linalg", "func::FuncOp"> {
  let summary = "Convert multi tosa.reduce to one linalg.reduce if is possible.";
  let description = [{
    Pass that converts multi tosa.reduce to one linalg.reduce. Use this pass to enable
    multi-dims reduction.
  }];
  let constructor = "mlir::createTosaMultiReduceToLinalgPass()";
}

//===----------------------------------------------------------------------===//
// FusionToMemVec
//===----------------------------------------------------------------------===//

def ConvertFusionToMemVec : Pass<"lower-fusion"> {
  let summary = "Convert Fusion dialect to MemRef or Vector dialect";
  let constructor = "mlir::createLowerFusionPass()";
}

//===----------------------------------------------------------------------===//
// VectorTransferLower
//===----------------------------------------------------------------------===//

def VectorTransferLower : Pass<"vector-transfer-lower"> {
  let summary = "VectorTransferLower";
  let description = [{
    Pass that converts vector transfer read/write to vector load/store.
  }];
  let constructor = "mlir::createVectorTransferLowerPass()";
 }
 //===---------------------------------------------------------------------===//
// ConvertMindSporeToTosa
//===----------------------------------------------------------------------===//

def ConvertMindSporeToTosa : Pass<"convert-mindspore-to-tosa", "func::FuncOp"> {
  let summary = "Convert MindSpore ops to TOSA ops";
  let description = [{
    This pass assumes that TOSA ops are responsible for emitting error
    guards in case of shape mismatches.
  }];
  let constructor = "mlir::createMindSporeToTosaPass()";
}

// ConvertMindSporeToLinalg
//===----------------------------------------------------------------------===//
def ConvertMindSporeToLinalg : Pass<"convert-mindspore-to-linalg", "func::FuncOp"> {
  let summary = "Convert MindSpore ops to Linalg ops";
  let description = [{
    ...
  }];
  let constructor = "mlir::createMindSporeToLinalgPass()";
}

//===----------------------------------------------------------------------===//
// MindSporeFinalizingLower
//===----------------------------------------------------------------------===//
def MindSporeFinalizingLower : Pass<"mindspore-finalizing-lower", "func::FuncOp"> {
  let summary = "Convert MindSpore ops to Linalg ops";
  let description = [{
    ...
  }];
  let constructor = "mlir::createMindSporeFinalizingLowerPass()";
}

//===----------------------------------------------------------------------===//
// LinalgExtLower
//===----------------------------------------------------------------------===//

def LinalgExtLower : Pass<"linalg-ext-lower", "func::FuncOp"> {
  let summary = "LinalgExtLower";
  let description = [{
    Pass that converts LinalgExt operations to the equivalent operations using the
    operations in Affine Dialect.
  }];
  let constructor = "mlir::createLinalgExtLowerPass()";
  let options = [
    Option<"vectorSize", "virtual-vector-size", "int64_t", /*default=*/"8",
               "Specify a vector size for vectorization">
  ];
}

//===----------------------------------------------------------------------===//
// MathExtSpecicalLower
//===----------------------------------------------------------------------===//
def MathExtLower : Pass<"math-ext-lower", "mlir::ModuleOp"> {
  let summary = "lower MathExt ops to  standard Math/SCF/Arith ops";
  let description = [{
    ...
  }];
  let constructor = "mlir::createMathExtLowerPass()";
}

//===----------------------------------------------------------------------===//
// MathExt Convert to LLVM Libm
//===----------------------------------------------------------------------===//
def MathExtToLibm : Pass<"math-ext-to-libm", "mlir::ModuleOp"> {
  let summary = "lower MathExt ops to  LLVM call ops";
  let description = [{
    ...
  }];
  let constructor = "mlir::createMathExtToLibmPass()";
}


//===----------------------------------------------------------------------===//
// PureOpenMPToLLVM
//===----------------------------------------------------------------------===//
def PureOpenMPToLLVM : Pass<"convert-pure-omp-to-llvm", "mlir::ModuleOp"> {
  let summary = "lower OepnMP dialect to LLVM dialect";
  let description = [{
    This pass only deal with openmp dialect, while MLIR's convert-openmp-to-llvm
    also converts func, arith, memref and other related dialects.
  }];
  let constructor = "mlir::createPureOpenMPToLLVMPass()";
}

#endif // AKG_CONVERSION_PASSES
