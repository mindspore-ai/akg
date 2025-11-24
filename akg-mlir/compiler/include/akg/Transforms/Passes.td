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

#ifndef AKG_MLIR_TRANSFORMS_PASSES
#define AKG_MLIR_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def AKGOperatorIdentify : Pass<"akg-operator-identify", "func::FuncOp"> {
  let summary = "Identify all operator type templates.";
  let constructor = "mlir::createAKGOperatorIdentifyPass()";
}

def FoldDimension : Pass<"fold-dimension", "func::FuncOp"> {
  let summary = "Fold consecutive dimensions together if requirements satisfied.";
  let constructor = "mlir::createFoldDimensionPass()";
}

def CopyRemoval : Pass<"copy-removal"> {
  let summary = "Remove redundant copy operations";
  let constructor = "mlir::createCopyRemovalPass()";
}

def CopyElision : Pass<"copy-elision"> {
  let summary = "Remove unecessary copy operations";
  let description = [{
    Remove unecessary copy operations
  }];
  let constructor = "mlir::createCopyElisionPass()";
}

def InferSymbolicShapes : Pass<"infer-symbolic-shapes", "func::FuncOp"> {
  let summary = "Infer symbolic shapes info across mlir operations";
  let description = [{
    Infer symbolic shape expression according to the operator's semantics, and
    propagate those symbolic info between ops.
  }];

  let constructor = "mlir::createInferSymbolicShapesPass()";
}

def RemoveSymbolic : Pass<"remove-symbolic"> {
  let summary = "Remove symbolic attribute from Tensor Type";
  let constructor = "mlir::createSymbolicRemovalPass()";
}

def SymbolRemoval : Pass<"symbol-removal"> {
  let summary = "Remove symbol after polytops";
  let constructor = "mlir::createSymbolRemovalPass()";
}

def StoreLoadElim : Pass<"elim-store-load"> {
  let summary = "Remove redundant store-and-immediate load pair";
  let constructor = "mlir::createStoreLoadElimPass()";
}

def PromoteTempBuffer : Pass<"promote-temp-buffer"> {
  let summary = "Promote temp buffer to gpu's workgroup/private address";
  let constructor = "mlir::createPromoteTempBufferPass()";
}

def CopyAttributesToGpu : Pass<"copy-attributes-to-gpu"> {
  let summary = "This pass will copy all the attributes of funcOp to GPUFuncOp";
  let constructor = "mlir::createCopyAttributesToGpuPass()";
}

def DumpShapeInfo : Pass<"dump-shape-info"> {
  let summary = "This pass will dump the global shape info for dynamic pipeline";
  let constructor = "mlir::createDumpShapeInfoPass()";
  let options = [
    Option<"fileName", "file-name", "std::string", /*default=*/"",
           "mindspore json file name.">,
  ]; 
}

def LoadGlobalConfig : Pass<"load-global-config"> {
  let summary = "This pass will load the global configs";
  let constructor = "mlir::createLoadGlobalConfigPass()";
  let options = [
    Option<"fileName", "file-name", "std::string", /*default=*/"",
           "tuned repository file path.">,
  ];
}

def AKGFuncOutlining : Pass<"akg-func-outlining", "mlir::ModuleOp"> {
  let summary = "usr for CPU func outlining  ";
  let constructor = "mlir::createAKGFuncOutliningPass()";
}

def AKGParallelLaunch : Pass<"akg-parallel-launch","mlir::ModuleOp"> {
  let summary = "add support parallel launch callback for Mindspore runtimes";
  let constructor = "mlir::createAKGParallelLaunchPass()";
}

#endif // AKG_MLIR_TRANSFORMS_PASSES
