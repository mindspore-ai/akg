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

#ifndef AKG_MLIR_DIALECT_GPU_PASSES
#define AKG_MLIR_DIALECT_GPU_PASSES

include "mlir/Pass/PassBase.td"

def AKGGPUMapping
    : Pass<"akg-gpu-map-parallel-loops", "mlir::func::FuncOp"> {
  let summary = "Map loops to GPU hardware dimensions.";
  let constructor = "mlir::createAKGGPUMapping()";
  let dependentDialects = [
    "mlir::gpu::GPUDialect"
  ];  
  let description = "Map loops to GPU hardware dimensions using a heuristic algorithm.";
  let dependentDialects = ["mlir::gpu::GPUDialect"];
  let options = [
    Option<"singleProcess", "single-process", "bool", /*default=*/"false",
        "Whether to use single process (default: false)">,
  ];

}

def GpuUseAllReduceWithAtomicReturn
    : Pass<"gpu-use-all-reduce-with-atomic-return", "mlir::func::FuncOp"> {
  let summary = "match gpu reduction pattern, enable gpu.all_reduce op and support atomic return";
  let constructor = "mlir::createGpuUseAllReduceWithAtomicReturnPass()";
  let dependentDialects = [
    "mlir::gpu::GPUDialect", "scf::SCFDialect"
  ];   
  let description = "Replace gpu reduce ops to gpu.all_reduce. support atomic return if this reduction op is a block-cross op";
}

def GetOrderMapBeforeAfterGpuOutlining
    : Pass<"get-order-map-before-after-gpu-outlining", "mlir::func::FuncOp"> {
  let summary = "Get a order map between gpu-kernel-outlining pass";
  let constructor = "mlir::createGetOrderMapBeforeAfterGpuOutliningPass()";
  let options = [
    Option<"path", "path", "std::string", /*default=*/"\"order_map.txt\"",
           "The text file to store mapping order.">,
  ];  
  let dependentDialects = [
    "mlir::gpu::GPUDialect"
  ];   
  let description = "Get a order map between gpu-kernel-outlining pass";
}

def GpuKernelOutliningExt : Pass<"gpu-kernel-outlining-ext", "ModuleOp"> {
  let summary = "Outline gpu.launch bodies to kernel functions(ext version)";
  let constructor = "mlir::createGpuKernelOutliningExt()";
  let dependentDialects = ["mlir::DLTIDialect"];
}

def StoreAxisInfo : Pass<"store-axis-info", "mlir::func::FuncOp"> {
  let summary = "Convert axis info from mlir attributes and store to global var";
  let constructor = "mlir::createStoreAxisInfoPass()";
}

def LoadAxisInfo : Pass<"load-axis-info", "mlir::func::FuncOp"> {
  let summary = "Load axis info from global var and apply to mlir attributes";
  let constructor = "mlir::createLoadAxisInfoPass()";
}

#endif // AKG_MLIR_DIALECT_GPU_PASSES
