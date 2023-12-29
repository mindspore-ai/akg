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

#ifndef COMPILER_INCLUDE_AKG_PIPELINES_GPUPIPELINES_GPUCODEGEN_H_
#define COMPILER_INCLUDE_AKG_PIPELINES_GPUPIPELINES_GPUCODEGEN_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {

struct GPUCodegenPipelineOptions : public PassPipelineOptions<GPUCodegenPipelineOptions> {
  Option<std::string> arch{*this, "arch", llvm::cl::desc("the gpu architecture, e.g. 'sm_70' or 'sm_80'"),
                           llvm::cl::init("sm_70")};
};

void createGpuCodegenPipeline(OpPassManager &pm, const GPUCodegenPipelineOptions &options);

inline void registerGpuCodegenPipeline() {
  PassPipelineRegistration<GPUCodegenPipelineOptions>("gpu-codegen", "Codegen Pipeline for GPU",
                                                      createGpuCodegenPipeline);
}
}  // namespace mlir

#endif  // COMPILER_INCLUDE_AKG_PIPELINES_GPUPIPELINES_GPUCODEGEN_H_

