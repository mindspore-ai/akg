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

#ifndef COMPILER_INCLUDE_AKG_PIPELINES_GPUPIPELINES_GPUOPT_H_
#define COMPILER_INCLUDE_AKG_PIPELINES_GPUPIPELINES_GPUOPT_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {

struct GPUPipelineOptions : public PassPipelineOptions<GPUPipelineOptions> {
  Option<bool> gpuFast{*this, "gpu-fast", llvm::cl::desc("gpu affine optimize"), llvm::cl::init(false)};

  Option<std::string> scheduleOption{*this, "scheduleOption", llvm::cl::desc("An option of schedule optimization."),
                                     llvm::cl::init("polySchedule")};

  Option<bool> saveTemps{*this, "save-temps", llvm::cl::desc("Save temporary files"), llvm::cl::init(false)};

  Option<std::string> arch{*this, "arch", llvm::cl::desc("the gpu architecture, e.g. 'sm_70' or 'sm_80'"),
                           llvm::cl::init("sm_70")};
  Option<std::string> jsonFileName{*this, "json-file-name", llvm::cl::desc("mindspore json file name"),
                                   llvm::cl::init("")};
  Option<std::string> globalConfigFile{*this, "global-config-file", llvm::cl::desc("tuned repository file path"),
                                       llvm::cl::init("")};
};

void createGpuOptPipeline(OpPassManager &pm, const GPUPipelineOptions &options);

inline void registerGpuOptPipeline() {
  PassPipelineRegistration<GPUPipelineOptions>("gpu-opt", "Akg Opt Pipeline for GPU", createGpuOptPipeline);
}

}  // namespace mlir

#endif  // COMPILER_INCLUDE_AKG_PIPELINES_GPUPIPELINES_GPUOPT_H_

