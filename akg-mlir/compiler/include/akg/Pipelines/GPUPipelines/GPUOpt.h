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

namespace cl = llvm::cl;
namespace mlir {

struct GPUPipelineOptions : public PassPipelineOptions<GPUPipelineOptions> {
  Option<std::string> scheduleOption{*this, "scheduleOption", cl::desc("An option of schedule optimization."),
                                     cl::init("polySchedule")};

  Option<bool> saveTemps{*this, "save-temps", cl::desc("Save temporary files"), cl::init(false)};

  Option<std::string> arch{*this, "arch", cl::desc("the gpu architecture, e.g. 'sm_70' or 'sm_80'"), cl::init("sm_70")};
  Option<std::string> jsonFileName{*this, "json-file-name", cl::desc("mindspore json file name"), cl::init("")};
  Option<std::string> globalConfigFile{*this, "global-config-file", cl::desc("tuned repository file path"),
                                       cl::init("")};
  Option<int> stage{*this, "stage",
                    cl::desc("the optimizing stage, '1'  (mindspore -> affine) and '2' (affine -> llvm)"), cl::init(1)};
};

void createGpuOptPipeline(OpPassManager &pm, const GPUPipelineOptions &options);

inline void registerGpuOptPipeline() {
  PassPipelineRegistration<GPUPipelineOptions>("gpu-opt", "Akg Opt Pipeline for GPU", createGpuOptPipeline);
}

}  // namespace mlir

#endif  // COMPILER_INCLUDE_AKG_PIPELINES_GPUPIPELINES_GPUOPT_H_

