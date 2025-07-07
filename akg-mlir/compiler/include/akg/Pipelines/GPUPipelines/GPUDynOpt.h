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

#ifndef COMPILER_INCLUDE_AKG_PIPELINES_GPUPIPELINES_GPUDYNOPT_H_
#define COMPILER_INCLUDE_AKG_PIPELINES_GPUPIPELINES_GPUDYNOPT_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {

struct GPUDynPipelineOptions : public PassPipelineOptions<GPUDynPipelineOptions> {
  Option<std::string> scheduleOption{*this, "scheduleOption", llvm::cl::desc("An option of schedule optimization."),
                                     llvm::cl::init("polySchedule")};

  Option<bool> saveTemps{*this, "save-temps", llvm::cl::desc("Save temporary files"), llvm::cl::init(false)};

  Option<std::string> arch{*this, "arch", llvm::cl::desc("the gpu architecture, e.g. 'sm_70' or 'sm_80'"),
                           llvm::cl::init("sm_70")};
  Option<std::string> jsonFileName{*this, "json-file-name", llvm::cl::desc("mindspore json file name"),
                                   llvm::cl::init("")};
  Option<std::string> tilingMode{
    *this, "tiling-mode",
    llvm::cl::desc("the mode of auto tiling, can be chosen from ['auto', 'static'], mode 'static' will force auto "
                   "tiling to use static tile size for all dynamic shape cases; mode 'auto' will let auto tiling to "
                   "use dynamic tile size for some special dynamic shape cases."),
    llvm::cl::init("auto")};
};

void createGpuDynOptPipeline(OpPassManager &pm, const GPUDynPipelineOptions &options);

inline void registerGpuDynOptPipeline() {
  PassPipelineRegistration<GPUDynPipelineOptions>("gpu-dyn-opt", "Akg Opt Pipeline for GPU Dynamic Shape",
                                                  createGpuDynOptPipeline);
}

}  // namespace mlir

#endif  // COMPILER_INCLUDE_AKG_PIPELINES_GPUPIPELINES_GPUDYNOPT_H_

