/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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
#ifndef AKG_PIPELINES_ASCENDPIPELINES_ASCENDOPT_H_
#define AKG_PIPELINES_ASCENDPIPELINES_ASCENDOPT_H_

#include <string>
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {
struct AscendOptPipelineOptions : public PassPipelineOptions<AscendOptPipelineOptions> {
  Option<bool> ascendFast{*this, "ascend-fast", llvm::cl::desc("lower to cpu backend as a stub"), llvm::cl::init(true)};

  Option<bool> enableParallel{*this, "enable-parallel", llvm::cl::desc("ascend enable parallel"), llvm::cl::init(true)};

  Option<bool> dynamicShape{*this, "dynamic-shape", llvm::cl::desc("cpu dynamic shape"), llvm::cl::init(false)};

  Option<bool> saveTemps{*this, "save-temps", llvm::cl::desc("Save temporary files"), llvm::cl::init(false)};

  Option<bool> enableLoopFusion{*this, "enable-loop-fusion", llvm::cl::desc("ascend enable loop fusion"),
                                llvm::cl::init(true)};

  Option<std::string> target{*this, "process", llvm::cl::desc("the backend info"), llvm::cl::init("ascend")};

  Option<std::string> arch{*this, "arch",
                           llvm::cl::desc("the ascend architecture, e.g. 'Ascend910B1' or 'Ascend910_9581'"),
                           llvm::cl::init("Ascend910B1")};

  Option<std::string> jsonFileName{*this, "json-file-name", llvm::cl::desc("mindspore json file name"),
                                   llvm::cl::init("")};
  Option<bool> enableHIVMCompile{*this, "enable-hivm-compile", llvm::cl::desc("Enable HIMV compile pipeline"),
                                 llvm::cl::init(false)};

  Option<bool> enableAutoMultiBuffer{*this, "enable-auto-multi-buffer", llvm::cl::desc("Enable auto multi buffer"),
                                     llvm::cl::init(true)};

  Option<bool> enableGraphSyncSolver{*this, "enable-graph-sync-solver",
                                     llvm::cl::desc("Use hivm graph-sync-solver instead of inject-sync"),
                                     llvm::cl::init(false)};

  Option<bool> enableInjectBarrierAllSync{
    *this, "enable-barrier-all", llvm::cl::desc("Enable barrier all mode for HIVM inject sync"), llvm::cl::init(false)};
  Option<bool> enableAutoInjectSync{*this, "enable-auto-inject-sync",
                                    llvm::cl::desc("Enable auto inject sync for HIVM"), llvm::cl::init(true)};
  Option<bool> enableAutoStorageAlign{*this, "enable-auto-align", llvm::cl::desc("Enable auto align for HIVM"),
                                      llvm::cl::init(true)};

  Option<bool> enableMemoryDisplay{*this, "enable-memory-display",
                                   llvm::cl::desc("Enable ascend memory display tools."), llvm::cl::init(false)};

  Option<unsigned> optLevel{*this, "opt-level", llvm::cl::desc("Optimization level (default: 2)"), llvm::cl::init(2)};
};

void createAscendOptPipeline(OpPassManager &pm, const AscendOptPipelineOptions &options);

inline void registerAscendOptPipeline() {
  PassPipelineRegistration<AscendOptPipelineOptions>("ascend-opt", "Akg Opt Pipeline for Ascend",
                                                     createAscendOptPipeline);
}
}  // namespace mlir
#endif
