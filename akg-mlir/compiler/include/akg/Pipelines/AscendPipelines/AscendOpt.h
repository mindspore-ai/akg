/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#ifndef COMPILER_INCLUDE_AKG_PIPELINES_ASCENDPIPELINES_ASCENDOPT_H_
#define COMPILER_INCLUDE_AKG_PIPELINES_ASCENDPIPELINES_ASCENDOPT_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
using namespace mlir;
namespace cl = llvm::cl;
namespace mlir {
struct AscendOptPipelineOptions : public PassPipelineOptions<AscendOptPipelineOptions> {
  Option<bool> ascendFast{*this, "ascend-fast", cl::desc("lower to cpu backend as a stub"), cl::init(true)};

  Option<bool> enableParallel{*this, "enable-parallel", cl::desc("ascend enable parallel"), cl::init(true)};

  Option<bool> useBishengPipeline{*this, "use-bisheng-pipeline", cl::desc("use bisheng pipeline"), cl::init(true)};

  Option<bool> dynamicShape{*this, "dynamic-shape", cl::desc("cpu dynamic shape"), cl::init(false)};

  Option<bool> saveTemps{*this, "save-temps", cl::desc("Save temporary files"), cl::init(false)};

  Option<bool> enableAKGLoopFusion{*this, "enable-akg-loop-fusion", cl::desc("ascend enable akg loop fusion"), cl::init(false)};

  Option<std::string> target{*this, "process", cl::desc("the backend info"), cl::init("ascend")};

  Option<std::string> arch{*this, "arch", cl::desc("the ascend architecture, e.g. '910A' or '910B'"), cl::init("910B")};

  Option<std::string> jsonFileName{*this, "json-file-name", cl::desc("mindspore json file name"), cl::init("")};
};

void createAscendOptPipeline(OpPassManager &pm, const AscendOptPipelineOptions &options);

inline void registerAscendOptPipeline() {
  PassPipelineRegistration<AscendOptPipelineOptions>("ascend-opt", "Akg Opt Pipeline for Ascend",
                                                     createAscendOptPipeline);
}
}  // namespace mlir
#endif