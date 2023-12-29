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
#ifndef COMPILER_INCLUDE_AKG_PIPELINES_CPUOPT_H_
#define COMPILER_INCLUDE_AKG_PIPELINES_CPUOPT_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
using namespace mlir;
namespace cl = llvm::cl;
namespace mlir {
struct CpuOptPipelineOptions : public PassPipelineOptions<CpuOptPipelineOptions> {
  Option<bool> cpuFast{*this, "cpu-fast", cl::desc("cpu affine optimize"), cl::init(true)};

  Option<bool> enableParallel{*this, "enable-parallel", cl::desc("cpu enable parallel"), cl::init(true)};

  Option<bool> dynamicShape{*this, "dynamic-shape", cl::desc("cpu dynamic shape"), cl::init(false)};

  Option<bool> saveTemps{*this, "save-temps", cl::desc("Save temporary files"), cl::init(false)};

  Option<std::string> target{*this, "process", cl::desc("the backend info"), cl::init("cpu")};

  Option<std::string> arch{*this, "arch", cl::desc("the cpu architecture, e.g. 'aarch64' or 'x86_64'"),
                           cl::init("aarch64")};

  Option<std::string> feature{*this, "feature", cl::desc("the cpu feature, e.g. 'sse', 'avx' or 'neon'"),
                              cl::init("neon")};
  Option<std::string> jsonFileName{*this, "json-file-name", cl::desc("mindspore json file name"), cl::init("")};

  Option<bool> dynShapeEnablePoly{*this, "dynamic-shape-enable-polytops", cl::desc("cpu dynamic shape enable polytops"),
                                  cl::init(false)};

  Option<bool> cpuOutlining{*this, "cpu-outlining", cl::desc("outline lambda func for external openmp schedule"),
                            cl::init(false)};

  Option<std::string> outliningPlatform{
    *this, "outlining-platform",
    cl::desc("choose the cooperation platform to run outlining kernel, (MindSpore or MLIR)"), cl::init("MLIR")};
 
  Option<int> stage{*this, "stage",
                    cl::desc("the optimizing stage, '1'  (mindspore -> affine) and '2' (affine -> llvm)"), cl::init(1)};
};

void createCpuOptPipeline(OpPassManager &pm, const CpuOptPipelineOptions &options);

inline void registerCpuOptPipeline() {
  PassPipelineRegistration<CpuOptPipelineOptions>("cpu-opt", "Akg Opt Pipeline for CPU", createCpuOptPipeline);
}
}  // namespace mlir
#endif
