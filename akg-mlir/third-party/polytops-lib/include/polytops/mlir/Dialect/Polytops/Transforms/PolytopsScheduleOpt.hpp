// Copyright 2023-2024 Huawei Technologies Co., Ltd
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef INCLUDE_POLYTOPS_MLIR_DIALECT_POLYTOPS_TRANSFORMS_POLYTOPSSCHEDULEOPT_HPP_
#define INCLUDE_POLYTOPS_MLIR_DIALECT_POLYTOPS_TRANSFORMS_POLYTOPSSCHEDULEOPT_HPP_

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Headers
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassOptions.h>
#include <mlir/Pass/PassRegistry.h>

#include <string>

namespace polytops::mlir {
struct PolytopsSchedulePipelineOptions : public ::mlir::PassPipelineOptions<PolytopsSchedulePipelineOptions> {
  Option<std::string> target{
      *this, "target", llvm::cl::desc("target backend (cpu, gpu, ascend)"), llvm::cl::init("cpu")};
};

void createPolytopsScheduleOptPipeline(::mlir::OpPassManager& pm, const PolytopsSchedulePipelineOptions& options);

inline void registerPolytopsScheduleOptPipeline() {
  ::mlir::PassPipelineRegistration<PolytopsSchedulePipelineOptions>(
      "schedule-opt", "Polytops full pipeline pass", createPolytopsScheduleOptPipeline);
}
}  // namespace polytops::mlir

#endif  //  INCLUDE_POLYTOPS_MLIR_DIALECT_POLYTOPS_TRANSFORMS_POLYTOPSSCHEDULEOPT_HPP_
