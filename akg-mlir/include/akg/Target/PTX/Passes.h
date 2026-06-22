/**
 * Copyright 2023-2026 Huawei Technologies Co., Ltd
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

#ifndef AKG_TARGET_PTX_PASSES_H_
#define AKG_TARGET_PTX_PASSES_H_

#include <string>
#include <memory>
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

/// Convert kernel functions in GPU dialect to PTX
std::unique_ptr<OperationPass<gpu::GPUModuleOp>> createSerializeToPTXPass(
  unsigned optLevel, const std::string &libdeviceFile, const std::string &triple, const std::string &targetChip,
  const std::string &features, std::string &ptx);

}  // namespace mlir

#endif  // AKG_TARGET_PTX_PASSES_H_
