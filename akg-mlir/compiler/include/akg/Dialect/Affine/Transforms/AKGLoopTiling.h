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

#ifndef COMPILER_INCLUDE_AKG_DIALECT_AFFINE_TRANSFORMS_AKGLOOPTILING_H_
#define COMPILER_INCLUDE_AKG_DIALECT_AFFINE_TRANSFORMS_AKGLOOPTILING_H_

#include <memory>
#include <string>
#include "llvm/ADT/SmallVector.h"
#include "mlir/Pass/Pass.h"

using llvm::SmallVector;
using llvm::SmallVectorImpl;

namespace mlir {
namespace func {
class FuncOp;
}  // namespace func

std::unique_ptr<OperationPass<func::FuncOp>> createAKGLoopTilingPass();
std::unique_ptr<OperationPass<func::FuncOp>> createAKGLoopTilingPass(const std::string &target, bool useAutoTiling);
std::unique_ptr<OperationPass<func::FuncOp>> createAKGLoopTilingPass(const std::string &target, bool useAutoTiling,
                                                                     const std::string &tilingMode);
std::unique_ptr<OperationPass<func::FuncOp>> createAKGLoopTilingPass(const std::string &target,
                                                                     const std::string &feature, bool useAutoTiling);
std::unique_ptr<OperationPass<func::FuncOp>> createAKGLoopTilingPass(const std::string &target, bool useAutoTiling,
                                                                     const std::string &arch,
                                                                     const std::string &feature);
std::unique_ptr<OperationPass<func::FuncOp>> createAKGLoopTilingPass(const std::string &target, bool useAutoTiling,
                                                                     const std::string &arch,
                                                                     const std::string &feature,
                                                                     const SmallVector<unsigned, 6> &inputTileSizes);
std::unique_ptr<OperationPass<func::FuncOp>> createAKGLoopTilingPass(uint64_t cacheSizeBytes);

}  // namespace mlir

#endif  // COMPILER_INCLUDE_AKG_DIALECT_AFFINE_TRANSFORMS_AKGLOOPTILING_H_
