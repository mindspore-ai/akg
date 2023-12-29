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

#ifndef COMPILER_INCLUDE_AKG_DIALECT_AFFINE_TRANSFORMS_AKGVECTORIZE_H_
#define COMPILER_INCLUDE_AKG_DIALECT_AFFINE_TRANSFORMS_AKGVECTORIZE_H_

#include <memory>
#include <string>
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace func {
class FuncOp;
}  // namespace func

std::unique_ptr<OperationPass<func::FuncOp>> createAKGVectorizePass();
std::unique_ptr<OperationPass<func::FuncOp>> createAKGVectorizePass(const std::string &target,
                                                                    const std::string &feature);

}  // namespace mlir

#endif  // COMPILER_INCLUDE_AKG_DIALECT_AFFINE_TRANSFORMS_AKGVECTORIZE_H_

