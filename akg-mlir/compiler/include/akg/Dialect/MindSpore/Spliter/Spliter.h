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
#ifndef COMPILER_INCLUDE_AKG_DIALECT_MINDSPORE_SPLITER_SPLITER_H_
#define COMPILER_INCLUDE_AKG_DIALECT_MINDSPORE_SPLITER_SPLITER_H_
#include "akg/Dialect/MindSpore/Spliter/Utils.h"
namespace mlir::spliter {
using OpArea = std::vector<Operation *>;
using ValueSeq = std::vector<Value>;
using TypeSeq = std::vector<Type>;
llvm::SmallVector<func::FuncOp> split(func::FuncOp funcOp);
func::FuncOp splitArea(func::FuncOp func, const OpArea &area, OpBuilder &builder);
}  // namespace mlir::spliter

#endif  // COMPILER_INCLUDE_AKG_DIALECT_MINDSPORE_SPLITER_SPLITER_H_
