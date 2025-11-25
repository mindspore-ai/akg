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

#ifndef COMPILER_INCLUDE_AKG_TARGET_PTX_TOPTX_H_
#define COMPILER_INCLUDE_AKG_TARGET_PTX_TOPTX_H_

#include <stack>
#include <string>
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

namespace mlir {

void registerToPTXTranslation();
LogicalResult translateToPTX(Operation *op, raw_ostream &os, const std::string &kernelName,
                             const std::string &arch = "sm_70");

}  // namespace mlir

#endif  // COMPILER_INCLUDE_AKG_TARGET_PTX_TOPTX_H_

