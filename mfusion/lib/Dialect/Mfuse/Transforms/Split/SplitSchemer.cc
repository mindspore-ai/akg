/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include "mfusion/Dialect/Mfuse/Transforms/Split/SplitSchemer.h"

#include "mfusion/Analysis/Split/Area.h"
#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"
#include "mlir/IR/Operation.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace mfuse {
namespace split {

//===----------------------------------------------------------------------===//
// SplitSchemer implementation
//===----------------------------------------------------------------------===

bool SplitSchemer::needInline(size_t groupId) const {
  if (groupId >= need_inline_.size()) {
    std::string err_msg = "The group_id " + std::to_string(groupId) + " is out of range of group num " +
                          std::to_string(need_inline_.size());
    llvm::report_fatal_error(llvm::StringRef(err_msg));
  }
  return need_inline_[groupId] != 0;
}

}  // namespace split
}  // namespace mfuse
}  // namespace mlir