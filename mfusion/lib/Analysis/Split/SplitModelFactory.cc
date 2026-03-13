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

#include "mfusion/Analysis/Split/SplitModelFactory.h"

#include "mfusion/Analysis/Split/SplitModelInitPattern.h"
#include "mfusion/Dialect/Mfuse/Support/OpConstants.h"

namespace mlir {
namespace mfuse {
namespace split {

std::shared_ptr<SplitModel> SplitModelFactory::createSplitModel(const std::string &processor) {
  auto it = creators_.find(processor);
  if (it != creators_.end()) {
    return it->second();
  }
  std::string err_msg = "No split model creator found for processor: " + processor;
  llvm::report_fatal_error(llvm::StringRef(err_msg));
}

// Register DVMSplitModel for "dvm" processor
SPLIT_MODEL_REGISTER(kProcessorDVM, DVMSplitModel);
// Register AKGSplitModel for "akg" processor
SPLIT_MODEL_REGISTER(kProcessorAKG, AKGSplitModel);
// Register BishengSplitModel for "bisheng" processor
SPLIT_MODEL_REGISTER(kProcessorBisheng, BishengSplitModel);

}  // namespace split
}  // namespace mfuse
}  // namespace mlir
