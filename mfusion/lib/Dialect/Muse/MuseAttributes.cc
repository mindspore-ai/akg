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

#include "mfusion/Dialect/Muse/Muse.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/raw_ostream.h"
#include "symengine/parser.h"

namespace mlir::muse {

std::vector<SymEngine::RCP<const SymEngine::Basic>> SymbolicShapeAttr::getSymEngineExprs() const {
  std::vector<SymEngine::RCP<const SymEngine::Basic>> exprs;
  auto listAttr = getExprs();
  exprs.reserve(listAttr.size());

  for (auto attr : listAttr) {
    auto strAttr = attr.dyn_cast<mlir::StringAttr>();
    if (!strAttr) {
      llvm::errs() << "muse.symbolic_shape expects StringAttr entries\n";
      continue;
    }
    exprs.push_back(SymEngine::parse(strAttr.getValue().str()));
  }
  return exprs;
}

}  // namespace mlir::muse
