/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "tvm.h"
#include "codegen/util.h"

namespace akg {
namespace ir {
/* Add attr passed from composite for a specific tensor
 * === Example 1 ===
 * for (ax0, 0, 16)
 *   for (a10, 0, 16)
 *     T(ax0, ax1) = input0(ax0, ax1)
 * -->
 *  for (ax0, 0, 16)
 *   for (a10, 0, 16)
 *     // attr [{"enable_auto_inplace": (int)1}] attrs = 1
 *     T(ax0, ax1) = input0(ax0, ax1)
 */
class AddTensorAttrsMutator : public IRMutator {
 public:
  explicit AddTensorAttrsMutator(const Map<Tensor, Map<std::string, NodeRef>> &attrs) {
    for (const auto &kv : attrs) {
      (void)attrs_.emplace(kv.first->op, kv.second);
    }
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    if (attrs_.count(op->func) > 0) {
      return AttrStmt::make(attrs_[op->func], kTensorAttrs, Expr(1), s);
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  std::unordered_map<FunctionRef, Map<std::string, NodeRef>, NodeHash, NodeEqual> attrs_;
};

Stmt AddTensorAttrs(const Stmt &stmt, const Map<Tensor, Map<std::string, NodeRef>> &attrs) {
  AddTensorAttrsMutator attr_adder(attrs);
  Stmt s = attr_adder.Mutate(stmt);
  return s;
}
}  // namespace ir
}  // namespace akg
