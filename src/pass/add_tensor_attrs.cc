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

namespace akg {
namespace ir {
class TensorAttrsAdder : public IRMutator {
 public:
  explicit TensorAttrsAdder(const Map<Tensor, Map<std::string, NodeRef>> &attrs) {
    for (const auto &kv : attrs) {
      (void)attrs_.emplace(kv.first->op, kv.second);
    }
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    if (attrs_.count(op->func) > 0) {
      return AttrStmt::make(attrs_[op->func], "tensor_attrs", Expr(1), s);
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
    std::unordered_map<FunctionRef, Map<std::string, NodeRef>, NodeHash, NodeEqual>attrs_;
};

Stmt AddTensorAttrs(Stmt stmt, const Map<Tensor, Map<std::string, NodeRef>> &attrs) {
  return TensorAttrsAdder(attrs).Mutate(stmt);
}
}  // namespace ir
}  // namespace akg
