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
#ifndef PROMOTE_DATA_TYPE_H_
#define PROMOTE_DATA_TYPE_H_

#include "tvm.h"
#include <tvm/arithmetic.h>

namespace akg {
namespace ir {
class OverflowChecker : public IRVisitor {
 public:
  explicit OverflowChecker() {}
  ~OverflowChecker() override = default;
  void Visit_(const Store *op) override;
  void Visit_(const Load *op) override;
  void Visit_(const Mul *op) override;
  void Visit_(const Add *op) override;
  void Visit_(const AttrStmt *op) override;
  void Visit(const NodeRef& node) final;
  bool need_promote_int64{false};
  const Variable *var_to_replace{nullptr};

 private:
  size_t block_extent_{0};
};

Stmt PromoteIndexDataType(Stmt stmt);
}  // namespace ir
}  // namespace akg

#endif  // PROMOTE_DATA_TYPE_H_