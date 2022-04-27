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

#include "composite/optimize/pass.h"

namespace akg {
class LogicalOrToAddMutator : public IRMutator {
 public:
  LogicalOrToAddMutator() = default;
  ~LogicalOrToAddMutator() override = default;

 private:
  Expr Mutate_(const Call *op, const Expr &e) final {
    if (op->name == "LogicalOr") {
      return Call::make(op->type, "Add", op->args, Call::CallType::PureIntrinsic);
    }
    return IRMutator::Mutate_(op, e);
  }
};

Stmt LogicalOrToAdd(const Stmt &s, BuildInfo *) { return LogicalOrToAddMutator().Mutate(s); }
}  // namespace akg
