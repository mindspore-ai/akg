/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
// rename MatMul to BatchMatMul
class RenameMatmulMutator : public IRMutator {
 public:
  explicit RenameMatmulMutator() {}
  ~RenameMatmulMutator() override = default;

  Stmt Mutate_(const Provide *op, const Stmt &s) {
    auto call = op->value.as<Call>();
    if (call == nullptr || call->name != "MatMul") {
      return IRMutator::Mutate_(op, s);
    }
    return Provide::make(
      op->func, 0, Call::make(op->value.type(), "BatchMatMul", call->args, Call::CallType::PureIntrinsic), op->args);
  }
};

Stmt RenameMatmul(const Stmt &s, BuildInfo*) { return RenameMatmulMutator().Mutate(s); }
}  // namespace akg
