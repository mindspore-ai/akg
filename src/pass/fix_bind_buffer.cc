/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "pass/utils.h"

namespace akg {
namespace ir {
/*
 * TVM may generate different instances of loop variable with a same name,
 * leading to ambiguity in later passes.
 * This pass unifies the loop variables according to the name in definition,
 * and reports error for undefined variables.
 */

Stmt FixBindBuffer(const Stmt &stmt, const Map<Tensor, Buffer> &extern_buffer) {
  Stmt s = stmt;
  for (const auto &it : extern_buffer) {
    s = TensorStringSubstitute(s, it.second->name, it.first->op, it.first->value_index);
  }
  return s;
}
}  // namespace ir
}  // namespace akg
