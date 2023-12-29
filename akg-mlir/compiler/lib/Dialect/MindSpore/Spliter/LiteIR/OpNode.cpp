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
#include "akg/Dialect/MindSpore/Spliter/OpNode.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include "akg/Dialect/MindSpore/Spliter/Node.h"

namespace mlir::spliter {
std::string PrimOp::toString() const {
  std::ostringstream oss;
  oss << Node::toString();
  oss << " = " << this->op << "(";
  for (size_t i = 0; i < inputs.size(); i++) {
    if (inputs[i]->nodeType() == NType::Primitive) {
      oss << inputs[i]->Node::toString();
    } else {
      oss << inputs[i]->toString();
    }
    if (i != inputs.size() - 1) {
      oss << ", ";
    }
  }
  oss << ")";
  return oss.str();
}
}  // namespace mlir::spliter
