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

#include "akg/Dialect/Affine/Analysis/Config.h"

#include <limits>
#include <numeric>

namespace mlir {
namespace akg {
namespace autotiling {

void Config::mergeConstraints() {
  for (auto cons : constraints) {
    this->finalConstraint.max = std::min<int>(this->finalConstraint.max, cons.max);
    this->finalConstraint.step = std::lcm(this->finalConstraint.step, cons.step);
    this->finalConstraint.min = std::max<int>(this->finalConstraint.min, cons.min);
    this->finalConstraint.min = std::max<int>(this->finalConstraint.min, this->finalConstraint.step);
    for (auto cand : cons.candidates) {
      this->finalConstraint.candidates.push_back(cand);
    }
  }
}

std::vector<int> Config::getValidCandidates() {
  if (!this->finalConstraint.candidates.empty()) {
    return this->finalConstraint.candidates;
  }
  // If expr becomes complex in later version, we can solve expr constraints here.
  for (int cand = this->finalConstraint.min; cand <= this->finalConstraint.max; cand += this->finalConstraint.step) {
    validCandidates.push_back(cand);
  }
  return validCandidates;
}
}  // namespace autotiling
}  // namespace akg
}  // namespace mlir

