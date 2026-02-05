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

#ifndef MFUSION_ANALYSIS_SYMBOLIC_SHAPE_SYM_EXPR_BUILDER_H
#define MFUSION_ANALYSIS_SYMBOLIC_SHAPE_SYM_EXPR_BUILDER_H

#include <cstdint>
#include <string>

#include "symengine/basic.h"

namespace mfusion {

// SymEngine expression builder focused on constructing SymExpr nodes.
class SymExprBuilder {
 public:
  using SymExpr = SymEngine::RCP<const SymEngine::Basic>;

  // SymEngine symbol construction.
  SymExpr makeSymbol(const std::string &name) const;
  // SymEngine integer construction.
  SymExpr makeInteger(int64_t value) const;
  // SymEngine arithmetic construction helpers.
  SymExpr makeAdd(const SymExpr &lhs, const SymExpr &rhs) const;
  SymExpr makeMul(const SymExpr &lhs, const SymExpr &rhs) const;
  SymExpr makeFloorDiv(const SymExpr &lhs, const SymExpr &rhs) const;
  SymExpr makeCeilDiv(const SymExpr &lhs, const SymExpr &rhs) const;
};

}  // namespace mfusion

#endif  // MFUSION_ANALYSIS_SYMBOLIC_SHAPE_SYM_EXPR_BUILDER_H
