/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#ifndef PASS_ANALYZE_ALIGN_H_
#define PASS_ANALYZE_ALIGN_H_

#include <tvm/ir.h>
#include <tvm/ir_mutator.h>

#include <string>
#include <set>

#include "pass/utils.h"
#include "arith_expr_simplify.h"
#include "expr_alg_simplify.h"

namespace akg {
namespace ir {
const std::set<std::string> exclude_list = {
  "mad",
  "scatter",
  "vec_binary_proposal_sort",
  "vec_binary_topk_sort",
  "vec_binary_nms",
  "vec_binary_iou",
  "vec_binary_dropout",
  "vec_single_four2five_nchw",
  "opt_broadcast",
  "reduce_reorder",
};
class IndexOptimizer : public air::ir::IRMutator {
 public:
  explicit IndexOptimizer(bool rm = false) : var2expr(), rm_load_(rm) {}
  ~IndexOptimizer() override = default;

#define MUTATE_OP(OP)                               \
  Expr Mutate_(const OP *op, const Expr &e) final { \
    Var v("tmp");                                   \
    var2expr.Set(v, e);                             \
    return v;                                       \
  }
  MUTATE_OP(Div)
  MUTATE_OP(Mod)
  MUTATE_OP(FloorDiv)
  MUTATE_OP(FloorMod)
#undef MUTATE_OP

  Expr Mutate_(const Load *op, const Expr &e) final {
    if (rm_load_) {
      Var v("tmp");
      var2expr.Set(v, e);
      return v;
    }
    return e;
  }

  Map<Var, Expr> var2expr;

 private:
  bool rm_load_;
};
}  // namespace ir
}  // namespace akg
#endif  // PASS_ANALYZE_ALIGN_H_
