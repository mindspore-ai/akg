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
#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <ir_pass.h>
#include <pass/ir_util.h>
#include <algorithm>
#include "pass/utils.h"

namespace akg {
namespace ir {
// we need to analyze which loop can be vectorized and which can not?
// can: loop_var only referenced by just one Provide/Store
// can not: loop_var shared by multiple Provide/Store
// when analyzing Provide/Store, we need to Recognize which this Provide/Store should be marked

// for those have been marked, we just ignore it
Stmt VectorizeFor::Mutate_(const AttrStmt *op, const Stmt &s) {
  if (op->attr_key == "pragma_emit_insn" || op->attr_key == "pragma_im2col" || op->attr_key == "pragma_fractal" ||
      op->attr_key == "pragma_filter" || op->attr_key == "pragma_ub_gm") {
    in_pragma_ = true;
    Stmt stmt = IRMutator::Mutate_(op, s);
    in_pragma_ = false;
    return stmt;
  }
  return IRMutator::Mutate_(op, s);
}

Stmt VectorizeFor::Mutate_(const Evaluate *op, const Stmt &s) {
  if (Equal(op->value, Expr(0))) {
    return s;
  }
  provide_store++;
  return s;
}
Stmt VectorizeFor::Mutate_(const Provide *op, const Stmt &s) {
  provide_store++;
  in_provide_store = true;
  cur_provide_store = op;
  Stmt stmt = IRMutator::Mutate_(op, s);
  in_provide_store = false;
  return stmt;
}

Stmt VectorizeFor::Mutate_(const Store *op, const Stmt &s) {
  provide_store++;
  in_provide_store = true;
  cur_provide_store = op;
  Stmt stmt = IRMutator::Mutate_(op, s);
  in_provide_store = false;
  return stmt;
}

Stmt VectorizeFor::Mutate_(const For *op, const Stmt &s) {
  int backup = provide_store;
  Stmt stmt = IRMutator::Mutate_(op, s);
  const Variable *var = op->loop_var.get();
  // this is a vectorized for
  // for(cc1, 0, 2) { --> cc1 can't be vectorized
  //   for(cc2, 0, 4)
  //    A_local_UB[cc2] = A[(((cc1*16) + cc2) + 16)]
  //   for (cc2, 0, 16) {
  //     reduce[cc2] = (reduce[cc2] + A_local_UB[cc2])
  //   }
  // }
  if (!in_pragma_ && var && var_in_provide_store.count(var) && var_in_provide_store[var].size() == 1 &&
      provide_store - backup == 1) {
    const For *n = stmt.as<For>();
    // clean
    var_in_provide_store.erase(var);
    CHECK(n);
    return For::make(n->loop_var, n->min, n->extent, ForType::Vectorized, n->device_api, n->body);
  }
  // should not keep var in map
  if (var && var_in_provide_store.count(var)) {
    var_in_provide_store.erase(var);
  }
  return stmt;
}

Expr VectorizeFor::Mutate_(const Variable *op, const Expr &e) {
  if (in_provide_store && cur_provide_store != nullptr) {
    if (var_in_provide_store.count(op)) {
      var_in_provide_store[op].insert(cur_provide_store);
    } else {
      std::unordered_set<const Node *> provide_store_tmp;
      provide_store_tmp.insert(cur_provide_store);
      var_in_provide_store[op] = provide_store_tmp;
    }
  }
  return IRMutator::Mutate_(op, e);
}
}  // namespace ir
}  // namespace akg
