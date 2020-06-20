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
#include <tvm/base.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_visitor.h>
#include <tvm/api_registry.h>
#include <tvm/operation.h>
#include <tvm/expr.h>
#include <ir_pass.h>
#include <arithmetic/pattern_match.h>
#include <floating.h>
#include <functional>
#include <unordered_map>

#include "pass/utils.h"

namespace akg {
namespace ir {
// remove empty realize
class CleanRealize : public IRMutator {
 public:
  Stmt Mutate_(const Realize *op, const Stmt &s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    if (usedef_.count(op->func.get()) == 0) {
      op = stmt.as<Realize>();
      CHECK(op);
      return op->body;
    }
    return stmt;
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    usedef_.insert(op->func.get());
    return IRMutator::Mutate_(op, s);
  }

  Expr Mutate_(const Call *op, const Expr &e) final {
    if (op->call_type == Call::Halide) {
      usedef_.insert(op->func.get());
    }
    return IRMutator::Mutate_(op, e);
  }

 private:
  std::unordered_set<const Node *> usedef_;
};

// Context information for match function and replace function
struct Context {
  std::unordered_map<const Variable *, Range> &dom_map;  // domain map for loop axis
  std::unordered_map<Tensor, size_t> &count_map;         // count the number of appearance of a tensor

  // Add more fields if more context info is required
};

struct RewritePattern {
  // return whether the statement matches this pattern
  std::function<bool(Stmt, Context *)> match_func;
  // return the replaced results
  std::function<Stmt(Stmt, Context *)> replace_func;
};

class CallCounter : public IRVisitor {
 public:
  void Visit_(const Call *op) final {
    if (op->call_type == Call::CallType::Halide) {
      Tensor t = Downcast<ktvm::Operation>(op->func).output(op->value_index);
      count_map_[t]++;
    }
    IRVisitor::Visit_(op);
  }
  explicit CallCounter(std::unordered_map<Tensor, size_t> &count_map) : count_map_(count_map) {}
  ~CallCounter() override = default;
  std::unordered_map<Tensor, size_t> &count_map_;
};

class StatementPatternRewriter : public IRMutator {
 public:
  explicit StatementPatternRewriter(const std::vector<RewritePattern> &patterns) : patterns_(patterns) {}
  ~StatementPatternRewriter() override = default;

  Stmt Rewrite(Stmt stmt) {
    count_map_.clear();
    dom_map_.clear();

    // build context information
    CallCounter(count_map_).Visit(stmt);

    return Mutate(stmt);
  }

  Stmt Mutate(Stmt stmt) override {
    Context ctx{dom_map_, count_map_};
    bool modified = true;

    while (modified) {  // recursively apply rules
      modified = false;
      for (size_t i = 0; i < patterns_.size(); ++i) {
        if (patterns_[i].match_func(stmt, &ctx)) {
          stmt = patterns_[i].replace_func(stmt, &ctx);
          modified = true;
        }
      }
    }

    return IRMutator::Mutate(stmt);
  }

  Stmt Mutate_(const For *op, const Stmt &stmt) override {
    dom_map_[op->loop_var.get()] = Range::make_by_min_extent(op->min, op->extent);
    Stmt ret = IRMutator::Mutate_(op, stmt);
    dom_map_.erase(op->loop_var.get());
    return ret;
  }

 private:
  const std::vector<RewritePattern> &patterns_;
  std::unordered_map<const Variable *, Range> dom_map_;
  std::unordered_map<Tensor, size_t> count_map_;
};

template <typename T>
ktvm::arith::PConst<Expr> fold(T expr) {
  return ktvm::arith::PConst<Expr>(Simplify_cce(expr.Eval()));
}

#define TRY_REWRITE(before, after)                                                      \
  RewritePattern{[&](Stmt stmt, Context *ctx) -> bool { return (before).Match(stmt); }, \
                 [&](Stmt stmt, Context *ctx) -> Stmt { return (after).Eval(); }},

#define TRY_REWRITE_IF(before, after, condition)                                                     \
  RewritePattern{[&](Stmt stmt, Context *ctx) -> bool { return (before).Match(stmt) && condition; }, \
                 [&](Stmt stmt, Context *ctx) -> Stmt { return (after).Eval(); }},

Stmt StmtPatternRewrite(Stmt stmt) {
  ktvm::arith::PVar<Floating> f1, f2, f3;
  ktvm::arith::PVar<Array<Expr> > i1, i2, i3, i4;
  ktvm::arith::PTensor A, B, C, D, in, out;
  ktvm::arith::PVar<Expr> value1, value2, cond;
  ktvm::arith::PVar<Expr> min1, extent1, min2, extent2, min3, extent3, min4, extent4, min5, extent5;
  ktvm::arith::PVar<Stmt> other, then_body, else_body;
  ktvm::arith::PConst<Floating> zero(0.0f);
  ktvm::arith::PVar<Var> n, c1, h, w, c0;
  ktvm::arith::PVar<NodeRef> useless;

  std::vector<RewritePattern> patterns{
    // A[i] = 0
    // A[i] = A[i] + B[j]
    // ->
    // A[i] = B[j]
    TRY_REWRITE(STMT_LIST(PROVIDE(A[i1], zero), PROVIDE(A[i1], A[i1] + B[i2]), other),
                STMT_LIST(PROVIDE(A[i1], B[i2]), other))

    // A[i] = B[i] + 0
    // -->
    // A[i] = B[i]
    TRY_REWRITE(STMT_LIST(PROVIDE(A[i1], B[i2] + zero), other), STMT_LIST(PROVIDE(A[i1], B[i2]), other))

    // B[j] = A[i] * 2
    // C[k] = B[j] * 4
    // ->
    // C[k] = A[i] * 8
    TRY_REWRITE_IF(STMT_LIST(PROVIDE(B[i2], A[i1] * f1), PROVIDE(C[i3], B[i2] * f2), other),
                   STMT_LIST(PROVIDE(C[i3], A[i1] * fold(f1 * f2)), other),
                   (ctx->count_map[B.Eval()] == 1 && LimitCheck<Floating>(f1, f2)))

    // B[j] = A[i] * 2
    // C[k] = B[j] + 1
    // D[l] = C[k] * 16
    // ->
    // B[i] = A[i] * 32
    // D[l] = B[i] + 16
    TRY_REWRITE_IF(STMT_LIST(PROVIDE(B[i2], A[i1] * f1), PROVIDE(C[i3], B[i2] + f2), PROVIDE(D[i3], C[i2] * f3), other),
                   STMT_LIST(PROVIDE(B[i2], A[i1] * fold(f1 * f3)), PROVIDE(D[i3], B[i2] + fold(f2 * f3)), other),
                   // only appear once, so we can safely re-assign it
                   (ctx->count_map[B.Eval()] == 1 && ctx->count_map[C.Eval()] == 1 && LimitCheck<Floating>(f1, f3) &&
                    LimitCheck<Floating>(f2, f3)))

    // if (cond)
    //   body1
    // else
    //   body2
    // ->
    // if (cond)
    //   body1
    // if (!cond)
    //   body2
    /*
        TRY_REWRITE(
            STMT_LIST( IF_THEN_ELSE(cond, then_body, else_body),
                       other),
            // ->
            STMT_LIST( IF_THEN(cond, then_body),
                       IF_THEN(!cond, else_body),
                       other)
        )
    */

    // expand c0 (see notes below)
    TRY_REWRITE(
      STMT_LIST(ATTR_STMT(useless, "pragma_emit_insn", value1,
                          FOR(n, min1, extent1,
                              FOR(c1, min2, extent2,
                                  FOR(h, min3, extent3,
                                      FOR(w, min4, extent4,
                                          FOR(c0, min5, extent5,
                                              PROVIDE_SEP(out, in(n, 2 * c1 + truncdiv(c0, 16), h, w, truncmod(c0, 16)),
                                                          n, c1, h, w, c0))))))),
                other),
      // ->
      STMT_LIST(
        ATTR_STMT(useless, "pragma_emit_insn", value1,
                  FOR(n, min1, extent1,
                      FOR(c1, min2, extent2,
                          FOR(h, min3, extent3,
                              FOR(w, min4, extent4,
                                  FOR(c0, min5, truncdiv(extent5, 2),
                                      PROVIDE_SEP(out, in(n, 2 * c1, h, w, truncmod(c0, 16)), n, c1, h, w, c0))))))),
        ATTR_STMT(
          useless, "pragma_emit_insn", value1,
          FOR(n, min1, extent1,
              FOR(c1, min2, extent2,
                  FOR(h, min3, extent3,
                      FOR(w, min4, extent4,
                          FOR(c0, min5, truncdiv(extent5, 2),
                              PROVIDE_SEP(out, in(n, 2 * c1 + 1, h, w, truncmod(c0, 16)), n, c1, h, w, c0 + 16))))))),
        other))};

  StatementPatternRewriter rewriter(patterns);
  stmt = rewriter.Rewrite(stmt);
  return CleanRealize().Mutate(stmt);
}

/* Expand C0
 *
 *  for (n, 0, 32) {
 *    for (c1, 0, 2) {
 *      for (h, 0, 33) {
 *        for (w, 0, 33) {
 *          for (c0, 0, 32) {
 *            output[n, c1, h, w, c0] = input[n, 2 * c1 + c0 / 16, h, w, c0 % 16]
 *          }
 *        }
 *      }
 *    }
 *  }
 *
 *  Transform to:
 *
 *  for (n, 0, 32) {
 *    for (c1, 0, 2) {
 *      for (h, 0, 33) {
 *        for (w, 0, 33) {
 *          for (c0, 0, 16) {
 *            output[n, c1, h, w, c0] = input[n, 2 * c1, h, w, c0]
 *          }
 *        }
 *      }
 *    }
 *  }
 *  for (n, 0, 32) {
 *    for (c1, 0, 2) {
 *      for (h, 0, 33) {
 *        for (w, 0, 33) {
 *          for (c0, 0, 16) {
 *            output[n, c1, h, w, c0 + 16] = input[n, 2 * c1 + 1, h, w, c0]
 *          }
 *        }
 *      }
 *    }
 *  }
 *
 */
}  // namespace ir
}  // namespace akg
