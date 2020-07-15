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
#include <tvm/tensor.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_visitor.h>
#include <tvm.h>
#include <arithmetic/pattern_match.h>
#include <queue>

namespace akg {
namespace ir {
// forward declaration
class ExprMutator;

struct InstructionPattern {
  std::function<int(Expr, const std::unordered_map<Expr, int, air::NodeHash, air::NodeEqual> &)>
    score_func;                                           // assign score to a subtree
  std::function<Expr(Expr, ExprMutator &)> replace_func;  // replace a subtree with this instruction
};

// Mutate expression according to selection choices
class ExprMutator : public IRMutator {
 public:
  ExprMutator(const std::vector<InstructionPattern> &ins_pattern,
              const std::unordered_map<Expr, int, air::NodeHash, air::NodeEqual> &choice_map)
      : ins_pattern_(ins_pattern), choice_map_(choice_map) {}
  ~ExprMutator() override = default;

  Expr Mutate(Expr expr) override {
    Expr ret;
    auto idx = choice_map_.find(expr);
    if (idx == choice_map_.end() || idx->second == -1) {
      expr_stack_.push_back(expr);
      ret = IRMutator::Mutate(expr);
      expr_stack_.pop_back();
    } else {  // match an intrinsic
      CHECK_GT(ins_pattern_.size(), idx->second);
      ret = ins_pattern_[idx->second].replace_func(expr, *this);
    }
    return ret;
  }

  std::vector<Stmt> assign_stmt_;

 private:
  Expr Mutate_(const Call *op, const Expr &e) final {
    Array<Expr> args;
    std::transform(op->args.begin(), op->args.end(), std::back_inserter(args.CopyOnWrite()->data),
                   [this](const Expr &x) { return (Mutate(x)); });
    return Call::make(op->type, op->name, args, op->call_type, op->func, op->value_index);
  }

  const std::vector<InstructionPattern> &ins_pattern_;
  const std::unordered_map<Expr, int, air::NodeHash, air::NodeEqual> &choice_map_;

  std::vector<Expr> expr_stack_;
};

// Select instructions by dynamic programming on the tree
class InstructionSelector : public IRVisitor {
 public:
  explicit InstructionSelector(const std::vector<InstructionPattern> &ins_patterns) : ins_patterns_(ins_patterns) {}
  ~InstructionSelector() override = default;

  void Visit(const NodeRef &node) override {
    IRVisitor::Visit(node);
    Expr expr = Downcast<Expr>(node);

    int max_score = score_map_[expr];
    int max_i = -1;

    // try patterns
    for (size_t i = 0; i < ins_patterns_.size(); ++i) {
      int score = ins_patterns_[i].score_func(expr, score_map_);
      if (score > max_score) {
        max_score = score;
        max_i = static_cast<int>(i);
      }
    }

    score_map_[expr] = max_score;
    choice_map_[expr] = max_i;
  }

  std::unordered_map<Expr, int, air::NodeHash, air::NodeEqual> score_map_;
  std::unordered_map<Expr, int, air::NodeHash, air::NodeEqual> choice_map_;

 private:
  const std::vector<InstructionPattern> &ins_patterns_;
};

// Expand complicated expression to three address code
// Instruction selection is applied
class StmtMutator : public IRMutator {
 public:
  explicit StmtMutator(const std::vector<InstructionPattern> &ins_pattern) : ins_patterns_(ins_pattern) {}
  ~StmtMutator() override = default;

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    // If needed, use analyzer to rewrite op first
    Expr value = op->value;

    // select instructions
    InstructionSelector selector(ins_patterns_);
    selector.Visit(value);

    // If needed, add special operations for reduction here
    // mutate according to the result of instruction selection
    ExprMutator mutator(ins_patterns_, selector.choice_map_);
    value = mutator.Mutate(op->value);

    // If needed, remove the last useless copy
    mutator.assign_stmt_.push_back(Provide::make(op->func, op->value_index, value, op->args));

    return Block::make(mutator.assign_stmt_);
  }

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    FunctionRef func = Downcast<FunctionRef>(op->node);
    attr_node_[func] = op;
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const For *op, const Stmt &s) final { return IRMutator::Mutate_(op, s); }

 private:
  std::unordered_map<FunctionRef, const AttrStmt *, air::NodeHash, air::NodeEqual> attr_node_;
  const std::vector<InstructionPattern> &ins_patterns_;
};

Expr GetArg(const Call *op, size_t idx) {
  Array<Expr> args;
  CHECK(idx < op->args.size()) << "index out of bounds";
  args.push_back(op->args[idx]);
  return Call::make(op->type, op->name, args, op->call_type, op->func, op->value_index);
}

Stmt ExprPatternRewrite(Stmt stmt) {
  std::vector<air::arith::PVar<Expr>> pvars(3);
  air::arith::PVar<Expr> &x = pvars[0];
  const int NORMAL = 2;
  const int UNMATCH = -1;

  std::vector<InstructionPattern> ins_patterns{
    // reshape(a[i]) --> a[i]
    // transpose(a[i]) --> a[i]
    InstructionPattern{
      [&x](const Expr &expr, const std::unordered_map<Expr, int, air::NodeHash, air::NodeEqual> &score_map) -> int {
        if (const auto op = expr.as<Call>()) {
          if (call_reshape(x).Match(GetArg(op, 0)) || call_transpose(x).Match(GetArg(op, 0))) {
            return NORMAL;
          }
        }
        return UNMATCH;
      },
      [&x](Expr expr, ExprMutator &mutator) -> Expr {
        // Get first arg
        if (const auto op = expr.as<Call>()) {
          if (call_reshape(x).Match(GetArg(op, 0)) || call_transpose(x).Match(GetArg(op, 0))) {
            Expr x_eval = mutator.Mutate(x.Eval());
            return mutator.Mutate(x_eval);
          }
        }
        return expr;
      }},

    // mad(a, b) --> a + b
    InstructionPattern{
      [](const Expr &expr, const std::unordered_map<Expr, int, air::NodeHash, air::NodeEqual> &score_map) -> int {
        if (const auto op = expr.as<Call>()) {
          if (op->name == "mad") {
            return NORMAL;
          }
        }
        return UNMATCH;
      },
      [](Expr expr, ExprMutator &mutator) -> Expr {
        if (const auto op = expr.as<Call>()) {
          if (op->name == "mad") {
            CHECK_GE(op->args.size(), 2);
            return op->args[0] + op->args[1];
          }
        }
        return expr;
      }},
  };

  stmt = StmtMutator(ins_patterns).Mutate(stmt);
  return stmt;
}
}  // namespace ir
}  // namespace akg
