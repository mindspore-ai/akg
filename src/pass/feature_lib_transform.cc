/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include <tvm/ir_functor_ext.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_pass.h>
#include <tvm.h>
#include <ir_pass.h>
#include <pass/ir_util.h>
#include <pass/utils.h>
#include <algorithm>
#include <stack>

namespace akg {
namespace ir {
using LoopVarsTable = std::unordered_map<const Variable *, Range>;
using SubTensorTable = std::unordered_map<FunctionRef, Tensor, air::NodeHash, air::NodeEqual>;

enum LIB_CATEGORY { TENSOR_OF_T = 0, HYBRID_MIX, TRIGONO, IM2COL };

class LibAllocator : public IRVisitor {
 public:
  LibAllocator() {}
  ~LibAllocator() override = default;

  LIB_CATEGORY Run(const Stmt &s) {
    this->Visit(s);
    return category_;
  }

  SubTensorTable Table() { return remap_; }

  void Visit_(const AttrStmt *op) final {
    if (auto hybrid = op->node.as<air::HybridOpNode>()) {
      for (auto input : hybrid->inputs) {
        for (auto output : hybrid->outputs) {
          if (input == output) {
            category_ = LIB_CATEGORY::HYBRID_MIX;
            substitue_[hybrid->func_name()] = input;
          }
        }
      }
    }

    return IRVisitor::Visit_(op);
  }

  void Visit_(const Call *op) final {
    if ((op->name == "sin") || (op->name == "cos") || (op->name == "sinh") || (op->name == "cosh")) {
      category_ = LIB_CATEGORY::TRIGONO;
    } else if (op->name == "load3d_l1_ub") {
      category_ = LIB_CATEGORY::IM2COL;
    }
  }

  void Visit_(const Provide *op) final {
    if ((category_ == LIB_CATEGORY::HYBRID_MIX) && (substitue_.count(op->func->func_name()) > 0)) {
      remap_[op->func] = substitue_[op->func->func_name()];
    }

    return IRVisitor::Visit_(op);
  }

 private:
  LIB_CATEGORY category_{LIB_CATEGORY::TENSOR_OF_T};
  std::unordered_map<std::string, Tensor> substitue_;
  SubTensorTable remap_;
};

inline bool rangeCompare(const Range &a, const Range &b) {
  return (air::ir::Compare(a->min, b->min) == 0) && (air::ir::Compare(a->extent, b->extent) == 0);
}
/*
 * Example before this pass:
 * input   var1([0, 1000000])
 * input   var2([0, 1], [0, 16])
 * output  compute([0, 1000000]. [0, 16])
 * realize compute([0, 1000000], [0, 16]) {
    for (cc0, 0, 1000000) {
      for (cc1, 0, 16) {
        for (cc2, 0, 1) {
          if (cc2 == var1(cc0)) {
            compute(cc0, cc1) = var2(cc2, cc1);
          }
        }
      }
    }
  }

  How this pass works:
  First, we identify the equal condition 'cc2 == var1(cc0)' and find extention of variable cc2 is 1.
  Second, we remove this condition from ifThenElse condition expression.

  After this pass:
  realize compute([0, 1000000], [0, 16]) {
    for (cc0, 0, 1000000) {
      for (cc1, 0, 16) {
        for (cc2, 0, 1) {
          compute(cc0, cc1) = var2(cc2, cc1);
        }
      }
    }
  }
 * */
class TensorOfTensorTransform : public IRMutator {
 public:
  TensorOfTensorTransform() {}
  ~TensorOfTensorTransform() override = default;

  Stmt Run(const Stmt s) { return this->Mutate(s); }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (loop_vars_table.count(op->loop_var.get()) == 0) {
      loop_vars_table[op->loop_var.get()] = Range::make_by_min_extent(op->min, op->extent);
    }

    return IRMutator::Mutate_(op, s);
  }

  bool loopOneInTable(const Expr &expr) {
    if (expr.as<Variable>() != nullptr) {
      const auto var = expr.as<Variable>();
      if (loop_vars_table.count(var) > 0) {
        return rangeCompare(loop_vars_table[var], Range::make_by_min_extent(Expr(0), Expr(1)));
      }
    }

    return false;
  }

  bool needTransform(const Expr &arg) {
    if (arg.as<EQ>() != nullptr) {
      return (loopOneInTable(arg.as<EQ>()->a) || loopOneInTable(arg.as<EQ>()->b));
    }

    if (arg.as<And>() != nullptr) {
      return (needTransform(arg.as<And>()->a) || needTransform(arg.as<And>()->b));
    }

    return false;
  }

  void condTransform(const Expr &arg) {
    if (arg.as<EQ>() != nullptr) {
      auto eq_expr = arg.as<EQ>();
      if (!(loopOneInTable(eq_expr->a) || loopOneInTable(eq_expr->b))) {
        sel_conds.push_back(arg);
      }
    } else if (arg.as<And>() != nullptr) {
      auto and_expr = arg.as<And>();
      condTransform(and_expr->a);
      condTransform(and_expr->b);
    } else {
      sel_conds.push_back(arg);
    }

    return;
  }

  Stmt Mutate_(const IfThenElse *op, const Stmt &s) final {
    auto empty_case = op->else_case;
    if ((empty_case == Stmt()) && needTransform(op->condition)) {
      sel_conds.clear();
      condTransform(op->condition);
      Stmt then_case = Mutate(op->then_case);

      if (sel_conds.size() == 0) {
        return then_case;
      }

      Expr cond = sel_conds[0];
      for (size_t idx = 1; idx < sel_conds.size(); ++idx) {
        cond = And::make(cond, sel_conds[idx]);
      }

      return IfThenElse::make(cond, then_case);
    }

    return IRMutator::Mutate_(op, s);
  }

 private:
  std::vector<Expr> sel_conds;
  LoopVarsTable loop_vars_table;
};

const std::vector<double> TAYLOR_SIN_PRE = {
  -0.16666666666666667, -0.05, -0.023809528095280952, -0.013888888888888888, -0.0090909090909090909,
};
const std::vector<double> TAYLOR_COS_PRE = {
  0.000024801587301, 0.0013888888889, 0.0416666666667, 0.5000000000, 1.000000,
};
enum TRIGONOTYPE { OTHERS = 0, SIN, COS, SINH, COSH };

class TaylorExpan : public IRMutator {
 public:
  TaylorExpan() {}
  ~TaylorExpan() = default;

  Stmt Run(const Stmt &s) {
    make_call_ = [](const Tensor &load, const Provide *op) {
      return Call::make(load->dtype, load->op->name, op->args, op->value.as<Call>()->Halide, load->op,
                        load->value_index);
    };

    return Mutate(s);
  }

 private:
  TRIGONOTYPE TrigonoFunc(const Expr &expr) {
    if (expr.as<Call>()) {
      if (expr.as<Call>()->name == "sin") {
        return TRIGONOTYPE::SIN;
      } else if (expr.as<Call>()->name == "cos") {
        return TRIGONOTYPE::COS;
      } else if (expr.as<Call>()->name == "sinh") {
        return TRIGONOTYPE::SINH;
      } else if (expr.as<Call>()->name == "cosh") {
        return TRIGONOTYPE::COSH;
      }
    }

    return TRIGONOTYPE::OTHERS;
  }

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    Stmt res = IRMutator::Mutate_(op, s);
    FunctionRef func = Downcast<FunctionRef>(op->node);
    Tensor to_expand;

    if (index_node_.count(func) > 0) {
      int idx = index_node_[func];
      to_expand = Downcast<Operation>(func).output(idx);
    }

    if (expan_taylor_.count(to_expand) > 0) {
      const Realize *ref_real = realize_node_[to_expand];
      for (const auto &t : expan_taylor_.at(to_expand)) {
        Region bounds;
        for (size_t i = 0; i < t->shape.size(); ++i) {
          bounds.push_back(Range::make_by_min_extent(0, t->shape[i]));
        }

        res = Realize::make(t->op, t->value_index, t->dtype, bounds, ref_real->condition, res);
        res = AttrStmt::make(t->op, op->attr_key, op->value, res);
      }
    }

    return res;
  }

  Stmt Mutate_(const Realize *op, const Stmt &s) final {
    realize_node_[Downcast<Operation>(op->func).output(op->value_index)] = op;
    index_node_[op->func] = op->value_index;

    return IRMutator::Mutate_(op, s);
  }

  template <typename T>
  Stmt StmtCreater(const Expr &first, const Expr &second, const FunctionRef &func, const int &index,
                   const Provide *op) {
    Expr prov_expr = T::make(first, second);

    return Provide::make(func, index, prov_expr, op->args);
  }

  Stmt MakeStmtArray(const Stmt &first, const std::vector<Stmt> &array) {
    Stmt res = first;
    for (auto s : array) {
      res = Block::make(res, s);
    }

    return res;
  }

  void AddTaylorTable(const Provide *op, const std::vector<Tensor> &array) {
    Tensor realize_mark = Downcast<Operation>(op->func).output(op->value_index);
    if (expan_taylor_.count(realize_mark) == 0) {
      expan_taylor_[realize_mark] = array;
    }
  }

  Tensor GetFirstTensor(const Expr &expr) {
    std::vector<Tensor> tensors;
    PostOrderVisit(expr, [&tensors](const NodeRef &node) {
      const Call *call = node.as<Call>();
      if ((call != nullptr) && call->func.defined()) {
        tensors.push_back(Downcast<Operation>(call->func).output(call->value_index));
      }
    });

    CHECK_GE(tensors.size(), 1u);
    return tensors[0];
  }

  Stmt TaylorExpansionHyperbolic(const Provide *op, const TRIGONOTYPE &type) {
    Tensor to_expand = GetFirstTensor(op->value);
    Tensor minus =
      PlaceholderOpNode::make("taylor_" + std::to_string(ct_++), to_expand->shape, to_expand->dtype).output(0);

    std::vector<Tensor> allocate_tensors = {minus};
    std::vector<Stmt> stmt_vec;

    // minus = -x
    Stmt first = StmtCreater<Mul>(make_call_(to_expand, op), FloatImm::make(to_expand->dtype, -1.000), minus->op,
                                  minus->value_index, op);

    Tensor exp =
      PlaceholderOpNode::make("taylor_" + std::to_string(ct_++), to_expand->shape, to_expand->dtype).output(0);
    allocate_tensors.push_back(exp);

    // t_exp = exp(x)
    stmt_vec.push_back(
      Provide::make(exp->op, exp->value_index,
                    Call::make(to_expand->dtype, "exp", {make_call_(to_expand, op)}, Call::PureIntrinsic), op->args));

    Tensor exp_minus =
      PlaceholderOpNode::make("taylor_" + std::to_string(ct_++), to_expand->shape, to_expand->dtype).output(0);
    allocate_tensors.push_back(exp_minus);

    // t_exp_ = exp(-x)
    stmt_vec.push_back(Provide::make(exp_minus->op, exp_minus->value_index,
                                     Call::make(to_expand->dtype, "exp", {make_call_(minus, op)}, Call::PureIntrinsic),
                                     op->args));

    // t_minus = t_exp - t_exp_
    Tensor binary =
      PlaceholderOpNode::make("taylor_" + std::to_string(ct_++), to_expand->shape, to_expand->dtype).output(0);
    allocate_tensors.push_back(binary);
    if (type == TRIGONOTYPE::SINH) {
      stmt_vec.push_back(
        StmtCreater<Sub>(make_call_(exp, op), make_call_(exp_minus, op), binary->op, binary->value_index, op));
    } else if (type == TRIGONOTYPE::COSH) {
      stmt_vec.push_back(
        StmtCreater<Add>(make_call_(exp, op), make_call_(exp_minus, op), binary->op, binary->value_index, op));
    }

    // t_muls = t_minus * 0.5
    Tensor muls =
      PlaceholderOpNode::make("taylor_" + std::to_string(ct_++), to_expand->shape, to_expand->dtype).output(0);
    allocate_tensors.push_back(muls);
    stmt_vec.push_back(StmtCreater<Mul>(make_call_(binary, op), FloatImm::make(to_expand->dtype, 0.5000), op->func,
                                        op->value_index, op));

    std::reverse(allocate_tensors.begin(), allocate_tensors.end());
    AddTaylorTable(op, allocate_tensors);

    return MakeStmtArray(first, stmt_vec);
  }

  Stmt TaylorExpansionModCos(const Provide *op, const size_t &series) {
    series_ = series;

    Tensor to_expand = GetFirstTensor(op->value);
    Tensor pow_tensor =
      PlaceholderOpNode::make("taylor_" + std::to_string(ct_++), to_expand->shape, to_expand->dtype).output(0);
    Expr call_pow = make_call_(pow_tensor, op);
    std::vector<Tensor> allocate_tensors = {pow_tensor};
    std::vector<Stmt> stmt_vec;
    // pow = x * x
    Stmt first = StmtCreater<Mul>(make_call_(to_expand, op), make_call_(to_expand, op), pow_tensor->op,
                                  pow_tensor->value_index, op);

    std::stack<Expr> items;
    items.push(FloatImm::make(to_expand->dtype, TAYLOR_COS_PRE[0]));
    for (size_t i = 1; i < series_; ++i) {
      CHECK(i < TAYLOR_COS_PRE.size());
      Tensor t_mul =
        PlaceholderOpNode::make("taylor_" + std::to_string(ct_++), to_expand->shape, to_expand->dtype).output(0);
      allocate_tensors.push_back(t_mul);

      // t_mul = t_pow * prefix
      stmt_vec.push_back(StmtCreater<Mul>(items.top(), call_pow, t_mul->op, t_mul->value_index, op));

      Tensor t_muls =
        PlaceholderOpNode::make("taylor_" + std::to_string(ct_++), to_expand->shape, to_expand->dtype).output(0);
      allocate_tensors.push_back(t_muls);

      // t_mul = -1.000 * t_mul
      stmt_vec.push_back(StmtCreater<Mul>(FloatImm::make(to_expand->dtype, -1.000), make_call_(t_mul, op), t_muls->op,
                                          t_muls->value_index, op));

      // t_add = prefix + t_mul
      int value_index;
      FunctionRef func;
      if (i < series_ - 1) {
        Tensor t_add =
          PlaceholderOpNode::make("taylor_" + std::to_string(ct_++), to_expand->shape, to_expand->dtype).output(0);
        allocate_tensors.push_back(t_add);
        items.push(make_call_(t_add, op));
        func = t_add->op;
        value_index = t_add->value_index;
      } else {
        func = op->func;
        value_index = op->value_index;
      }
      stmt_vec.push_back(StmtCreater<Add>(FloatImm::make(to_expand->dtype, TAYLOR_COS_PRE[i]), make_call_(t_muls, op),
                                          func, value_index, op));
    }

    std::reverse(allocate_tensors.begin(), allocate_tensors.end());
    AddTaylorTable(op, allocate_tensors);

    return MakeStmtArray(first, stmt_vec);
  }

  Stmt TaylorExpansionModSin(const Provide *op, const size_t &series) {
    series_ = series;

    Tensor to_expand = GetFirstTensor(op->value);
    Tensor pow_tensor =
      PlaceholderOpNode::make("taylor_" + std::to_string(ct_++), to_expand->shape, to_expand->dtype).output(0);

    Expr call_pow = make_call_(pow_tensor, op);
    std::vector<Tensor> allocate_tensors = {pow_tensor};
    std::vector<Stmt> stmt_vec;
    // pow = x * x
    Stmt first = StmtCreater<Mul>(make_call_(to_expand, op), make_call_(to_expand, op), pow_tensor->op,
                                  pow_tensor->value_index, op);

    std::stack<Tensor> bases;
    std::stack<Tensor> items;
    bases.push(to_expand);
    items.push(to_expand);
    for (size_t i = 0; i < series_; ++i) {
      CHECK(i < TAYLOR_SIN_PRE.size());
      Tensor t_pown =
        PlaceholderOpNode::make("taylor_" + std::to_string(ct_++), to_expand->shape, to_expand->dtype).output(0);
      Tensor t_mul =
        PlaceholderOpNode::make("taylor_" + std::to_string(ct_++), to_expand->shape, to_expand->dtype).output(0);
      allocate_tensors.push_back(t_pown);
      allocate_tensors.push_back(t_mul);

      // t_pow = item * pow
      stmt_vec.push_back(StmtCreater<Mul>(call_pow, make_call_(items.top(), op), t_pown->op, t_pown->value_index, op));
      // t_mul = t_pow * prefix
      stmt_vec.push_back(StmtCreater<Mul>(FloatImm::make(to_expand->dtype, TAYLOR_SIN_PRE[i]), make_call_(t_pown, op),
                                          t_mul->op, t_mul->value_index, op));

      // t_add = t_base + t_mul
      int value_index;
      FunctionRef func;
      Expr base_expr = make_call_(bases.top(), op);
      if (i < series_ - 1) {
        Tensor t_add =
          PlaceholderOpNode::make("taylor_" + std::to_string(ct_++), to_expand->shape, to_expand->dtype).output(0);
        bases.push(t_add);
        allocate_tensors.push_back(t_add);
        func = t_add->op;
        value_index = t_add->value_index;
      } else {
        func = op->func;
        value_index = op->value_index;
      }
      stmt_vec.push_back(StmtCreater<Add>(base_expr, make_call_(t_mul, op), func, value_index, op));
      items.push(t_mul);
    }

    std::reverse(allocate_tensors.begin(), allocate_tensors.end());
    AddTaylorTable(op, allocate_tensors);

    return MakeStmtArray(first, stmt_vec);
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    TRIGONOTYPE type = TrigonoFunc(op->value);
    switch (type) {
      case TRIGONOTYPE::SIN:
        return TaylorExpansionModSin(op, 5);
      case TRIGONOTYPE::COS:
        return TaylorExpansionModCos(op, 5);
      case TRIGONOTYPE::SINH:
        return TaylorExpansionHyperbolic(op, type);
      case TRIGONOTYPE::COSH:
        return TaylorExpansionHyperbolic(op, type);
      default:
        break;
    }

    return IRMutator::Mutate_(op, s);
  }

 private:
  std::unordered_map<Tensor, std::vector<Tensor>> expan_taylor_;
  std::unordered_map<Tensor, const Realize *> realize_node_;
  std::unordered_map<FunctionRef, int, air::NodeHash, air::NodeEqual> index_node_;
  std::function<Expr(const Tensor &, const Provide *)> make_call_;
  size_t series_{4};
  static int ct_;
};

Stmt HybridMixSubstitue(const Stmt &s, const SubTensorTable &table) {
  Stmt res = s;
  for (auto item : table) {
    res = TensorSubstitute(s, item.first, item.second->op, item.second->value_index);
  }

  return res;
}

int TaylorExpan::ct_ = 0;

class FloorDivOpt : public IRMutator {
 public:
  Stmt Run(const Stmt &s) {
    Stmt res = Mutate(s);
    for (auto item : new_let_stmts_) {
      res = LetStmt::make(item.first, item.second, res);
    }

    return res;
  }

 private:
  bool InVarMap(const Expr &target) {
    for (auto item : new_let_stmts_) {
      if (Compare(item.second, target) == 0) {
        return false;
      }
    }

    return true;
  }

  Expr Value(const Expr &target) {
    for (auto item : new_let_stmts_) {
      if (Compare(item.second, target) == 0) {
        return item.first;
      }
    }

    return target;
  }

  Expr Mutate_(const FloorDiv *op, const Expr &e) final {
    Expr second = FloorDiv::make(op->a, op->b);
    if (InVarMap(second)) {
      Var tmp("_div_" + std::to_string(ct_++), op->type);
      new_let_stmts_.push_back(std::make_pair(tmp, second));
      return tmp;
    }

    return Value(second);
  }

  std::vector<std::pair<Var, Expr>> new_let_stmts_;
  static int ct_;
};

int FloorDivOpt::ct_ = 0;

Stmt FeatureLibTransform(const Stmt stmt) {
  LibAllocator allocator;

  LIB_CATEGORY type = allocator.Run(stmt);
  if (type == LIB_CATEGORY::TENSOR_OF_T) {
    return TensorOfTensorTransform().Run(stmt);
  } else if (type == LIB_CATEGORY::TRIGONO) {
    return TaylorExpan().Run(stmt);
  } else if (type == LIB_CATEGORY::HYBRID_MIX) {
    return HybridMixSubstitue(stmt, allocator.Table());
  } else if (type == LIB_CATEGORY::IM2COL) {
    return FloorDivOpt().Run(stmt);
  }

  return stmt;
}
}  // namespace ir
}  // namespace akg
