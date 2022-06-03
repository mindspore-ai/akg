/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#include <dmlc/common.h>
#include <tvm/ir.h>
#include <tvm/tensor.h>
#include <tvm/ir_functor_ext.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_visitor.h>
#include <tvm/api_registry.h>
#include <tvm.h>
#include <floating.h>
#include <limits>
#include <queue>
#include <algorithm>
#include "pass/utils.h"
#include "pass/rewrite_simplify_cce.h"

namespace akg {
namespace ir {
using VarSet = std::unordered_set<Var, air::NodeHash, air::NodeEqual>;

// forward declaration
class ThreeAddressExprMutator;

class ExprArgsFetcher : public IRVisitor {
 public:
  explicit ExprArgsFetcher(Array<Expr> args) : args_(args), index_(args_.size() - 1) {}
  ~ExprArgsFetcher() override = default;

  Array<Expr> GetArgs(const Expr &e) {
    Visit(e);
    if (max_dim >= args_.size()) {
      return args_;
    }
    Array<Expr> args;
    while (index_ < args_.size()) {
      args.push_back(args_[index_]);
      index_++;
    }
    if (CountVars(args) == CountVars(args_)) {
      return args_;
    }
    return args;
  }

  bool MustBroadcast(const Expr &e) {
    if (is_constant(e) || CountVars(e) == 0) {
      return false;
    }
    size_t size = GetArgs(e).size();
    return size > max_dim;
  }

  void Visit_(const Call *op) override {
    if (op->call_type == Call::CallType::Halide) {
      max_dim = max_dim < op->args.size() ? op->args.size() : max_dim;
      for (Expr arg : op->args) {
        size_t index = GetIndex(arg);
        index_ = index_ > index ? index : index_;
      }
    } else {
      CHECK(op->call_type == Call::CallType::PureIntrinsic);
      for (Expr e : op->args) {
        Array<Expr> args = GetArgs(e);
        max_dim = max_dim < args.size() ? args.size() : max_dim;
        for (Expr arg : args) {
          size_t index = GetIndex(arg);
          index_ = index_ > index ? index : index_;
        }
      }
    }
  }

 private:
  size_t GetIndex(const Expr &arg) {
    if (is_constant(arg)) {
      return index_;
    }
    for (size_t i = 0; i < args_.size(); ++i) {
      if (args_[i].same_as(arg)) {
        return i;
      }
    }
    return index_;
  }

  Array<Expr> args_;
  size_t index_;
  size_t max_dim{0};
};

class ScalarOperandFinder : public IRVisitor {
 public:
  bool Find(const Expr &e) {
    Visit(e);
    return find_;
  }

  // float32(input_2(i0) override) < float32(input_3(i0))
  void Visit_(const Cast *op) override {
    if (op->type.is_float()) in_float_cast_ = true;
    IRVisitor::Visit_(op);
    in_float_cast_ = false;
  }

  void Visit_(const Call *op) override {
    if (op->call_type == Call::CallType::Halide) {
      if (!in_index_ && (op->type.is_int() || op->type.is_uint()) && !in_float_cast_) {
        find_ = true;
      }
      in_index_++;
      IRVisitor::Visit_(op);
      in_index_--;
    }
  }

  void Visit_(const Variable *op) override {
    if (!in_index_) {
      find_ = true;
    }
  }

 private:
  int in_index_{0};
  bool find_{false};
  bool in_float_cast_{false};
};

// Assign a hash value for an expression. This is used for common expression elmination
class ExprHasher : public air::ir::ExprFunctor<size_t(const Expr &n)> {
 public:
  ExprHasher() : cross_simplify_(false) {}
  explicit ExprHasher(bool cross_simplify) : cross_simplify_(cross_simplify) {}
  ~ExprHasher() override = default;

 private:
  size_t VisitExpr_(const Add *op) final { return VisitExpr(op->a) + VisitExpr(op->b); }
  size_t VisitExpr_(const Sub *op) final { return VisitExpr(op->a) - VisitExpr(op->b); }
  size_t VisitExpr_(const Mul *op) final { return VisitExpr(op->a) * VisitExpr(op->b); }
  size_t VisitExpr_(const Div *op) final { return dmlc::HashCombine(VisitExpr(op->a), VisitExpr(op->b)); }
  size_t VisitExpr_(const Call *op) final {
    if (op->call_type == Call::CallType::PureIntrinsic) {
      pure_intrinsic_level++;
    }
    size_t ret = std::hash<const Node *>{}(op->func.get());
    if (cross_simplify_ && (op->func.get() == nullptr)) {
      ret = std::hash<std::string>{}(op->name);
    }
    for (size_t i = 0; i < op->args.size(); ++i) {
      ret = dmlc::HashCombine(ret, VisitExpr(op->args[i]));
    }
    if (op->call_type == Call::CallType::PureIntrinsic) {
      pure_intrinsic_level--;
    }
    return ret;
  }

  size_t VisitExpr_(const Variable *op) final {
    if (cross_simplify_) {
      return std::hash<std::string>()(op->name_hint);
    } else {
      return std::hash<const Node *>()(op);
    }
  }

  size_t VisitExpr_(const FloatImm *op) final { return std::hash<double>()(op->value); }

  size_t VisitExpr_(const IntImm *op) final { return std::hash<int64_t>()(op->value); }

  size_t VisitExprDefault_(const Node *op) final {
    if (cross_simplify_ || pure_intrinsic_level > 0) {
      if (op->IsInstance<Cast>()) {  // Support for cases of float16(A), float32(A)
        auto cast_op = static_cast<const Cast *>(op);
        auto value_hash = VisitExpr(cast_op->value);
        std::ostringstream os;
        os << cast_op->type;
        auto type_hash = std::hash<std::string>()(os.str());
        return dmlc::HashCombine(type_hash, value_hash);
      }
    }
    return std::hash<const Node *>()(op);
  }

  bool cross_simplify_{false};
  int pure_intrinsic_level{0};
};

// poly does not support both AND and OR to exist in an expression.
class PolyUnsupportedExprChecker : public IRVisitor {
 public:
  bool isSupported(const Expr &expr) {
    and_found = false;
    or_found = false;
    Visit(expr);
    return !(and_found && or_found);
  }

 private:
  void Visit_(const And *expr) override {
    and_found = true;
    Visit(expr->a);
    Visit(expr->b);
  }

  void Visit_(const Or *expr) override {
    or_found = true;
    Visit(expr->a);
    Visit(expr->b);
  }

  bool or_found{false};
  bool and_found{false};
};

// Collect all Tensors used in an Expr
std::unordered_set<Tensor> GetExprTensors(const Expr expr) {
  std::unordered_set<Tensor> tensors;
  PostOrderVisit(expr, [&tensors](const NodeRef &node) {
    const Call *t_call = node.as<Call>();
    if (t_call != nullptr && t_call->func.defined()) {
      tensors.insert(Downcast<Operation>(t_call->func).output(t_call->value_index));
    }
  });
  return tensors;
}

// Mutate expression according to selection choices
class ThreeAddressExprMutator : public IRMutator {
 public:
  ThreeAddressExprMutator(const Tensor output, const Array<Expr> &args, const Array<Expr> &out_args,
                          const Array<Expr> &shape, const std::unordered_set<const Call *> &broadcast,
                          bool IsReductionOp, bool cross_stmt_simplify, bool is_simple = false)
      : output_(output),
        args_(args),
        out_args_(out_args),
        shape_(shape),
        broadcast_(broadcast),
        IsReductionOp_(IsReductionOp),
        cross_simplify_(cross_stmt_simplify),
        hasher_(cross_stmt_simplify),
        is_simple_(is_simple) {
    CHECK_EQ(args_.size(), shape_.size());
    if (shape_.empty()) {  // scalar values should have at least one dimension and contains one element
      shape_.push_back(1);
      args_.push_back(0);
    }
    expand_floatimm_.push_back(true);  // expand float immediate by default
  }
  ~ThreeAddressExprMutator() override = default;

  std::unordered_map<size_t, std::pair<Expr, Expr>> GetCommonExpr() { return common_exprs_; }
  void SetCommonExpr(std::unordered_map<size_t, std::pair<Expr, Expr>> global_common_expr) {
    common_exprs_.insert(global_common_expr.begin(), global_common_expr.end());
  }

  Expr AllocateTmp(Expr value, Array<Expr> args = {}) {
    // detect common expression
    size_t hash_value = hasher_(value);
    auto x = common_exprs_[hash_value];
    if (Equal(x.first, value)) {
      return x.second;
    }
    if (cross_simplify_) {
      for (auto tmp_it : common_exprs_) {
        if (Equal(tmp_it.second.second, value)) {
          return value;
        }
      }
    }

    // allocate new immediate tensor
    Tensor imm;
    if (args.empty()) {
      args = args_;
    }
    std::string name = output_->op->name + "_" + std::to_string(ct_++);
    imm = PlaceholderOpNode::make(name, GetShape(args), value.type()).output(0);
    imm_tensors.push_back(imm);
    imm_ops.insert(imm->op);

    // update common expr
    assign_stmt.push_back(Provide::make(imm->op, imm->value_index, value, args));
    Expr ret = Call::make(value.type(), imm->op->name, args, Call::CallType::Halide, imm->op, imm->value_index);
    common_exprs_[hash_value] = std::make_pair(value, ret);
    imm2hash_[imm->op] = hash_value;
    return ret;
  }

  Expr AssignTmp(const Expr tmp_tensor, Expr value) {
    Tensor imm = GetImmTensor(tmp_tensor);

    // delete old cached common expr
    size_t old_hash = imm2hash_[imm->op];
    common_exprs_.erase(old_hash);

    // update new common expr
    Array<Expr> args = args_;
    if (is_simple_) {
      args = ExprArgsFetcher(args_).GetArgs(value);
    }
    assign_stmt.push_back(Provide::make(imm->op, imm->value_index, value, args));
    size_t hash_value = hasher_(value);
    Expr ret = Call::make(value.type(), imm->op->name, args, Call::CallType::Halide, imm->op, imm->value_index);
    common_exprs_[hash_value] = std::make_pair(value, ret);
    imm2hash_[imm->op] = hash_value;
    return ret;
  }

  bool IsTmpTensor(const Expr expr) {
    const Call *node = expr.as<Call>();

    if (node == nullptr) {
      return false;
    }
    return imm_ops.count(node->func);
  }

  bool IsTmpTensor(const Tensor t) {
    if (!t.defined()) {
      return false;
    }
    return imm_ops.count(t->op);
  }

  Tensor GetImmTensor(const Expr expr) {
    const Call *node = expr.as<Call>();
    CHECK(node != nullptr);
    CHECK(imm_ops.count(node->func));
    return Downcast<Operation>(node->func).output(node->value_index);
  }

  // forward declaration
  Expr Mutate(Expr expr) override;

  template <typename T>
  Expr MutateBinaryOp(const T *op, const Expr &e) {
    in_call_++;
    Expr l = Mutate(op->a);
    Expr r = Mutate(op->b);
    in_call_--;

    Array<Expr> args = args_;
    if (is_simple_) {
      args = ExprArgsFetcher(args_).GetArgs(T::make(l, r));
    }
    bool broadcast_l = !IsReductionOp_ && !is_constant(l) && CountVars(args) > CountVars(l);
    bool broadcast_r = !IsReductionOp_ && !is_constant(r) && CountVars(args) > CountVars(r);

    // We must split broadcast tensors that has "V-B" axes pattern
    // in which "V" is vectorized loop and "B" is broadcasted loop.
    auto IsBlockingVectorization = [this](const Expr &node) -> bool {
      if (auto call = node.as<Call>()) {
        // e.g.1 Dst[cc0][cc1] = Src0[cc0][0] * Src1[cc0][cc1]
        // e.g.2 Dst[cc0][cc1][cc2] = Src0[cc0][0][cc2] * Src1[cc0][cc1][cc2]
        bool explicit_blocking = false;
        for (int i = 0; i < static_cast<int>(call->args.size()) - 1; ++i) {
          auto cur = call->args[i];
          auto next = call->args[i + 1];
          if (!is_constant(cur) && is_constant(next)) {
            // "V-B" pattern
            explicit_blocking = true;
            break;
          }
        }

        // e.g. Dst[cc0][cc1] = Src0[cc0] * Src1[cc0][cc1]
        bool implicit_blocking = false;
        if (!explicit_blocking && out_args_.size() > 0 && call->args.size() > 0) {
          auto dst_index = out_args_[out_args_.size() - 1];
          auto src_index = call->args[call->args.size() - 1];
          implicit_blocking = (dst_index.as<Variable>() && src_index.as<Variable>() &&
                               dst_index.as<Variable>()->name_hint != src_index.as<Variable>()->name_hint);
        }
        return explicit_blocking || implicit_blocking;
      }
      return false;
    };

    if (op->template IsInstance<Add>() || op->template IsInstance<Mul>()) {
      if (broadcast_l && (broadcast_r || is_constant(r))) {
        l = AllocateTmp(l, args);
      } else if (broadcast_r && is_constant(l)) {
        r = AllocateTmp(r, args);
      }
      if (CountVars(args) > CountVars(r) && ExprArgsFetcher(out_args_).MustBroadcast(r)) {
        r = AllocateTmp(r, args);
      }

      // do split afterall
      if (broadcast_l && IsBlockingVectorization(l)) {
        l = AllocateTmp(l, args);
      }
      if (broadcast_r && IsBlockingVectorization(r)) {
        r = AllocateTmp(r, args);
      }
    }
    return AllocateTmp(T::make(Mutate(l), Mutate(r)), args);
  }

  Expr Mutate_(const Add *op, const Expr &e) final { return MutateBinaryOp<Add>(op, e); }
  Expr Mutate_(const Sub *op, const Expr &e) final {
    in_call_++;
    Expr l = Mutate(op->a);
    Expr r = Mutate(op->b);
    in_call_--;
    if (is_constant(l)) {
      // fix the missing of vsubs (e.g.  b[i] = 1.0 - a[i] -> tmp[i] = a[i] * -1;  b[i] = tmp[i] + 1.0
      Expr tmp = AllocateTmp(Mul::make(r, make_const(r.type(), -1.0)));

      if (isZero(l)) return tmp;

      return AllocateTmp(Add::make(tmp, l));
    }

    const Call *_a = l.as<Call>();
    const Call *_b = r.as<Call>();
    if (_a && _b && IsReductionOp_ && CountVars(l) < CountVars(r)) {
      // for a[i] = a[i] - b[i, j] -> tmp[i, j] = b[i, j] * -1; a[i] = a[i] + tmp[i, j]
      Expr tmp = AllocateTmp(Mul::make(r, make_const(r.type(), -1.0)));
      return AllocateTmp(Add::make(l, tmp));
    }

    return AllocateTmp(Sub::make(l, r));
  }
  Expr Mutate_(const Mul *op, const Expr &e) final { return MutateBinaryOp<Mul>(op, e); }
  Expr Mutate_(const Div *op, const Expr &e) final { return MutateBinaryOp<Div>(op, e); }
  Expr Mutate_(const Mod *op, const Expr &e) final { return MutateBinaryOp<Mod>(op, e); }
  Expr Mutate_(const Max *op, const Expr &e) final {
    if (in_call_) {
      return AllocateTmp(IRMutator::Mutate_(op, e));
    }
    in_call_++;
    Expr ret = IRMutator::Mutate_(op, e);
    in_call_--;
    return ret;
  }
  Expr Mutate_(const Min *op, const Expr &e) final {
    if (in_call_) {
      return AllocateTmp(IRMutator::Mutate_(op, e));
    }
    in_call_++;
    Expr ret = IRMutator::Mutate_(op, e);
    in_call_--;
    return ret;
  }

  Expr Mutate_(const Call *op, const Expr &e) final {
    if (op->call_type == Call::CallType::Halide) {
      // broadcast for a[i, j] = cast(a[j]) -> t[i, j] = a[j]; a[i, j] = cast(t[i, j])
      if (expr_stack.size() >= binary_size_ && expr_stack[expr_stack.size() - binary_size_]->IsInstance<Cast>() &&
          CountVars(args_) > CountVars(e)) {
        return AllocateTmp(e);
      }

      // C[i] = A[i] op B[N-i]; ==> B'[i] = B[N-i]; C[i] = A[i] op B'[i]
      // only support the last axis reverse indexing
      if (op->args.size() > 0) {
        VarSet vars;
        GatherVars(op->args[op->args.size() - 1], &vars);
        if (vars.size() == 1) {
          auto coff = air::arith::DetectLinearEquation(op->args[op->args.size() - 1], {*vars.begin()});
          if (coff.size() > 0 && coff[0].as<IntImm>() && coff[0].as<IntImm>()->value < 0) {
            return AllocateTmp(e);
          }
        }
      }

      // need transpose A[i, j] = op(B[j, i]); ==> B'[i, j] = B[j, i]; A[i, j] = op(B'[i, j])
      if (args_.size() >= 1 && args_[args_.size() - 1]->IsInstance<Variable>() &&
          op->args[op->args.size() - 1]->IsInstance<Variable>()) {
        const Var innermost = Downcast<Var>(args_[args_.size() - 1]);
        if ((IsReductionOp_ && expr_stack.size() >= triple_size_) ||
            (!IsReductionOp_ && expr_stack.size() >= binary_size_ && op->args.size() > 1)) {
          Expr x = expr_stack[expr_stack.size() - binary_size_];
          const Call *call = x.as<Call>();
          if (!(call && (call->name == "proposal_sort" || call->name == "topk_sort" || call->name == "iou" ||
                         call->name == "nms" || call->name == "four2five_nchw" || call->name == "vmadd" ||
                         call->name == "vmla"))) {
            VarSet vars;
            GatherVars(op->args[op->args.size() - 1], &vars);
            if (vars.count(innermost) == 0) {
              return AllocateTmp(e);
            }
          }
        }
      }

      bool broadcast = true;
      if (expr_stack.size() >= binary_size_) {
        Expr x = expr_stack[expr_stack.size() - binary_size_];
        if (x->IsInstance<Add>() || x->IsInstance<Mul>()) {
          broadcast = false;
        }
        const Call *call = x.as<Call>();
        if (call && (call->name == "proposal_sort" || call->name == "topk_sort" || call->name == "iou" ||
                     call->name == "nms" || call->name == "vmadd" || call->name == "vmla")) {
          broadcast = false;
        }
      }

      // broadcast when need
      if (broadcast_.count(op) && broadcast) {
        if (expr_stack.size() >= binary_size_ && expr_stack[expr_stack.size() - binary_size_]->IsInstance<Div>()) {
          Array<Expr> args = ExprArgsFetcher(args_).GetArgs(expr_stack[expr_stack.size() - binary_size_]);
          if (CountVars(e) < CountVars(args)) {
            return AllocateTmp(e, args);
          }
        } else {
          return AllocateTmp(e);
        }
      }
      // this is going to generate a tensor of tensor expr, like A(B(i))
      return e;
    } else if (op->call_type == Call::CallType::PureIntrinsic && op->name == air::ir::intrinsic::tvm_if_then_else) {
      // do not split the condition of tvm_if_then_else
      Array<Expr> args;
      in_call_++;
      // do not expand FloatImm if found scalars operands in condition
      expand_floatimm_.push_back(!ScalarOperandFinder().Find(op->args[0]));
      args.push_back(op->args[0]);
      args.push_back(Mutate(op->args[1]));
      args.push_back(Mutate(op->args[2]));
      expand_floatimm_.pop_back();
      in_call_--;
      return AllocateTmp(Call::make(op->type, op->name, args, op->call_type, op->func, op->value_index));
    } else {
      Array<Expr> args;
      in_call_++;
      for (const auto &x : op->args) {
        args.push_back(is_constant(x) && op->name == "vaxpy" ? x : Mutate(x));
      }
      in_call_--;
      if (op->name == "vmadd" || op->name == "vaxpy" || op->name == "vmla") {
        // detect common expression
        size_t hash_value = hasher_(e);
        auto x = common_exprs_[hash_value];
        if (Equal(x.first, e)) {
          return x.second;
        }
        Expr ret = FixMultivarInsn(op, args);
        common_exprs_[hash_value] = std::make_pair(e, ret);
        return ret;
      }
      return AllocateTmp(Call::make(op->type, op->name, args, op->call_type, op->func, op->value_index));
    }
  }

  Expr Mutate_(const Select *op, const Expr &e) final {
    // do not split the condition of Select
    in_call_++;
    Expr cond = CanonicalSimplify(op->condition);
    if (!PolyUnsupportedExprChecker().isSupported(cond)) {
      cond = Simplify_cce(op->condition);
      if (!PolyUnsupportedExprChecker().isSupported(cond)) {
        cond = op->condition;
      }
    }
    if (!ScalarOperandFinder().Find(cond)) {
      cond = Mutate(cond);
    }
    Expr ret = AllocateTmp(Select::make(cond, Mutate(op->true_value), Mutate(op->false_value)));
    in_call_--;
    return ret;
  }

  Expr Mutate_(const Cast *op, const Expr &e) final {
    if (in_call_) {
      return AllocateTmp(IRMutator::Mutate_(op, e));
    }
    in_call_++;
    Expr ret = IRMutator::Mutate_(op, e);
    in_call_--;
    return ret;
  }

  template <typename T>
  Expr MutateConstOp(const T *op, const Expr &e) {
    std::set<std::string> excludeSet = {"nms"};
    bool excludeIntrin = expr_stack.size() >= 2 && expr_stack[expr_stack.size() - 2]->IsInstance<Call>() &&
                         excludeSet.count(expr_stack[expr_stack.size() - 2].as<Call>()->name) > 0;
    if (in_call_ && expand_floatimm_.back() &&
        ((expr_stack.size() >= 2 && (expr_stack[expr_stack.size() - 2]->IsInstance<Call>()    // log(0.1)
                                     || expr_stack[expr_stack.size() - 2]->IsInstance<Max>()  // Max(a, 0.1), Max(a,1)
                                     || expr_stack[expr_stack.size() - 2]->IsInstance<Min>()))) &&
        !excludeIntrin  // Don't handle nms intrin
    ) {
      return AllocateTmp(e);
    } else {
      return IRMutator::Mutate_(op, e);
    }
  }

  Expr Mutate_(const FloatImm *op, const Expr &e) final { return MutateConstOp(op, e); }
  Expr Mutate_(const IntImm *op, const Expr &e) final { return MutateConstOp(op, e); }

  void AddBroadCastCallIfNeed(const Call *op, const Expr &e) {
    if (broadcast_.find(op) == broadcast_.end()) {
      return;
    }
    const Call *new_call = e.as<Call>();
    CHECK_NOTNULL(new_call);
    broadcast_.insert(new_call);
  }

  std::vector<Stmt> assign_stmt;
  std::vector<Tensor> imm_tensors;
  std::unordered_set<FunctionRef, air::NodeHash, air::NodeEqual> imm_ops;

 private:
  Expr FixMultivarInsn(const Call *op, const Array<Expr> &args) {
    auto arg2 = IsTmpTensor(args[2]) ? args[2] : AllocateTmp(args[2]);
    Array<Expr> new_args({args[0], args[1], arg2});
    if (level_ > 1) {
      return AssignTmp(arg2, Call::make(op->type, op->name, new_args, op->call_type, op->func, op->value_index));
    } else {
      auto result = AssignTmp(arg2, Call::make(op->type, op->name, new_args, op->call_type, op->func, op->value_index));
      return AllocateTmp(result);
    }
  }

  Array<Expr> GetShape(const Array<Expr> &args) {
    if (CountVars(args) == CountVars(args_)) {
      return shape_;
    }
    const size_t dim = args.size();
    const size_t maxDim = output_->shape.size();
    CHECK_LE(dim, maxDim);
    Array<Expr> shape;
    size_t index = maxDim - dim;
    while (index < maxDim) {
      shape.push_back(output_->shape[index]);
      index++;
    }
    return shape;
  }

  Tensor output_;
  Array<Expr> args_;
  Array<Expr> out_args_;
  Array<Expr> shape_;

  std::unordered_map<size_t, std::pair<Expr, Expr>> common_exprs_;  // hash value -> <match expr, replace expr>
  // imm tensor -> hash value of the expr in the tensor
  std::unordered_map<FunctionRef, size_t, air::NodeHash, air::NodeEqual> imm2hash_;

  int level_{0};
  int in_call_{0};
  std::vector<Expr> expr_stack;

  std::unordered_set<const Call *> broadcast_;

  static int ct_;
  bool disable_selection_{false};
  std::vector<bool> expand_floatimm_;
  bool IsReductionOp_{false};
  bool cross_simplify_;
  ExprHasher hasher_;
  bool is_simple_;
  size_t binary_size_{2};
  size_t triple_size_{3};
};

Expr ThreeAddressExprMutator::Mutate(Expr expr) {
  level_++;
  expr_stack.push_back(expr);
  Expr ret = IRMutator::Mutate(expr);
  expr_stack.pop_back();
  level_--;
  return ret;
}

int ThreeAddressExprMutator::ct_ = 0;

class InstructionMutator : IRMutator {
 public:
  explicit InstructionMutator(ThreeAddressExprMutator &mutator, Array<Expr> &args) : mutator_(mutator), args_(args) {}
  ~InstructionMutator() = default;

  Expr Mutate(Expr value) { return IRMutator::Mutate(value); }

  // VMADD.type {f16, f32} [Xd], [Xn], [Xm], Xt, MASK
  //       [Xd] = [Xn] * [Xd] + [Xm]
  // VAXPY.type {f16, f32, fmix} [Xd], [Xn], Xm, Xt, MASK
  //       [Xd] = Xm * [Xn] + [Xd]
  Expr Mutate_(const Add *op, const Expr &e) {
    Expr l = Mutate(op->a);
    Expr r = Mutate(op->b);
    if (is_constant(l) && is_constant(r)) {
      return ConstantFold<Add>(l, r);
    }
    if (is_constant(l) || is_constant(r)) {
      return Add::make(l, r);
    }

    bool is_left_candidate = IsCandidate(l);
    if (!is_left_candidate && !IsCandidate(r)) {
      return Add::make(l, r);
    }
    Expr candidate = is_left_candidate ? l : r;
    Expr non_candidate = is_left_candidate ? r : l;
    if (op->type != non_candidate.type() || op->type.bits() < candidate.type().bits()) {
      return Add::make(l, r);
    }
    Array<Expr> args;
    const Mul *mul = candidate.as<Mul>();
    bool is_left_constant = is_constant(mul->a);
    if (is_left_constant || is_constant(mul->b)) {
      args.push_back(is_left_constant ? mul->a : mul->b);
      args.push_back(is_left_constant ? mul->b : mul->a);
      args.push_back(non_candidate);
    } else {
      args.push_back(mul->a);
      args.push_back(non_candidate);
      args.push_back(mul->b);
    }

    if (is_constant(args[0])) {
      return Add::make(l, r);
    }

    if (!is_constant(args[0]) && (op->type != args[0].type() || op->type != args[2].type())) {
      return Add::make(l, r);
    }

    if (CountVars(args[1]) != CountVars(args[2]) ||
        (!is_constant(args[0]) && CountVars(args[0]) != CountVars(args[1])) || CountVars(args[2]) != CountVars(args_)) {
      return Add::make(l, r);
    }

    return Call::make(args[0].type(), is_constant(args[0]) ? "vaxpy" : "vmadd", args, Call::CallType::PureIntrinsic);
  }

  Expr Mutate_(const Sub *op, const Expr &e) {
    Expr l = Mutate(op->a);
    Expr r = Mutate(op->b);
    if (is_constant(l) && is_constant(r)) {
      return ConstantFold<Sub>(l, r);
    }
    return Sub::make(l, r);
  }

  Expr Mutate_(const Mul *op, const Expr &e) {
    Expr l = Mutate(op->a);
    Expr r = Mutate(op->b);
    bool is_left_constant = is_constant(l);
    bool is_right_constant = is_constant(r);
    if (!is_left_constant && !is_right_constant) {
      return Mul::make(l, r);
    }
    if (is_left_constant && is_right_constant) {
      return ConstantFold<Mul>(l, r);
    }
    Expr constant = is_left_constant ? l : r;
    Expr nonconstant = is_left_constant ? r : l;
    if (const Add *add = nonconstant.as<Add>()) {
      return MulExprMutator<Add>(constant, add);
    } else if (const Sub *sub = nonconstant.as<Sub>()) {
      return MulExprMutator<Sub>(constant, sub);
    }
    return Mul::make(l, r);
  }

  Expr Mutate_(const Div *op, const Expr &e) {
    Expr l = Mutate(op->a);
    Expr r = Mutate(op->b);
    if (is_constant(l) && is_constant(r)) {
      return ConstantFold<Div>(l, r);
    } else if (is_constant(l)) {
      l = mutator_.AllocateTmp(l, ExprArgsFetcher(args_).GetArgs(Div::make(l, r)));
    }
    return Div::make(l, r);
  }

  // vrelu  [Xd] = max([Xn], 0)
  // vmaddrelu  [Xd] = max(vmadd [Xd], 0)
  Expr Mutate_(const Max *op, const Expr &e) {
    // relu only support fp16
    if (!op->type.is_float() || op->type.bits() != 16) {
      return Max::make(Mutate(op->a), Mutate(op->b));
    }
    bool is_left_zero = isZero(op->a);
    bool is_right_zero = IsZero(op->b);
    if (!is_left_zero && !is_right_zero) {
      return Max::make(Mutate(op->a), Mutate(op->b));
    }
    Expr expr = op->a;
    if (is_left_zero) {
      expr = op->b;
    }

    if (const Call *call = expr.as<Call>()) {
      if (call->call_type == Call::CallType::PureIntrinsic && call->name == "vmadd") {
        return Call::make(op->type, "vmaddrelu", call->args, Call::CallType::PureIntrinsic);
      }
    }
    return Call::make(op->type, "relu", {expr}, Call::CallType::PureIntrinsic);
  }

  // int32 floor/ceil/round/trunc() --> floor/ceil/round/trunc()
  // float(cc1) -> a[i] = cc1; cast(a[i])
  Expr Mutate_(const Cast *op, const Expr &e) {
    Expr value = Mutate(op->value);
    if (op->type.is_int() && value->IsInstance<Call>()) {
      const Call *call = value.as<Call>();
      if (call->name != "floor" && call->name != "ceil" && call->name != "round" && call->name != "trunc") {
        return Cast::make(op->type, value);
      }
      if (op->type == call->type) {
        return value;
      } else {
        return Call::make(op->type, call->name, call->args, call->call_type, call->func, call->value_index);
      }
    }
    if (op->type.is_float() && value->IsInstance<Variable>()) {
      return Cast::make(op->type, mutator_.AllocateTmp(value));
    }
    return Cast::make(op->type, value);
  }

  Expr Mutate_(const Select *op, const Expr &e) {
    Expr condition = Mutate(op->condition);
    Expr true_value = Mutate(op->true_value);
    Expr false_value = Mutate(op->false_value);
    if (const Not *notCond = condition.as<Not>()) {
      return Select::make(notCond->a, false_value, true_value);
    }
    if (const And *andCond = condition.as<And>()) {
      Expr tmpExpr = Select::make(andCond->a, true_value, false_value);
      return Select::make(andCond->b, tmpExpr, false_value);
    }
    if (const Or *orCond = condition.as<Or>()) {
      Expr tmpExpr = Select::make(orCond->a, true_value, false_value);
      return Select::make(orCond->b, true_value, tmpExpr);
    }
    return Select::make(condition, true_value, false_value);
  }

 private:
  template <typename T>
  Expr MulExprMutator(Expr &imm, const T *op) {
    Expr l = Mutate(op->a);
    Expr r = Mutate(op->b);

    // The precision of fp16 is low. We found that in some scenarios,
    // after the equivalent change of the calculation equation in the MulExprMutator,
    // the results will have precision errors.
    if (op->type.is_float16()) {
      return Mul::make(T::make(l, r), imm);
    }

    if (is_constant(l)) {
      return Mutate(T::make(ConstantFold<Mul>(imm, l), Mul::make(r, imm)));
    } else if (is_constant(r)) {
      return Mutate(T::make(ConstantFold<Mul>(imm, r), Mul::make(l, imm)));
    }
    return Mul::make(T::make(l, r), imm);
  }

  template <typename T>
  Expr ConstantFold(const Expr &a, const Expr &b) {
    CHECK(a.type().is_int() || a.type().is_uint() || a.type().is_float());
    if (a.type() != b.type()) {
      CHECK(a.type() == b.type());
    }
    CHECK(a.type() == b.type());
    if (const IntImm *int_a = a.as<IntImm>()) {
      const IntImm *int_b = b.as<IntImm>();
      return IntImm::make(a.type(), ComputeConstant<int64_t, T>(int_a->value, int_b->value));
    }
    if (const UIntImm *uint_a = a.as<UIntImm>()) {
      const UIntImm *uint_b = b.as<UIntImm>();
      return UIntImm::make(a.type(), ComputeConstant<uint64_t, T>(uint_a->value, uint_b->value));
    }
    const FloatImm *float_a = a.as<FloatImm>();
    const FloatImm *float_b = b.as<FloatImm>();
    return FloatImm::make(a.type(), ComputeConstant<double, T>(float_a->value, float_b->value));
  }

  template <typename Data, typename Op>
  Data ComputeConstant(Data d1, Data d2) {
    if (Op::_type_key == Mul::_type_key) {
      return d1 * d2;
    }
    if (Op::_type_key == Div::_type_key) {
      return d1 / d2;
    }
    if (Op::_type_key == Add::_type_key) {
      return d1 + d2;
    }
    CHECK(Op::_type_key == Sub::_type_key);
    return d1 - d2;
  }

  bool IsCandidate(const Expr &e) {
    if (!e->IsInstance<Mul>()) {
      return false;
    }
    const Mul *mul = e.as<Mul>();
    bool is_left_constant = is_constant(mul->a);
    bool is_right_constant = is_constant(mul->b);
    if (is_left_constant && is_right_constant) {
      return false;
    }
    return mul->a.type().is_float() && mul->a.type() == mul->b.type();
  }

  ThreeAddressExprMutator &mutator_;
  Array<Expr> args_;
};  // namespace ir

class ExprOptMutator : public IRMutator {
 public:
  explicit ExprOptMutator(ThreeAddressExprMutator &mutator, const Array<Expr> &args) : mutator_(mutator), args_(args) {}
  ~ExprOptMutator() override = default;

  Expr Mutate(Expr expr) {
    IRMutator::Mutate(expr);
    std::sort(exprs_.begin(), exprs_.end(), [this](Expr &e1, Expr &e2) -> bool {
      bool is_const = is_constant(e1);
      if (is_const || is_constant(e2)) {
        return !is_const;
      }
      Array<Expr> args1 = ExprArgsFetcher(args_).GetArgs(e1);
      Array<Expr> args2 = ExprArgsFetcher(args_).GetArgs(e2);
      if (args1.size() != args2.size()) {
        return args1.size() > args2.size();
      }
      if (sign_map_[e1.get()] != sign_map_[e2.get()]) {
        return !sign_map_[e1.get()];
      }
      return e1->IsInstance<Mul>();
    });
    if (exprs_.size() < 3) {
      return expr;
    }
    if (is_constant(exprs_[exprs_.size() - 2])) {
      return RebuildExpr();
    }
    Expr e = exprs_.front();
    Array<Expr> args = ExprArgsFetcher(args_).GetArgs(e);
    e = exprs_[exprs_.size() - 3];
    CHECK(sign_map_.find(e.get()) != sign_map_.end());
    if (sign_map_[e.get()]) {
      e = exprs_[exprs_.size() - 2];
    }
    if (args.size() > ExprArgsFetcher(args_).GetArgs(e).size()) {
      expr = RebuildExpr();
    }
    return expr;
  }

  Expr Mutate_(const Select *op, const Expr &e) {
    InitExprStatusIfNeed(e);
    Expr expr = Select::make(op->condition, ExprOptMutator(mutator_, args_).Mutate(op->true_value),
                             ExprOptMutator(mutator_, args_).Mutate(op->false_value));
    exprs_.push_back(expr);
    UpdateExprStatus(e, expr);
    return expr;
  }

  Expr Mutate_(const Add *op, const Expr &e) { return AnalyzeBinaryOpExpr(op, e); }

  Expr Mutate_(const Sub *op, const Expr &e) { return AnalyzeBinaryOpExpr(op, e); }

  Expr Mutate_(const Mul *op, const Expr &e) { return AnalyzeBinaryOpExpr(op, e); }

  Expr Mutate_(const Div *op, const Expr &e) { return AnalyzeBinaryOpExpr(op, e); }

  Expr Mutate_(const Mod *op, const Expr &e) { return AnalyzeBinaryOpExpr(op, e); }

  Expr Mutate_(const FloorDiv *op, const Expr &e) { return AnalyzeBinaryOpExpr(op, e); }

  Expr Mutate_(const FloorMod *op, const Expr &e) { return AnalyzeBinaryOpExpr(op, e); }

  Expr Mutate_(const Min *op, const Expr &e) { return AnalyzeBinaryOpExpr(op, e); }

  Expr Mutate_(const Max *op, const Expr &e) { return AnalyzeBinaryOpExpr(op, e); }

  Expr Mutate_(const EQ *op, const Expr &e) { return AnalyzeBinaryOpExpr(op, e); }

  Expr Mutate_(const NE *op, const Expr &e) { return AnalyzeBinaryOpExpr(op, e); }

  Expr Mutate_(const LT *op, const Expr &e) { return AnalyzeBinaryOpExpr(op, e); }

  Expr Mutate_(const LE *op, const Expr &e) { return AnalyzeBinaryOpExpr(op, e); }

  Expr Mutate_(const GT *op, const Expr &e) { return AnalyzeBinaryOpExpr(op, e); }

  Expr Mutate_(const GE *op, const Expr &e) { return AnalyzeBinaryOpExpr(op, e); }

  Expr Mutate_(const And *op, const Expr &e) { return AnalyzeBinaryOpExpr(op, e); }

  Expr Mutate_(const Or *op, const Expr &e) { return AnalyzeBinaryOpExpr(op, e); }

  Expr Mutate_(const Let *op, const Expr &e) {
    InitExprStatusIfNeed(e);
    Expr expr = Let::make(op->var, ExprOptMutator(mutator_, args_).Mutate(op->value),
                          ExprOptMutator(mutator_, args_).Mutate(op->body));
    exprs_.push_back(expr);
    UpdateExprStatus(e, expr);
    return expr;
  }

  Expr Mutate_(const Cast *op, const Expr &e) {
    InitExprStatusIfNeed(e);
    Expr expr = Cast::make(op->type, ExprOptMutator(mutator_, args_).Mutate(op->value));
    exprs_.push_back(expr);
    UpdateExprStatus(e, expr);
    return expr;
  }

  Expr Mutate_(const Not *op, const Expr &e) {
    InitExprStatusIfNeed(e);
    Expr expr = Not::make(ExprOptMutator(mutator_, args_).Mutate(op->a));
    exprs_.push_back(expr);
    UpdateExprStatus(e, expr);
    return expr;
  }

  Expr Mutate_(const Load *op, const Expr &e) {
    InitExprStatusIfNeed(e);
    Expr expr = Load::make(op->type, op->buffer_var, ExprOptMutator(mutator_, args_).Mutate(op->index),
                           ExprOptMutator(mutator_, args_).Mutate(op->predicate));
    exprs_.push_back(expr);
    UpdateExprStatus(e, expr);
    return expr;
  }

  Expr Mutate_(const Reduce *op, const Expr &e) {
    InitExprStatusIfNeed(e);
    Array<Expr> source;
    for (Expr src : op->source) {
      source.push_back(ExprOptMutator(mutator_, args_).Mutate(src));
    }
    Expr expr = Reduce::make(op->combiner, source, op->axis, ExprOptMutator(mutator_, args_).Mutate(op->condition),
                             op->value_index);
    exprs_.push_back(expr);
    UpdateExprStatus(e, expr);
    return expr;
  }

  Expr Mutate_(const Shuffle *op, const Expr &e) {
    InitExprStatusIfNeed(e);
    Array<Expr> vectors;
    for (Expr v : op->vectors) {
      vectors.push_back(ExprOptMutator(mutator_, args_).Mutate(v));
    }
    Array<Expr> indices;
    for (Expr indic : op->indices) {
      indices.push_back(ExprOptMutator(mutator_, args_).Mutate(indic));
    }
    Expr expr = Shuffle::make(vectors, indices);
    exprs_.push_back(expr);
    UpdateExprStatus(e, expr);
    return expr;
  }

  Expr Mutate_(const Call *op, const Expr &e) {
    InitExprStatusIfNeed(e);
    Array<Expr> args;
    for (Expr arg : op->args) {
      args.push_back(ExprOptMutator(mutator_, args_).Mutate(arg));
    }
    Expr expr = Call::make(op->type, op->name, args, op->call_type, op->func, op->value_index);
    exprs_.push_back(expr);
    mutator_.AddBroadCastCallIfNeed(op, expr);
    UpdateExprStatus(e, expr);
    return expr;
  }

  Expr Mutate_(const Ramp *op, const Expr &e) {
    InitExprStatusIfNeed(e);
    Expr expr = Ramp::make(ExprOptMutator(mutator_, args_).Mutate(op->base),
                           ExprOptMutator(mutator_, args_).Mutate(op->stride), op->lanes);
    exprs_.push_back(expr);
    UpdateExprStatus(e, expr);
    return expr;
  }

  Expr Mutate_(const Broadcast *op, const Expr &e) {
    InitExprStatusIfNeed(e);
    Expr expr = Broadcast::make(ExprOptMutator(mutator_, args_).Mutate(op->value), op->lanes);
    exprs_.push_back(expr);
    UpdateExprStatus(e, expr);
    return expr;
  }

  Expr Mutate_(const IntImm *op, const Expr &e) { return SaveAutomicExpr(e); }

  Expr Mutate_(const UIntImm *op, const Expr &e) { return SaveAutomicExpr(e); }

  Expr Mutate_(const FloatImm *op, const Expr &e) { return SaveAutomicExpr(e); }

  Expr Mutate_(const StringImm *op, const Expr &e) { return SaveAutomicExpr(e); }

  Expr Mutate_(const Variable *op, const Expr &e) { return SaveAutomicExpr(e); }

 private:
  void InitExprStatusIfNeed(const Expr &e) {
    const Object *object_e = e.get();
    if (notation_map_.find(object_e) == notation_map_.end()) {
      notation_map_[object_e] = e->GetTypeKey();
    }
    if (sign_map_.find(object_e) == sign_map_.end()) {
      sign_map_[object_e] = true;
    }
  }

  void UpdateExprStatus(const Expr &before, const Expr &after) {
    const Object *b = before.get();
    const Object *a = after.get();
    CHECK(notation_map_.find(b) != notation_map_.end());
    notation_map_[a] = notation_map_[b];
    CHECK(sign_map_.find(b) != sign_map_.end());
    sign_map_[a] = sign_map_[b];
  }

  bool IsNewRoot(const Expr &e) {
    CHECK(notation_map_.find(e.get()) != notation_map_.end());
    std::string root = notation_map_[e.get()];
    std::string type_key = e->GetTypeKey();
    return !((root == Add::_type_key || root == Sub::_type_key) &&
             (type_key == Add::_type_key || type_key == Sub::_type_key)) &&
           !((root == Mul::_type_key || root == Div::_type_key) &&
             (type_key == Mul::_type_key || type_key == Div::_type_key));
  }

  template <typename T>
  Expr AnalyzeBinaryOpExpr(const T *op, const Expr &e) {
    InitExprStatusIfNeed(e);
    const Object *object_e = e.get();
    std::string root_of_e = notation_map_[object_e];
    bool pos_of_e = sign_map_[object_e];
    std::string type_key = e->GetTypeKey();
    Expr expr = e;
    if (IsNewRoot(e)) {
      expr = T::make(ExprOptMutator(mutator_, args_).Mutate(op->a), ExprOptMutator(mutator_, args_).Mutate(op->b));
      notation_map_[expr.get()] = root_of_e;
      sign_map_[expr.get()] = pos_of_e;
      exprs_.push_back(expr);
    } else {
      notation_map_[op->a.get()] = root_of_e;
      notation_map_[op->b.get()] = root_of_e;
      sign_map_[op->a.get()] = pos_of_e;
      sign_map_[op->b.get()] = (type_key == Sub::_type_key || type_key == Div::_type_key) ? !pos_of_e : pos_of_e;
      expr = T::make(IRMutator::Mutate(op->a), IRMutator::Mutate(op->b));
    }
    UpdateExprStatus(e, expr);
    return expr;
  }

  Expr SaveAutomicExpr(const Expr &e) {
    InitExprStatusIfNeed(e);
    exprs_.push_back(e);
    return e;
  }

  Expr RebuildExpr() {
    CHECK(!exprs_.empty());
    Expr expr = exprs_.back();
    exprs_.pop_back();
    while (!exprs_.empty()) {
      expr = RebuildExpr(expr, exprs_.back());
      exprs_.pop_back();
    }
    return expr;
  }

  Expr RebuildExpr(const Expr &expr1, const Expr &expr2) {
    Expr expr = expr1;
    Expr opnd = expr2;
    if (sign_map_[expr2.get()] && !sign_map_[expr1.get()]) {
      expr = expr2;
      opnd = expr1;
    }

    if ((sign_map_[expr1.get()] && sign_map_[expr2.get()]) || (!sign_map_[expr1.get()] && !sign_map_[expr2.get()])) {
      if (notation_map_[expr1.get()] == Add::_type_key || notation_map_[expr1.get()] == Sub::_type_key) {
        expr = Add::make(expr, opnd);
      } else {
        expr = Mul::make(expr, opnd);
      }
    } else {
      if (notation_map_[expr1.get()] == Add::_type_key || notation_map_[expr1.get()] == Sub::_type_key) {
        expr = Sub::make(expr, opnd);
      } else {
        expr = Div::make(expr, opnd);
      }
    }
    notation_map_[expr.get()] = notation_map_[expr1.get()];
    sign_map_[expr.get()] = sign_map_[expr1.get()] || sign_map_[expr2.get()];
    return expr;
  }

  ThreeAddressExprMutator &mutator_;
  Array<Expr> args_;
  std::vector<Expr> exprs_;
  std::unordered_map<const Object *, std::string> notation_map_;
  std::unordered_map<const Object *, bool> sign_map_;
};

class InferUpperBound {
 private:
  class Bound {
   public:
    Expr min;
    Expr max;

    static Bound make(const Range range) {
      Bound bound;
      bound.min = range->min;
      bound.max = range->min + range->extent;
      return bound;
    }
    static Bound make(const Expr min, const Expr max) {
      Bound bound;
      bound.min = min;
      bound.max = max;
      return bound;
    }
  };

  Bound infer_range(const Expr &expr) {
    air::arith::Analyzer analyzer_;
    if (expr.as<IntImm>() || expr.as<UIntImm>() || expr.as<FloatImm>()) {
      return Bound::make(expr, expr);
    } else if (expr.as<Variable>()) {
      auto var = expr.as<Variable>()->name_hint;
      if (binds.count(var) > 0) {
        Bound var_min_range = infer_range(binds[var].min);
        Bound var_max_range = infer_range(binds[var].max);
        return Bound::make(var_min_range.min, var_max_range.max);
      } else {
        return Bound::make(expr, expr);
      }
    } else if (expr.as<Add>()) {
      auto add = expr.as<Add>();
      Bound bound_a = infer_range(add->a);
      Bound bound_b = infer_range(add->b);
      return Bound::make(Simplify_cce(bound_a.min + bound_b.min), Simplify_cce(bound_a.max + bound_b.max));
    } else if (expr.as<Sub>()) {
      auto sub = expr.as<Sub>();
      Bound bound_a = infer_range(sub->a);
      Bound bound_b = infer_range(sub->b);
      return Bound::make(Simplify_cce(bound_a.min - bound_b.max), Simplify_cce(bound_a.max - bound_b.min));
    } else if (expr.as<Mul>()) {
      auto mul = expr.as<Mul>();
      Bound bound_a = infer_range(mul->a);
      Bound bound_b = infer_range(mul->b);
      Bound bound;
      if (analyzer_.CanProve(bound_a.min >= 0) && analyzer_.CanProve(bound_b.min >= 0)) {
        bound.min = Simplify_cce(bound_a.min * bound_b.min);
      } else {
        bound.min = expr;
      }
      if (analyzer_.CanProve(bound_a.max >= 0) && analyzer_.CanProve(bound_b.max >= 0)) {
        bound.max = Simplify_cce(bound_a.max * bound_b.max);
      } else {
        bound.max = expr;
      }
      return bound;
    } else if (expr.as<Div>()) {
      auto div = expr.as<Div>();
      Bound bound_a = infer_range(div->a);
      Bound bound_b = infer_range(div->b);
      Bound bound;
      if (analyzer_.CanProve(bound_a.min >= 0) && analyzer_.CanProve(bound_b.max > 0)) {
        bound.min = Simplify_cce(bound_a.min / bound_b.max);
      } else {
        bound.min = expr;
      }
      if (analyzer_.CanProve(bound_a.max >= 0) && analyzer_.CanProve(bound_b.min > 0)) {
        bound.max = Simplify_cce(bound_a.max / bound_b.min);
      } else {
        bound.max = expr;
      }
      return bound;
    } else if (expr.as<Min>()) {
      auto min_expr = expr.as<Min>();
      Bound bound_a = infer_range(min_expr->a);
      Bound bound_b = infer_range(min_expr->b);
      return Bound::make(Simplify_cce(min(bound_a.min, bound_b.min)), Simplify_cce(min(bound_a.max, bound_b.max)));
    } else if (expr.as<Max>()) {
      auto max_expr = expr.as<Max>();
      Bound bound_a = infer_range(max_expr->a);
      Bound bound_b = infer_range(max_expr->b);
      return Bound::make(Simplify_cce(max(bound_a.min, bound_b.min)), Simplify_cce(max(bound_a.max, bound_b.max)));
    } else {
      return Bound::make(expr, expr);
    }
  }

 public:
  Expr run(const Expr &expr, const std::unordered_map<VarExpr, Range, air::NodeHash, air::NodeEqual> &dom_map) {
    for (auto bind : dom_map) {
      binds.emplace(bind.first->name_hint, Bound::make(bind.second));
    }
    Bound bound = infer_range(expr);
    return bound.max;
  }

 private:
  std::unordered_map<std::string, Bound> binds;
};

bool IsReductionOp(const Provide *op) {
  Tensor output = Downcast<Operation>(op->func).output(op->value_index);
  std::vector<bool> rhs_reduce;
  int call_num = 0;

  PostOrderVisit(op->value, [&rhs_reduce, output, &call_num, op](const NodeRef &node) {
    if (const Call *call = node.as<Call>()) {
      if (call->call_type == Call::CallType::Halide) {
        call_num++;
        if (Downcast<Operation>(call->func).output(call->value_index) == output) {
          bool match = true;
          for (size_t i = 0; i < call->args.size(); ++i) {
            if (!Equal(call->args[i], op->args[i])) {
              match = false;
            }
          }
          // A[j, j] = log(B[j, j])
          if (CountVars(call->args) == 1 && AllVars(call->args) > 1) {
            match = false;
          }
          rhs_reduce.push_back(match);
        }
      }
    }
  });

  if (rhs_reduce.size() != 1) {
    return false;
  }
  return rhs_reduce[0];
}

// Expand complicated expression to three address code
// Instruction selection is applied
class ThreeAddressStmtMutator : public IRMutator {
 public:
  ThreeAddressStmtMutator(bool reuse_variable, int minimum_split, bool cross_stmt_simplify)
      : reuse_variable_(reuse_variable), minimum_split_(minimum_split), cross_stmt_simplify_(cross_stmt_simplify) {}
  ~ThreeAddressStmtMutator() override = default;

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    // skip cube operators (conv2d, matmul)
    bool is_reduction = IsReductionOp(op);
    air::arith::Analyzer analyzer_;
    Expr value = analyzer_.rewrite_simplify(op->value);
    if (!PolyUnsupportedExprChecker().isSupported(value)) {
      value = Simplify_cce(op->value);
      if (!PolyUnsupportedExprChecker().isSupported(value)) {
        value = op->value;
      }
    }

    const Call *call = op->value.as<Call>();
    if (call && (call->name == "mad" || call->name == "load_im2col_c1_buf" || call->name == "divide_var")) {
      return IRMutator::Mutate_(op, s);
    }

    Tensor output = Downcast<Operation>(op->func).output(op->value_index);

    // special vectorization treatment for reduce operators
    Array<Expr> args = op->args;
    Array<Expr> shape = output->shape;

    if (is_reduction) {
      VarSet spatial_vars;
      std::vector<Var> all_vars;

      // collect reduction vars
      for (size_t i = 0; i < op->args.size(); ++i) {
        GatherVars(op->args[i], &spatial_vars);
      }
      all_vars = std::vector<Var>(spatial_vars.begin(), spatial_vars.end());
      GatherVars(value, &all_vars);

      VarSet reduce_vars;
      for (const auto &x : all_vars) {
        if (!spatial_vars.count(x)) {
          reduce_vars.insert(x);
        }
      }

      std::unordered_map<Var, VarSet, air::NodeHash, air::NodeEqual> edges;
      std::unordered_map<Var, size_t, air::NodeHash, air::NodeEqual> degree;
      VarSet new_args_vars;

      // sort reduction vars  (Here we use a simplified version, only deal with the relation
      // between spatial and reduce vars, while ignore the relation among reduce vars)
      // - 1. collect relations
      PostOrderVisit(value, [&reduce_vars, &spatial_vars, &new_args_vars, &edges](const NodeRef &node) {
        if (node->IsInstance<Call>() && node.as<Call>()->call_type == Call::Halide) {
          const Array<Expr> &call_args = node.as<Call>()->args;
          CHECK(call_args.defined());
          for (size_t i = 0; i < call_args.size(); ++i) {
            for (size_t j = i + 1; j < call_args.size(); ++j) {
              if (is_constant(call_args[i]) || !call_args[j]->IsInstance<Variable>()) {
                continue;
              }
              std::vector<Var> call_arg_vars;
              GatherVars(call_args[i], &call_arg_vars);
              if (call_arg_vars.size() == 1) {
                Var vi = call_arg_vars.front(), vj = Downcast<Var>(call_args[j]);
                if (!Equal(vi, vj)) {
                  new_args_vars.insert(vi);
                  new_args_vars.insert(vj);
                  edges[vi].insert(vj);
                }
              }
            }
          }
        }
      });

      // for non-variable terms, attach them to its previous variable term
      std::unordered_map<Var, std::vector<Expr>, air::NodeHash, air::NodeEqual> following_terms_arg;
      std::unordered_map<Var, std::vector<Expr>, air::NodeHash, air::NodeEqual> following_terms_shape;
      VarSet vars_add_to_args(reduce_vars.begin(), reduce_vars.end());

      size_t i = 0;
      while (i < args.size()) {
        size_t j = i + 1;
        if (!is_constant(args[i])) {
          std::vector<Var> arg_vars;
          GatherVars(args[i], &arg_vars);
          for (const auto &x : arg_vars) {
            Var vi = x;
            if (new_args_vars.size() == 0 && vars_add_to_args.size() == 0) {
              vars_add_to_args.insert(vi);
            } else if (new_args_vars.find(vi) != new_args_vars.end()) {
              vars_add_to_args.insert(vi);
              size_t k = j;
              while (k < args.size() && is_constant(args[k])) {
                following_terms_arg[vi].push_back(args[k]);
                following_terms_shape[vi].push_back(shape[k]);
                k++;
              }
            }
          }
        }
        i = j;
      }

      // topo-sort
      Array<Expr> new_args;
      Array<Expr> new_shape;

      size_t check_ct = 0;
      std::queue<Var> out_queue;

      for (const auto &iter : edges) {
        for (const auto &to : iter.second) {
          degree[to]++;
        }
      }

      for (const auto &x : all_vars) {
        if (degree[x] == 0) {
          out_queue.push(x);
        }
      }

      while (check_ct < all_vars.size()) {
        if (out_queue.empty()) {
          size_t min_degree = std::numeric_limits<int>::max();
          for (const auto &x : all_vars) {
            if (degree[x] > 0 && degree[x] < min_degree) {
              min_degree = degree[x];
            }
          }
          for (const auto &x : reduce_vars) {
            if (degree[x] == min_degree) {
              out_queue.push(x);
              degree[x] = 0;
              break;
            }
          }
          if (out_queue.empty()) {
            for (const auto &x : vars_add_to_args) {
              if (degree[x] == min_degree) {
                out_queue.push(x);
                degree[x] = 0;
                break;
              }
            }
          }
        }
        check_ct++;
        const Var x = out_queue.front();
        out_queue.pop();

        if (vars_add_to_args.count(x)) {
          new_args.push_back(x);
          CHECK_GT(dom_map.count(x), 0);
          new_shape.push_back(dom_map[x]->min + dom_map[x]->extent);

          CHECK_EQ(following_terms_arg[x].size(), following_terms_shape[x].size());
          for (size_t dim = 0; dim < following_terms_arg[x].size(); dim++) {
            const Expr &arg = following_terms_arg[x][dim];
            const Expr &shape_ = following_terms_shape[x][dim];
            bool index_is_const_zero = Equal(arg, Expr(0));
            bool dim_extent_is_one = Equal(shape_, Expr(1));
            if (!index_is_const_zero && !dim_extent_is_one) {
              new_args.push_back(arg);
              new_shape.push_back(shape_);
            }
          }
        }

        for (const auto &y : edges[x]) {
          if (--degree[y] == 0) {
            out_queue.push(y);
          }
        }
      }
      CHECK_EQ(check_ct, all_vars.size());
      args = !new_args.empty() ? new_args : args;
      shape = !new_shape.empty() ? new_shape : shape;
      CHECK_EQ(args.size(), shape.size());
    }

    // find broadcast call
    output_ = output;
    args_ = args;
    static_cast<void>(this->Mutate(op->value));
    // mutate according to the result of instruction selection
    ThreeAddressExprMutator mutator(output, args, op->args, shape, broadcast_, is_reduction, cross_stmt_simplify_,
                                    is_simple_);
    if (cross_stmt_simplify_) {
      // Bring over the common exprs from previous stage
      mutator.SetCommonExpr(global_common_expr_);
    }
    if (is_simple_) {
      value = ExprOptMutator(mutator, args_).Mutate(value);
    }
    value = InstructionMutator(mutator, args_).Mutate(value);
    value = mutator.Mutate(value);
    if (cross_stmt_simplify_) {
      // Take back the common exprs for next stages
      global_common_expr_ = mutator.GetCommonExpr();
    }

    std::unordered_set<Tensor> replaced_tensors;

    if (reuse_variable_ && (static_cast<int>(mutator.assign_stmt.size()) > minimum_split_)) {
      std::unordered_map<FunctionRef, int, NodeHash, NodeEqual> tensors_last_id;
      for (size_t ii = 0; ii < mutator.assign_stmt.size(); ii++) {
        const auto tmp_op = mutator.assign_stmt[ii].as<Provide>();
        std::unordered_set<Tensor> tmpTensors = GetExprTensors(tmp_op->value);
        for (auto it : tmpTensors) {
          if (mutator.IsTmpTensor(it)) {
            tensors_last_id[it->op] = static_cast<int>(ii);
          }
        }
      }

      for (int ii = 0; ii < (static_cast<int>(mutator.assign_stmt.size()) - 1); ii++) {
        const auto tmp_op = mutator.assign_stmt[ii].as<Provide>();
        std::unordered_set<Tensor> tmpTensors = GetExprTensors(tmp_op->value);
      }
      LOG(INFO) << "Replaced " << replaced_tensors.size() << " from a total of " << mutator.assign_stmt.size()
                << " tensors.";
    }

    // remove the last useless copy
    if (value->IsInstance<Call>() && mutator.imm_ops.count(value.as<Call>()->func)) {
      const auto last_provide = mutator.assign_stmt.back().as<Provide>();
      CHECK(last_provide != nullptr);
      value = last_provide->value;

      mutator.assign_stmt.pop_back();
      mutator.imm_tensors.pop_back();
    }

    mutator.assign_stmt.push_back(Provide::make(op->func, op->value_index, value, op->args));

    // store info for adding Realize/Produce
    if (replaced_tensors.empty()) {
      if (split_to_.count(output)) {
        for (auto &i : mutator.imm_tensors) {
          split_to_[output].push_back(i);
        }
      } else {
        split_to_[output] = mutator.imm_tensors;
      }
    } else {
      for (auto &i : mutator.imm_tensors) {
        if (replaced_tensors.find(i) == replaced_tensors.end()) split_to_[output].push_back(i);
      }
    }
    op_indices_[output->op].insert(output->value_index);

    return Block::make(mutator.assign_stmt);
  }

  Stmt Mutate_(const Realize *op, const Stmt &s) final {
    realize_node_[Operation(GetObjPtr(op->func.get())).output(op->value_index)] = op;
    return IRMutator::Mutate_(op, s);
  }

  Expr Mutate_(const Call *op, const Expr &e) final {
    // a[i] = a[i] + b[i, j]
    if (op->call_type == Call::CallType::Halide && Downcast<Operation>(op->func).output(op->value_index) != output_ &&
        CountVars(args_) > CountVars(e)) {
      broadcast_.insert(op);
    }
    return IRMutator::Mutate_(op, e);
  }

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    FunctionRef func = Downcast<FunctionRef>(op->node);
    attr_node_[func] = op;
    Stmt ret = IRMutator::Mutate_(op, s);
    if (op_indices_.count(func)) {
      Stmt inner = ret;
      for (int idx : op_indices_[func]) {
        Tensor output = Downcast<Operation>(func).output(idx);
        const Realize *ref_real = realize_node_[output];
        const AttrStmt *ref_attr = attr_node_[output->op];
        for (const auto &x : split_to_.at(output)) {
          Region bounds;
          for (size_t i = 0; i < x->shape.size(); ++i) {
            Expr upper_bound = InferUpperBound().run(x->shape[i], dom_map);
            bounds.push_back(Range::make_by_min_extent(0, upper_bound));
          }
          inner = Realize::make(x->op, x->value_index, x->dtype, bounds, ref_real->condition, inner);
          inner = AttrStmt::make(x->op, ref_attr->attr_key, ref_attr->value, inner);
        }
      }
      return inner;
    }
    return ret;
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (loop_level == 0) {
      is_simple_ = IsSimpleFor(op);
    }
    loop_level++;
    dom_map[op->loop_var] = Range::make_by_min_extent(op->min, op->extent);
    Stmt stmt = IRMutator::Mutate_(op, s);
    loop_level--;
    if (loop_level == 0) {
      is_simple_ = true;
    }
    return stmt;
  }

  static bool IsSimpleFor(const For *op) {
    if (const For *sub_for = op->body.as<For>()) {
      return IsSimpleFor(sub_for);
    }
    if (const Block *block = op->body.as<Block>()) {
      return IsSimpleBlock(block);
    }
    return op->body->IsInstance<Provide>();
  }

 private:
  static bool IsSimpleBlock(const Block *op) {
    if (op->first->IsInstance<Provide>() && op->rest->IsInstance<Provide>()) {
      return true;
    }
    if (op->first->IsInstance<Block>() && op->rest->IsInstance<Block>()) {
      return IsSimpleBlock(op->first.as<Block>()) && IsSimpleBlock(op->rest.as<Block>());
    }
    if (op->first->IsInstance<Provide>() && op->rest->IsInstance<Block>()) {
      return IsSimpleBlock(op->rest.as<Block>());
    }
    if (op->first->IsInstance<Block>() && op->rest->IsInstance<Provide>()) {
      return IsSimpleBlock(op->first.as<Block>());
    }
    return false;
  }

  std::unordered_map<Tensor, std::vector<Tensor>> split_to_;

  std::unordered_map<FunctionRef, std::set<int>, air::NodeHash, air::NodeEqual> op_indices_;
  std::unordered_map<Tensor, const Realize *> realize_node_;
  std::unordered_map<FunctionRef, const AttrStmt *, air::NodeHash, air::NodeEqual> attr_node_;

  std::unordered_map<VarExpr, Range, air::NodeHash, air::NodeEqual> dom_map;

  std::unordered_map<size_t, std::pair<Expr, Expr>> global_common_expr_;

  int loop_level{0};
  bool is_simple_{true};

  // mark broadcast
  Tensor output_;
  Array<Expr> args_;
  std::unordered_set<const Call *> broadcast_;
  bool reuse_variable_;
  int minimum_split_;
  bool cross_stmt_simplify_;
};

class ExprArgsExtract : public IRVisitor {
 public:
  explicit ExprArgsExtract(Array<Expr> args) : args_(args) {}
  ~ExprArgsExtract() override = default;

  Array<Expr> GetArgs(const Expr &e) {
    Visit(e);
    return args_;
  }

  void Visit_(const Call *op) override {
    if (op->call_type == Call::CallType::Halide) {
      for (Expr arg : op->args) {
        if (!Contain(args_, arg)) {
          args_.push_back(arg);
        }
      }
    }
  }

 private:
  bool Contain(const Array<Expr> &args, const Expr &arg) {
    for (Expr e : args) {
      if (e.same_as(arg)) {
        return true;
      }
    }
    return false;
  }

  Array<Expr> args_;
};

class LoopMutator : public IRMutator {
 public:
  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (loop_vars_.empty() && !ThreeAddressStmtMutator::IsSimpleFor(op)) {
      return s;
    }
    loop_vars_.push_back(op);
    Stmt stmt = IRMutator::Mutate(op->body);
    if (!provides_.empty()) {
      // This sort can generate a wrong schedule order,
      // sometimes it the statement wit small number of iterator must be at the beginning
      // sometimes it the statement wit small number of iterator must be at the end
      // sometimes it the statement wit small number of iterator must be mixe at the begin and at the end
      // need a dependence analyze to be sure that they can be move
      // provides_.sort([](const Provide *s1, const Provide *s2) -> bool { return s1->args.size() < s2->args.size(); });
      while (!provides_.empty()) {
        SplitProvides();
      }
    }
    for (size_t index = 0; index < stmts_.size(); ++index) {
      if (IsContain(args_[index], loop_vars_.back()->loop_var)) {
        stmts_[index] = For::make(op->loop_var, op->min, op->extent, op->for_type, op->device_api, stmts_[index]);
      }
    }
    loop_vars_.pop_back();
    if (loop_vars_.empty()) {
      stmt = stmts_.back();
      for (auto iter = ++stmts_.rbegin(); iter != stmts_.rend(); iter++) {
        stmt = Block::make(*iter, stmt);
      }
      stmts_.clear();
      args_.clear();
    }
    return stmt;
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    if (!loop_vars_.empty()) {
      provides_.push_back(op);
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  void SplitProvides() {
    const Provide *provide = provides_.back();
    Stmt stmt = Provide::make(provide->func, provide->value_index, provide->value, provide->args);
    provides_.pop_back();
    while (!provides_.empty()) {
      const Provide *next = provides_.back();
      if (provide->args.size() != next->args.size()) {
        break;
      }
      stmt = Block::make(Provide::make(next->func, next->value_index, next->value, next->args), stmt);
      provides_.pop_back();
    }
    stmts_.insert(stmts_.begin(), stmt);
    Array<Expr> all_args = ExprArgsExtract(provide->args).GetArgs(provide->value);
    args_.insert(args_.begin(), all_args);
  }

  bool IsContain(const Array<Expr> &args, const Var &var) {
    VarSet all_vars;
    for (Expr e : args) {
      GatherVars(e, &all_vars);
    }
    for (auto v : all_vars) {
      if (v.same_as(var)) {
        return true;
      }
    }
    return false;
  }

  std::list<const For *> loop_vars_{};
  std::list<const Provide *> provides_{};
  std::vector<Stmt> stmts_{};
  std::vector<Array<Expr>> args_{};
};

Stmt ToThreeAddress(Stmt stmt, bool reuse_variable, int minimum_split, bool cross_stmt_simplify) {
  stmt = ThreeAddressStmtMutator(reuse_variable, minimum_split, cross_stmt_simplify).Mutate(stmt);
  stmt = LoopMutator().Mutate(stmt);
  return Simplify_cce(stmt);
}
}  // namespace ir
}  // namespace akg
