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
#include <tvm/ir.h>
#include <tvm/expr.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <cmath>
#include "pass/utils.h"
#include "pass/rewrite_simplify_cce.h"

namespace akg {
namespace ir {
class TensorOperandFinder : public IRVisitor {
 public:
  bool Find(const Expr &e) {
    Visit(e);
    return find_;
  }

  void Visit_(const Call *op) override {
    if (op->func.as<OperationNode>()) {
      find_ = true;
    }
    IRVisitor::Visit_(op);
  }

 private:
  bool find_{false};
};

class DivRewriter : public IRMutator {
 public:
  Expr Mutate_(const Div *op, const Expr &e) final {
    const std::string productName = GetProductName();
    // In mini need mutate div to rec
    if (productName == "mini") {
      if (op->b.type().is_float()) {
        auto it =
          std::find_if(cache_.begin(), cache_.end(), [&](const std::pair<Expr, Expr> &i) { return Equal(e, i.first); });
        if (it != cache_.end()) {
          return (*it).second;
        }
        // rewrite division to vrec intrinsic
        // a / b -> a * (vrec(b))
        auto new_e = Simplify_cce(Mul::make(
          Mutate(op->a), Call::make(op->type, "rec", Array<Expr>{Mutate(op->b)}, Call::CallType::PureIntrinsic)));
        cache_.Set(e, new_e);
        return new_e;
      }
    }
    return IRMutator::Mutate_(op, e);
  }

 private:
  Map<Expr, Expr> cache_;
};

class RecRewriter : public IRMutator {
 public:
  Expr Mutate_(const Call *op, const Expr &e) final {
    if (op->name == "rec" && op->args.size() == 1) {
      // rewrite division to vrec intrinsic
      // rec(4) -> 1/4
      CHECK(isZero(op->args[0]) == false) << " Invalid expression! div 0 error ";
      if (isImm(op->args[0])) {
        return Simplify_cce(1 / op->args[0]);
      }
    }
    return IRMutator::Mutate_(op, e);
  }
};

// rewrite rsqrt to 1 / sqrt intrinsic on cloud
// rsqrt(data) -> 1 / sqrt(data)
class RsqrtRewriter : public IRMutator {
 public:
  Expr Mutate_(const Call *op, const Expr &e) final {
    const std::string productName = GetProductName();
    if (productName == "cloud" && op->name == "rsqrt" && op->args.size() == 1) {
      CHECK(isZero(op->args[0]) == false) << " Invalid expression! div 0 error ";
      return Simplify_cce(1 / Call::make(op->type, "sqrt", op->args, Call::CallType::PureIntrinsic));
    }
    return IRMutator::Mutate_(op, e);
  }
};

// float(floor(A[i])) --> floor(A[i]) when type(floor) = float
class RmCast : public IRMutator {
 public:
  Expr Mutate_(const Cast *op, const Expr &e) final {
    Expr cast = IRMutator::Mutate_(op, e);
    const Cast *ca = cast.as<Cast>();
    CHECK(ca);
    if (const Cast *cast_ca = ca->value.as<Cast>()) {
      if (cast_ca->type == op->type) {
        return ca->value;
      }
    }
    return cast;
  }
};

// a*b if(a==0 || b==0) return 0;
class MulZeroOpt : public IRMutator {
 public:
  Expr Mutate_(const Mul *op, const Expr &e) final {
    if (isZero(op->a)) return op->a;
    if (isZero(op->b)) return op->b;
    return IRMutator::Mutate_(op, e);
  }
};

// pow(A, B) --> exp(log(|A|)*B)
// if A is negative and B is odd, the result need to multiply -1
class PowRewriter : public IRMutator {
 public:
  Expr Mutate_(const Call *op, const Expr &e) final {
    if (op->name == "pow") {
      if (op->args[0].as<FloatImm>() && op->args[1].as<FloatImm>()) {
        return FloatImm::make(op->args[0].as<FloatImm>()->type,
                              pow(op->args[0].as<FloatImm>()->value, op->args[1].as<FloatImm>()->value));
      }

      if (isZero(Mutate(op->args[0]))) {
        return FloatImm::make(op->type, 1.0);
      }

      auto dtype = e.type();
      auto a = Mutate(op->args[0]);
      auto b = Mutate(op->args[1]);
      auto a_abs = Call::make(dtype, "fabs", {a}, Call::PureIntrinsic);
      auto exp_log_value = Call::make(
        dtype, "exp", {Mul::make(Call::make(dtype, "log", {a_abs}, Call::PureIntrinsic), b)}, Call::PureIntrinsic);

      // For the pow, if value of a is a negative number, the corresponding value of b must be an integer,
      // for example, the corresponding value of b is 1.0, -2.0, 3.0.
      // if b is odd, `-2 * (ceil(b*0.5)*2 - b) + 1` is -1, if b is even, the value is 1
      auto round_dtype = Int(32);
      auto b_half_ceil =
        Cast::make(dtype, Call::make(round_dtype, "ceil", {Mul::make(b, make_const(dtype, 0.5))}, Call::PureIntrinsic));
      auto odd = Add::make(Mul::make(Sub::make(Mul::make(b_half_ceil, make_const(dtype, 2)), b), make_const(dtype, -2)),
                           make_const(dtype, 1));

      return Select::make(LT::make(a, make_const(dtype, 0)), Mul::make(odd, exp_log_value), exp_log_value);
    }
    return IRMutator::Mutate_(op, e);
  }
};

// tanh(A) --> (exp(A)-exp(-A))/(exp(A)+exp(-A))
class TanhRewriter : public IRMutator {
 public:
  Expr Mutate_(const Call *op, const Expr &e) final {
    if (op->name == "tanh") {
      auto type = op->args[0].type();
      Expr a = Mutate(op->args[0]);
      Expr exp_a = Call::make(type, "exp", {a}, Call::PureIntrinsic);
      Expr exp_n_a = Call::make(type, "exp", {-a}, Call::PureIntrinsic);
      Expr num = Sub::make(exp_a, exp_n_a);
      Expr den = Add::make(exp_a, exp_n_a);

      return Div::make(num, den);
    }
    return IRMutator::Mutate_(op, e);
  }
};

// !a --> sub(1,a)
class NotRewriter : public IRMutator {
 public:
  Expr Mutate_(const Not *op, const Expr &e) final {
    // vsub instrunction does not support bool or int8 type
    air::DataType sub_dtype = Float(16);
    Expr res = Sub::make(make_const(sub_dtype, 1.0), Cast::make(sub_dtype, Mutate(op->a)));
    return Cast::make(e.type(), res);
  }
};

// In tvm, mod is truncmod
// mod(a,b) --> a - b*trunc(a/b)
// floormod(a,b) --> a - b*floor(a/b)
class ModRewriter : public IRMutator {
 public:
  template <typename T>
  Expr MutateModOp(const T *op, const Expr &e, const std::string &round_mode) {
    if (TensorOperandFinder().Find(e)) {
      // vdiv instruction dose not support int
      if (e.type().is_float()) {
        Expr a = Mutate(op->a);
        Expr b = Mutate(op->b);
        Expr qutotient = Div::make(a, b);
        Expr round_qutotient = Call::make(Int(32), round_mode, {qutotient}, Call::CallType::PureIntrinsic);
        return Sub::make(a, Mul::make(b, Cast::make(b.type(), round_qutotient)));
      }
    }
    return IRMutator::Mutate_(op, e);
  }

  Expr Mutate_(const Mod *op, const Expr &e) final { return MutateModOp(op, e, "trunc"); }
  Expr Mutate_(const FloorMod *op, const Expr &e) final { return MutateModOp(op, e, "floor"); }
};

// The trunc instruction is not supported on mini
// trunc(x) --> min(ceil(x),0) + max(floor(x),0)
class TruncRewriter : public IRMutator {
 public:
  Expr Mutate_(const Call *op, const Expr &e) final {
    if (op->name == "trunc" && TensorOperandFinder().Find(e)) {
      const std::string productName = GetProductName();
      if (productName != "cloud") {
        auto round_type = e.type();
        if (!round_type.is_int()) {
          round_type = Int(32);
        }
        Expr x = Mutate(op->args[0]);
        Expr ceil_x = Call::make(round_type, "ceil", {x}, Call::CallType::PureIntrinsic);
        Expr floor_x = Call::make(round_type, "floor", {x}, Call::CallType::PureIntrinsic);
        Expr zero = make_zero(round_type);
        Expr trunc_x = Add::make(Min::make(ceil_x, zero), Max::make(floor_x, zero));
        if (trunc_x.type() != e.type()) {
          trunc_x = Cast::make(e.type(), trunc_x);
        }
        return trunc_x;
      }
    }
    return IRMutator::Mutate_(op, e);
  }
};

// tvm_if_then_else(cond, true, false) --> select(cond, true, false)
class IfThenElseRewriter : public IRMutator {
  Expr Mutate_(const Call *op, const Expr &e) final {
    if (op->name == "tvm_if_then_else") {
      CHECK_EQ(op->args.size(), 3);
      auto select = Select::make(Mutate(op->args[0]), Mutate(op->args[1]), Mutate(op->args[2]));
      return select;
    }
    return IRMutator::Mutate_(op, e);
  }
};

// cmp(a,b) --> select(cmp(a,b), 1, 0)
class CmpRewriter : public IRMutator {
 public:
  template <typename T>
  Expr MutateCmpOp(const T *op, const Expr &e) {
    if (TensorOperandFinder().Find(e)) {
      Expr a = Mutate(op->a);
      Expr b = Mutate(op->b);
      auto sel_type = a.type();
      Expr sel = Select::make(T::make(a, b), make_const(sel_type, 1), make_zero(sel_type));
      // vector instrunction does not support float32 cast to bool(int8)
      if (sel.type() == Float(32)) {
        sel = Cast::make(Float(16), sel);
      }

      return Cast::make(e.type(), sel);
    }
    return IRMutator::Mutate_(op, e);
  }

  Expr Mutate_(const EQ *op, const Expr &e) final { return MutateCmpOp(op, e); }
  Expr Mutate_(const NE *op, const Expr &e) final { return MutateCmpOp(op, e); }
  Expr Mutate_(const LT *op, const Expr &e) final { return MutateCmpOp(op, e); }
  Expr Mutate_(const LE *op, const Expr &e) final { return MutateCmpOp(op, e); }
  Expr Mutate_(const GT *op, const Expr &e) final { return MutateCmpOp(op, e); }
  Expr Mutate_(const GE *op, const Expr &e) final { return MutateCmpOp(op, e); }

  // do not rewrite cmp op that already in cond of select
  Expr Mutate_(const Select *op, const Expr &e) final {
    Expr t = this->Mutate(op->true_value);
    Expr f = this->Mutate(op->false_value);
    return Select::make(op->condition, t, f);
  }

  // do not rewrite cmp op that in cond of if
  Stmt Mutate_(const IfThenElse *op, const Stmt &s) final {
    Stmt then_case = this->Mutate(op->then_case);
    Stmt else_case;
    if (op->else_case.defined()) {
      else_case = this->Mutate(op->else_case);
    }
    return IfThenElse::make(op->condition, then_case, else_case);
  }
};

/* Removes FloorMod and FloorDiv where denominator matches loop bounds.
 * Example:
 *   for (i, 0, I) {
 *     a(floormod(i, I)) = b(floordiv(i, I))
 *   }
 * should be optimized to:
 *   for (i, 0, I) {
 *     a(i) = b(0)
 *   }
 * Future work: enhance it with more general algebraic simplification.
 */
class ModDivOpt : public IRMutator {
 private:
  Stmt Mutate_(const For *op, const Stmt &s) final {
    loop_bounds.emplace(op->loop_var.get(), Simplify(op->min + op->extent));
    Stmt stmt = IRMutator::Mutate_(op, s);
    loop_bounds.erase(op->loop_var.get());
    return stmt;
  }

  Expr Mutate_(const FloorDiv *op, const Expr &e) final {
    Expr a = Simplify(op->a);
    Expr b = Simplify(op->b);
    if (auto var = a.as<Variable>()) {
      if (loop_bounds.count(var)) {
        if (Equal(loop_bounds[var], b)) {
          return Expr(0);
        }
      }
    }
    return IRMutator::Mutate_(op, e);
  }

  Expr Mutate_(const FloorMod *op, const Expr &e) final {
    Expr a = Simplify(op->a);
    Expr b = Simplify(op->b);
    if (auto var = a.as<Variable>()) {
      if (loop_bounds.count(var)) {
        if (Equal(loop_bounds[var], b)) {
          return a;
        }
      }
    }
    return IRMutator::Mutate_(op, e);
  }

  std::unordered_map<const Variable *, Expr> loop_bounds;
};

Stmt MathIntrinRewrite(Stmt stmt) {
  stmt = IfThenElseRewriter().Mutate(stmt);
  stmt = CmpRewriter().Mutate(stmt);
  stmt = ModRewriter().Mutate(stmt);
  stmt = TruncRewriter().Mutate(stmt);
  stmt = NotRewriter().Mutate(stmt);
  stmt = RsqrtRewriter().Mutate(stmt);
  stmt = TanhRewriter().Mutate(stmt);
  stmt = DivRewriter().Mutate(stmt);
  stmt = RmCast().Mutate(stmt);
  stmt = RecRewriter().Mutate(stmt);
  stmt = MulZeroOpt().Mutate(stmt);
  stmt = PowRewriter().Mutate(stmt);
  stmt = ModDivOpt().Mutate(stmt);
  return stmt;
}
}  // namespace ir
}  // namespace akg
