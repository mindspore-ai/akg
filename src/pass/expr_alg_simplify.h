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

#ifndef PASS_EXPR_COMPUTATION_H_
#define PASS_EXPR_COMPUTATION_H_

#include <tvm/ir.h>
#include <tvm/ir_functor_ext.h>
#include <tvm.h>
#include <src/pass/ir_util.h>
#include <src/arithmetic/pattern_match.h>

namespace akg {
namespace ir {
using std::pair;
using std::string;
using std::to_string;
using std::unordered_map;
using std::vector;

class ExprSimplifier : public air::ir::IRMutator {
  struct LoadInfo {
    Var buffer_var;
    Expr index;
    Expr predicate;
    air::DataType type;
  };

  struct CallInfo {
    air::DataType type;
    std::string name;
    Array<Expr> args;
    Call::CallType call_type;
    FunctionRef func;
    int value_index;
  };

  class DivMutator : public air::ir::IRMutator {
   private:
    Expr Mutate_(const Mul *op, const Expr &e) final;
  };

 public:
  Expr Simplify(const air::Expr &e);

  Expr Simplify(const air::Expr &e, const vector<Expr> &conds);

  Expr ReduceInequality(const air::Expr &e, const Var &reduceVar, bool scale = false, bool get_larger = false);

  bool Equals(const air::Expr &e1, const air::Expr &e2);

  bool CanProveWithParam(const Expr &e);

  bool CanProveWithPosParam(const Expr &e);

  bool IsDivisible(const air::Expr &e, const air::Expr &divisor);

  Expr Gcd(const air::Expr &e1, const air::Expr &e2);

  Expr Retrieval(const Expr &e);

  Array<Expr> GetPolynomial(const Expr &e1, const Expr &e2);

  Expr RetroConstToMin(const Expr &e);

  Expr HighDegIneqlSolver(const Expr &e, const Var &tar_var, bool get_larger);

  vector<int64_t> BisectSolver(Expr &e, Var &var, int64_t inf, int64_t sup);

 private:
  Expr Mutate_(const Min *op, const Expr &e) final;

  Expr Mutate_(const Max *op, const Expr &e) final;

  Expr Mutate_(const Mod *op, const Expr &e) final;

  Expr Mutate_(const FloorMod *op, const Expr &e) final;

  Expr Mutate_(const Cast *op, const Expr &e) final;

  Expr Mutate_(const FloorDiv *op, const Expr &e) final;

  Expr Mutate_(const Div *op, const Expr &e) final;

  Expr Mutate_(const Select *op, const Expr &e) final;

  template <class T>
  Expr BinaryMutate(const T *op, const Expr &e);

  Expr Mutate_(const GE *op, const Expr &e) final;

  Expr Mutate_(const GT *op, const Expr &e) final;

  Expr Mutate_(const LT *op, const Expr &e) final;

  Expr Mutate_(const LE *op, const Expr &e) final;

  Expr Mutate_(const EQ *op, const Expr &e) final;

  Expr Mutate_(const NE *op, const Expr &e) final;

  template <class T>
  Expr BinaryBoolMutate(const T *op, const Expr &e);

  Expr Mutate_(const And *op, const Expr &e) final;

  Expr Mutate_(const Or *op, const Expr &e) final;

  Expr Mutate_(const Not *op, const Expr &e) final;

  Expr Mutate_(const Load *op, const Expr &e) final;

  Expr Mutate_(const Call *op, const Expr &e) final;

  Expr Mutate_(const StringImm *op, const Expr &e) final;

  Expr Mutate_(const Variable *op, const Expr &e) final;

  vector<Expr> GatherRetroTerm(const Expr &e);

  Expr SubstituteDiv(const Expr &e, const Expr &substitute);

  Expr ExtraDivVar(const Expr &expr, const Var &reduce_var);

  Expr SimplifyWithInfo(const air::Expr &e, const vector<Expr> &conds) const;

  int VisitDivWithLcm(const Expr &e) const;

  Expr ReduceIneqlWithScale(const Expr &e, const Var &reduce_var, bool is_less, bool get_larger);

 private:
  unordered_map<const Variable *, const air::DataType> min_map_;
  unordered_map<const Variable *, const air::DataType> max_map_;
  unordered_map<const Variable *, const air::DataType> mod_map_;
  unordered_map<const Variable *, const air::DataType> cast_map_;
  unordered_map<const Variable *, const air::DataType> string_map_;
  unordered_map<const Variable *, const air::DataType> floordiv_map_;
  unordered_map<const Variable *, const air::DataType> div_map_;
  unordered_map<const Variable *, const air::DataType> select_map_;
  unordered_map<const Variable *, const air::DataType> and_map_;
  unordered_map<const Variable *, const air::DataType> or_map_;
  unordered_map<const Variable *, const air::DataType> not_map_;
  unordered_map<const Variable *, const air::DataType> load_map_;
  unordered_map<const Variable *, const air::DataType> call_map_;
  unordered_map<Var, vector<Expr>, air::NodeHash, air::NodeEqual> min_child_;
  unordered_map<Var, vector<Expr>, air::NodeHash, air::NodeEqual> max_child_;
  unordered_map<Var, vector<Expr>, air::NodeHash, air::NodeEqual> mod_child_;
  unordered_map<Var, vector<Expr>, air::NodeHash, air::NodeEqual> floordiv_child_;
  unordered_map<Var, vector<Expr>, air::NodeHash, air::NodeEqual> div_child_;
  unordered_map<Var, vector<Expr>, air::NodeHash, air::NodeEqual> select_child_;
  unordered_map<Var, vector<Expr>, air::NodeHash, air::NodeEqual> and_child_;
  unordered_map<Var, vector<Expr>, air::NodeHash, air::NodeEqual> or_child_;
  unordered_map<Var, vector<Expr>, air::NodeHash, air::NodeEqual> not_child_;
  unordered_map<Var, Expr, air::NodeHash, air::NodeEqual> cast_child_;
  unordered_map<Var, std::string, air::NodeHash, air::NodeEqual> string_child_;
  unordered_map<Var, LoadInfo, air::NodeHash, air::NodeEqual> load_child_;
  unordered_map<Var, CallInfo, air::NodeHash, air::NodeEqual> call_child_;

  unordered_map<Var, vector<Expr>, air::NodeHash, air::NodeEqual> div_scale_range_;

  Array<VarExpr> old_vars_;
  vector<Expr> info_;
  vector<Array<Expr>> cache_min_;
  vector<Array<Expr>> cache_max_;
  vector<Var> var_with_reduce_;
  vector<Var> divvar_with_reduce_;
  vector<pair<Var, Var>> div_mod_pair_;
  int min_op_count_ = 0;
  int max_op_count_ = 0;
  int mod_op_count_ = 0;
  int cast_op_count_ = 0;
  int floor_div_op_count_ = 0;
  int string_op_count_ = 0;
  int div_op_count_ = 0;
  int select_op_count_ = 0;
  int and_op_count_ = 0;
  int or_op_count_ = 0;
  int not_op_count_ = 0;
  int load_op_count_ = 0;
  int call_op_count_ = 0;
  bool is_retrieval_ = false;
  bool is_scale_{false};
  air::DataType highest_cast_type_{Int(32)};

  Var reduce_var_;
  bool is_less_than_{false};
};

void TestExprCompuationSimplify();
}  // namespace ir
}  // namespace akg

#endif  // PASS_EXPR_COMPUTATION_H_
