
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

#ifndef PASS_UTILS_H_
#define PASS_UTILS_H_
#include <dmlc/common.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <unordered_set>
#include <queue>
#include <arithmetic/pattern_match.h>
#include "tvm.h"

namespace akg {
namespace ir {
using air::ir::ExprUseVar;
using air::ir::substitute;

static const float HALF_MIN = 5.960464e-08;  // minimum number of float16
static const float HALF_MAX = 65504.0;       // maximum number of float16

struct  PairHash {
  template <typename T>
  size_t operator()(const std::pair<T, T> &a) const {
    return dmlc::HashCombine(std::hash<T>()(a.first), std::hash<T>()(a.second));
  }
};

using UnorderSet = std::unordered_set<Var, NodeHash, NodeEqual>;
bool IsCover(const Array<Expr> &big, const Array<Expr> &small);

Array<Array<Expr>> DetectNonLinearIndex(const Expr &e, const Array<Expr> &constVars = Array<Expr>());

Expr GetLinearCoefOfVar(const Expr &e, const Var &var);

Stmt RmEmptyEmitAttr(Stmt stmt);

Stmt TensorSubstitute(const Stmt &stmt, const FunctionRef &a, const FunctionRef &b, int b_value_index);

Stmt TensorStringSubstitute(const Stmt &stmt, const std::string &a, const FunctionRef &b, int b_value_index);

Expr TensorSubstitute(const Expr &e, const FunctionRef &a, const FunctionRef &b, int b_value_index);

Expr TensorStringSubstitute(const Expr &e, const std::string &a, const FunctionRef &b, int b_value_index);

Stmt SubstituteLoopVar(Stmt &s, const Variable *old_var, const Expr &new_var);

Range InferSimpleExprRange(Expr e, std::unordered_map<const Variable *, Range> *rmap);

bool IsVarInExpr(const Expr &needle, const Expr &haystack);

bool IsVarsInExpr(const std::vector<Var> &vars, const Expr &haystack);

bool IsFlexVarInIf(const Expr &var, const Array<Stmt> &expr);

class Bound {
 public:
  Expr min;
  Expr max;

  static Bound make(const Range range) {
    Bound bound;
    bound.min = range->min;
    bound.max = Simplify(range->min + range->extent - 1);
    return bound;
  }

  static Bound make(const Expr min, const Expr max) {
    Bound bound;
    bound.min = min;
    bound.max = max;
    return bound;
  }

  bool defined() { return this->min.defined() && this->max.defined(); }
};

enum class Interval { LTZERO = -2, LEZERO, ZERO, GEZERO, GTZERO, UNKNOWN };

enum class Interval_1 { LTZERO = -1, UNK, GEZERO, GEONE };

enum class Sign { NEG = -1, ZERO, POS, UNK };

int64_t Log2(uint64_t value);

bool IsZero(const Expr &e);

int64_t GetIntConst(const Expr &expr);

double GetFloatConst(const Expr &expr);

int GetInt32Const(const Expr &expr);

uint64_t GetUIntConst(const Expr &expr);

std::ostream &operator<<(std::ostream &os, const Bound &bound);

Bound InferBoundOfExpr(const Expr &expr, const std::unordered_map<const Variable *, Range> &var_bound_map);

Bound InferBoundOfExprWithCond(const Expr &expr, const Array<Expr> &constraints);
Bound InferBoundOfExprWithCond(const Expr &expr, const Array<Expr> &var_cst, const Array<Expr> &constraints,
                               const std::unordered_set<Var, NodeHash, NodeEqual> &vars_set);

Bound InferVarBound(const Expr &expr, const Array<Expr> &constraints);

Array<Expr> GetSortedConstraint(const Array<Expr> &constraints,
                                const std::unordered_set<Var, NodeHash, NodeEqual> &vars_set);

Range InferBound(const Expr &expr, const Array<Expr> &constraints);

static inline bool is_const_true(const Expr &expr) { return is_positive_const(expr); }

static inline bool is_const_false(const Expr &expr) { return is_const_int(expr, 0); }

Expr SimplifyConditionExpr(const Expr &expr, const std::unordered_map<const Variable *, Range> &var_bound_map);

Expr SimplifyExpr(const Expr &expr, const std::unordered_map<const Variable *, Range> &var_bound_map);

bool can_prove(const Expr &expr, const std::unordered_map<const Variable *, Range> &var_bound_map);

bool ExprPatternMatch(const Expr &expr, const Expr &pattern, std::vector<Expr> *matches = nullptr);

std::vector<Expr> ExtractSubExprs(const Expr &e);

std::string ExprToString(const Expr &expr);
std::string ExprToVarName(const Expr &expr);

inline bool isImm(const Expr &val) {
  return (val.as<FloatImm>()) || (val.as<IntImm>()) || (val.as<UIntImm>() || (val.as<StringImm>()));
}

Array<Expr> GetMinCondsSet(const Array<Expr> &constraints,
                           const std::unordered_set<Var, NodeHash, NodeEqual> &vars_set);

static inline bool IsConstExpr(const Expr &expr) {
  Expr simplified = Simplify(expr);
  return isImm(simplified);
}

bool IsAffineExprOfVars(const Expr &expr, const std::unordered_set<const Variable *> &vars);

Expr GetConstIntUpBound(const Expr &e);

Expr GetConstIntLowBound(const Expr &e);

int GetRangeWithParam(const Expr &expr);

int GetSign(const Expr &expr);

class DataDepender : public IRVisitor {
 public:
  DataDepender() {}
  ~DataDepender() override = default;

  bool DependWith(const DataDepender &other) {
    for (auto def : def_) {
      if (other.use_.count(def) || other.def_.count(def)) return true;
    }
    for (auto use : use_) {
      if (other.def_.count(use)) return true;
    }
    return false;
  }

  void Visit_(const Variable *op) override {
    use_.insert(op);
    IRVisitor::Visit_(op);
  }

  void Visit_(const Load *op) override {
    use_.insert(op->buffer_var.get());
    IRVisitor::Visit_(op);
  }

  void Visit_(const Store *op) override {
    def_.insert(op->buffer_var.get());
    IRVisitor::Visit_(op);
  }

  void Visit_(const Call *op) override {
    if (op->is_intrinsic(air::ir::intrinsic::tvm_access_ptr)) {
      const auto buf = op->args[1].as<Variable>();
      const auto rw = op->args[4].as<IntImm>();
      CHECK(buf != nullptr && rw != nullptr);
      if (static_cast<unsigned int>(rw->value) & 2) {
        def_.insert(buf);
      } else {
        use_.insert(buf);
      }
      Visit(op->args[1]);  // offset
      Visit(op->args[2]);  // extent
      return;
    }
    IRVisitor::Visit_(op);
  }

  std::unordered_set<const Variable *> def_;
  std::unordered_set<const Variable *> use_;
};

class SimplifyIfCondClass {
  std::pair<Expr, Bound> cond_bound;

 public:
  void GetCondBound(const EQ *op);
  void GetCondBound(const NE *op);
  void GetCondBound(const LT *op);
  void GetCondBound(const LE *op);
  void GetCondBound(const GE *op);
  void GetCondBound(const GT *op);

  bool CanProveValid(const Expr &cond, const Array<Expr> &constraints);
};

class RecoverFor : public IRMutator {
  Stmt Mutate_(const For *op, const Stmt &s) override {
    Stmt stmt = IRMutator::Mutate_(op, s);
    if (op->for_type == ForType::Vectorized) {
      const For *n = stmt.as<For>();
      CHECK(n);
      return For::make(n->loop_var, n->min, n->extent, ForType::Serial, n->device_api, n->body);
    }
    return stmt;
  }
};

class CondGraph {
  int vertices;
  std::list<int> *adj;
  std::queue<int> zero_set;
  std::vector<int> indegree;

 public:
  std::vector<int> sort_res;
  std::vector<std::tuple<int, int, Expr>> var_constraint;
  explicit CondGraph(int vertices);
  ~CondGraph();
  void AddEdge(int v, int w);
  bool TopoSort();
  void TopoSortConstraintByVar(const Array<Expr> &constraints, const UnorderSet &vars_set);
  void TopoSortConstraint(const Array<Expr> &constraints, const UnorderSet &vars_set);
  void AddEdgeByDetectOp(const int index, const Expr &expr);
  void AddEdgeInExpr(const int index, const Expr &expr);
};

class VectorizeFor : public IRMutator {
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) override;
  Stmt Mutate_(const Evaluate *op, const Stmt &s) override;
  Stmt Mutate_(const Provide *op, const Stmt &s) override;
  Stmt Mutate_(const Store *op, const Stmt &s) override;
  Stmt Mutate_(const For *op, const Stmt &s) override;
  Expr Mutate_(const Variable *op, const Expr &e) override;

 private:
  std::unordered_map<const Variable *, std::unordered_set<const Node *>> var_in_provide_store;
  bool in_provide_store{false};
  const Node *cur_provide_store{nullptr};
  int provide_store{0};
  bool in_pragma_{false};
};

// this adds check of FloatImm to expr_operator.h::is_const
inline bool is_constant(const Expr &x) {
  if (x.as<IntImm>() || x.as<UIntImm>()) {
    return true;
  } else if (x.as<FloatImm>()) {
    return true;
  } else if (const auto *op = x.as<Broadcast>()) {
    const Expr &val = op->value;
    if (val.as<IntImm>() || val.as<UIntImm>()) {
      return true;
    }
  }
  return false;
}

inline bool isZero(const Expr &val) {
  if (const auto fi = val.as<FloatImm>()) {
    if (fi->value == 0.0) return true;
  } else if (const auto ii = val.as<IntImm>()) {
    if (ii->value == 0) return true;
  } else if (const auto ui = val.as<UIntImm>()) {
    if (ui->value == 0) return true;
  }
  return false;
}

inline void GatherVars(const Expr expr, std::unordered_set<Var, air::NodeHash, air::NodeEqual> *vset) {
  PostOrderVisit(expr, [&vset](const NodeRef &node) {
    if (node.as<Variable>()) {
      vset->insert(Downcast<Var>(node));
    }
  });
}

inline void GatherVars(const Expr expr, std::vector<Var> *vec) {
  int pos = 0;
  PostOrderVisit(expr, [&vec, &pos](const NodeRef &node) {
    if (node.as<Variable>()) {
      auto tmpVar = Downcast<Var>(node);
      bool hasVar = false;
      for (const auto &value : *vec) {
        if (Equal(value, tmpVar)) {
          hasVar = true;
          break;
        }
      }
      if (!hasVar) {
        vec->insert(vec->begin() + pos, tmpVar);
        ++pos;
      }
    }
  });
}

inline int CountVars(const Expr &v) {
  std::unordered_set<Var, air::NodeHash, air::NodeEqual> vars;
  GatherVars(v, &vars);
  return static_cast<int>(vars.size());
}

inline int CountVars(const Array<Expr> &args) {
  std::unordered_set<Var, air::NodeHash, air::NodeEqual> vars;
  for (size_t i = 0; i < args.size(); ++i) {
    GatherVars(args[i], &vars);
  }
  return static_cast<int>(vars.size());
}

// may have repeat vars
inline int AllVars(const Array<Expr> &args) {
  std::unordered_set<Var, air::NodeHash, air::NodeEqual> vars;
  int num = 0;
  for (size_t i = 0; i < args.size(); ++i) {
    vars.clear();
    GatherVars(args[i], &vars);
    num += static_cast<int>(vars.size());
  }
  return num;
}

template <typename ObjType>
inline ObjectPtr<Object> GetObjPtr(const ObjType *ptr) {
  return air::runtime::GetObjectPtr<Object>(const_cast<ObjType *>(ptr));
}

template <class T>
bool LimitCheck(const air::arith::PVar<T> &n1, const air::arith::PVar<T> &n2);

class AttrIRMutator : public IRMutator {
 public:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) {
    Expr value = Mutate(op->value);
    Stmt body = Mutate(op->body);

    if (op->node->IsInstance<Map<std::string, NodeRef>::ContainerType>()) {
      auto attrs = Downcast<Map<std::string, NodeRef>>(op->value);
      std::unordered_map<std::string, NodeRef> new_attrs;
      for (auto kv : attrs) {
        auto new_node = kv.second;
        if (kv.second->IsInstance<Expr::ContainerType>()) {
          new_node = Mutate(air::Downcast<Expr>((kv.second)));
        } else if (kv.second->IsInstance<Range::ContainerType>()) {
          auto old = air::Downcast<Range>(kv.second);
          new_node = Range::make_by_min_extent(Mutate(old->min), Mutate(old->extent));
        }
        new_attrs.emplace(std::make_pair(kv.first, new_node));
      }
      return AttrStmt::make(Map<std::string, NodeRef>(new_attrs.begin(), new_attrs.end()), op->attr_key, value, body);
    }

    if (value.same_as(op->value) && body.same_as(op->body)) {
      return s;
    } else {
      return AttrStmt::make(op->node, op->attr_key, value, body);
    }
  }
};

Array<Expr> GetBinaryOpExprChildren(const Expr &e);

Expr GetBinaryOpName(const Expr &e);

Array<VarExpr> GetVarsInExpr(const Expr &expr, bool exclude_upper_case_vars = false);

/// Get index of item in array
/// \tparam T
/// \param array
/// \param elem
/// \param index
/// \return
template <typename T>
bool GetIndexOfElement(const Array<T> &array, const T &elem, size_t &index) {
  for (size_t i = 0; i < array.size(); ++i) {
    const auto item = array[i];
    if (Equal(elem, item)) {
      index = i;
      return true;
    }
  }

  return false;
}

std::string GetProductName();

bool IsHalideCall(const Expr &e);

bool ContainsHalideCall(const Array<Expr> args);

std::string GetOpReduceType(const Expr value);

extern const char *const LOCAL_BUF;
extern const char *const LOCAL_C1;
extern const char *const LOCAL_C0B;
extern const char *const LOCAL_C0C;
extern const char *const LOCAL_C1_LOCAL_C0A;
extern const char *const LOCAL_C1_LOCAL_C0B;
extern const char *const LOCAL_BUF_LOCAL_C0C;
extern const char *const FRACTAL_C1;
extern const char *const FRACTAL_C1_LOCAL_C0B;

constexpr auto AKG_REDUCE_SUM = "SumOp";
constexpr auto AKG_REDUCE_MIN = "MinOp";
constexpr auto AKG_REDUCE_MAX = "MaxOp";
constexpr auto AKG_REDUCE_AND = "AndOp";
constexpr auto AKG_REDUCE_OR = "OrOp";
constexpr auto AKG_REDUCE_PROD = "ProdOp";
constexpr auto AKG_REDUCE_UNSUPPORTED = "X";

constexpr auto AKG_TENSOR_NOT_PROMOTE = "TENSOR_NOT_PROMOTE";
constexpr auto AKG_INNER_TENSOR = "INNER_TENSOR";
constexpr auto AKG_TENSOR_OF_TENSOR = "TENSOR_OF_TENSOR";
constexpr auto AKG_ATOMIC_TOT = "atomic_tot";
constexpr auto AKG_REMOVE_SELF_DEPENDENCE = "REMOVE_SELF_DEPENDENCE";
constexpr auto CSR_AVG_ROW = "csr_avg_row";
constexpr auto CSR_MAP_THREAD = "csr_map_thread";

static constexpr auto ATTR_PREFETCH_MODE = "prefetch_mode";
enum class PrefetchMode {
  DEFAULT = 0, TRANSFERBUFFER, DOUBLEBUFFER, TRANSFERBUFFER_THREADGROUP, DOUBLEBUFFER_THREADGROUP
};

constexpr auto REDUCE_LIB_TYPE_ORIGIN = "origin";
constexpr auto REDUCE_LIB_TYPE_PARIS = "paris";
constexpr auto AKG_REDUCE_LIB_SPACE = "akg_reduce";
constexpr auto AKG_REDUCE_LIB_NAME = "AkgReduce";
constexpr auto AKG_KAHAN_LIB_NAME = "AkgKahanAccumulation";
constexpr auto PARIS_REDUCE_LIB_SPACE = "paris_reduce";
constexpr auto PARIS_REDUCE_LIB_NAME = "ParisReduce";
constexpr auto AKG_REDUCE_RETURN_NAME = "AkgAtomicReturn";
constexpr auto PARIS_REDUCE_RETURN_NAME = "ParisReturn";
constexpr auto BINARY_OPERATOR_LIFT_OPERAND_NAME = "lhs";
constexpr auto BINARY_OPERATOR_RIGHT_OPERAND_NAME = "rhs";

struct AtomicReturnData {
  std::string reduce_op;
  std::string akg_atomic_api;
  std::string akg_atomic_template_arg;
  Type output_tensor_data_type_info;
  Expr atomic_rhs;
  Stmt gm_write_stmt;
};

void ConstructAtomicReturnFuncName(const std::string &reduce_lib, const std::string &reduce_op,
                                   std::string &akg_atomic_api, std::string &akg_atomic_template_arg);
Stmt MakeAtomicStmt(const AtomicReturnData &atomic_data);
}  // namespace ir

std::string GetBufScope(const std::string &name);
bool AttrExists(air::Schedule sch, std::string attr_name);

constexpr auto PURE_INTRINSIC_WITH = "with";
constexpr auto PURE_INTRINSIC_ORIG = "orig";
}  // namespace akg

#endif  // PASS_UTILS_H_
