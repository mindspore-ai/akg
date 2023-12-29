/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file simple_passes.cc
 * \brief Implementation of simple passes
 */

/*
 * 2019.12.30 - Define new functions of StmtUseVar.
 */

#include <tvm/ir.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <ir_pass.h>

namespace air {
namespace ir {

using std::map;
using std::string;

class IRSideEffect : public IRVisitor {
 public:
  void Visit(const NodeRef& e) final {
    if (has_side_effect_) return;
    IRVisitor::Visit(e);
  }

  void Visit_(const Call* op) final {
    if (op->call_type == Call::CallType::Intrinsic && op->name == intrinsic::tvm_tuple) {
      IRVisitor::Visit_(op);
    } else if (!op->is_pure()) {
      has_side_effect_ = true; return;
    } else {
      IRVisitor::Visit_(op);
    }
  }

  bool has_side_effect_{false};
};

bool HasSideEffect(const Expr& e) {
  IRSideEffect v;
  v.Visit(e);
  return v.has_side_effect_;
}

class IRSubstitue : public IRMutator {
 public:
  explicit IRSubstitue(
      const std::unordered_map<const Variable*, Expr>& smap)
      : smap_(smap) {
  }

  Expr Mutate_(const Variable* op, const Expr& e) final {
    auto it = smap_.find(op);
    if (it != smap_.end()) {
      return it->second;
    } else {
      return e;
    }
  }

 private:
  const std::unordered_map<const Variable*, Expr>& smap_;
};

Stmt Substitute(Stmt stmt,
                const std::unordered_map<const Variable*, Expr>& value_map) {
  if (value_map.size() == 0) return stmt;
  return IRSubstitue(value_map).Mutate(stmt);
}

Expr Substitute(Expr expr,
                const std::unordered_map<const Variable*, Expr>& value_map) {
  if (value_map.size() == 0) return expr;
  return IRSubstitue(value_map).Mutate(expr);
}

Stmt Substitute(Stmt stmt, const Map<Var, Expr>& value_map) {
  std::unordered_map<const Variable*, Expr> vmap;
  for (const auto& kv : value_map) {
    vmap[kv.first.get()] = kv.second;
  }
  return Substitute(stmt, vmap);
}

Expr Substitute(Expr expr, const Map<Var, Expr>& value_map) {
  std::unordered_map<const Variable*, Expr> vmap;
  for (const auto& kv : value_map) {
    vmap[kv.first.get()] = kv.second;
  }
  return Substitute(expr, vmap);
}

class SubstituteCCE : public IRMutator {
  /* We don't need a Scope to check if variable inside let statements has
     same name as the first argument because we use variable pointer to
     match. */
  const map<const Variable *, Expr> &replace;
  Expr find_replacement(const Variable *s) const {
    auto iter = replace.find(s);
    if (iter != replace.end()) {
      return iter->second;
    } else {
      return Expr();
    }
  }

 public:
  explicit SubstituteCCE(const map<const Variable *, Expr> &m) : replace(m) {}
  ~SubstituteCCE() override = default;

  Expr Mutate_(const Variable *v, const Expr &e) override {
    Expr expr;
    Expr r = find_replacement(v);
    if (r.defined()) {
      expr = r;
    } else {
      expr = e;
    }
    return expr;
  }

  Expr Mutate_(const Let *op, const Expr &e) override {
    Expr new_value = this->IRMutator::Mutate(op->value);
    Expr new_body = this->IRMutator::Mutate(op->body);
    Expr expr;
    if (new_value.same_as(op->value) && new_body.same_as(op->body)) {
      expr = e;
    } else {
      expr = Let::make(op->var, new_value, new_body);
    }
    return expr;
  }

  Stmt Mutate_(const LetStmt *op, const Stmt &s) override {
    Expr new_value = this->IRMutator::Mutate(op->value);
    Stmt new_body = this->IRMutator::Mutate(op->body);
    Stmt stmt;
    if (new_value.same_as(op->value) && new_body.same_as(op->body)) {
      stmt = s;
    } else {
      stmt = LetStmt::make(op->var, new_value, new_body);
    }
    return stmt;
  }

  Stmt Mutate_(const For *op, const Stmt &s) {
    Expr new_min = this->IRMutator::Mutate(op->min);
    Expr new_extent = this->IRMutator::Mutate(op->extent);
    Stmt new_body = this->IRMutator::Mutate(op->body);
    Stmt stmt;
    if (new_min.same_as(op->min) && new_extent.same_as(op->extent) && new_body.same_as(op->body)) {
      stmt = s;
    } else {
      stmt = For::make(op->loop_var, new_min, new_extent, op->for_type, op->device_api, new_body);
    }
    return stmt;
  }
};

Expr substitute(const Variable *var, const Expr replacement, const Expr expr) {
  map<const Variable *, Expr> m;
  m[var] = replacement;
  return SubstituteCCE(m).IRMutator::Mutate(expr);
}

Stmt substitute(const Variable *var, const Expr replacement, const Stmt stmt) {
  map<const Variable *, Expr> m;
  m[var] = replacement;
  SubstituteCCE s(m);
  return s.IRMutator::Mutate(stmt);
}

Expr substitute(const map<const Variable *, Expr> &m, const Expr expr) {
  SubstituteCCE s(m);
  return s.IRMutator::Mutate(expr);
}

Stmt substitute(const map<const Variable *, Expr> &m, const Stmt stmt) {
  SubstituteCCE s(m);
  return s.IRMutator::Mutate(stmt);
}

class SubstituteExpr : public IRMutator {
 public:
  Expr find, replacement;

  using IRMutator::Mutate;

  Expr Mutate(Expr e) final {
    if (Equal(e, find)) {
      return replacement;
    } else {
      return IRMutator::Mutate(e);
    }
  }
};

Expr substitute(const Expr find, const Expr replacement, const Expr expr) {
  SubstituteExpr s;
  s.find = find;
  s.replacement = replacement;
  return s.Mutate(expr);
}

Stmt substitute(const Expr find, const Expr replacement, const Stmt stmt) {
  SubstituteExpr s;
  s.find = find;
  s.replacement = replacement;
  return s.Mutate(stmt);
}

class VarTouchVisitor : public IRVisitor {
 public:
  void Visit(const NodeRef& e) final {
    if (use_var_) return;
    IRVisitor::Visit(e);
  }

  void Visit_(const Variable* op) final {
    Handle(op);
  }

  void Visit_(const Load* op) final {
    Handle(op->buffer_var.get());
    IRVisitor::Visit_(op);
  }

  virtual void Handle(const Variable* var) = 0;

  bool use_var_{false};
};

class ExprUseVarVisitor : public VarTouchVisitor {
 public:
  explicit ExprUseVarVisitor(const Variable* var)
      : var_(var) {}

  void Handle(const Variable* var) final {
    if (var == var_) use_var_ = true;
  }
 private:
  const Variable* var_;
};

class ExprUseVSetVisitor : public VarTouchVisitor {
 public:
  explicit ExprUseVSetVisitor(
      const std::unordered_set<const Variable*>& vset)
      : vset_(vset) {}

  void Handle(const Variable* var) final {
    if (vset_.count(var)) use_var_ = true;
  }
 private:
  const std::unordered_set<const Variable*>& vset_;
};

bool ExprUseVar(const Expr& e, const Var& v) {
  ExprUseVarVisitor visitor(v.get());
  visitor.Visit(e);
  return visitor.use_var_;
}

bool StmtUseVar(const Stmt& s, const Var& v) {
  ExprUseVarVisitor visitor(v.get());
  visitor.Visit(s);
  return visitor.use_var_;
}

bool StmtUseVar(const Stmt& s, const std::unordered_set<const Variable*>& vset) {
  ExprUseVSetVisitor visitor(vset);
  visitor.Visit(s);
  return visitor.use_var_;
}

bool ExprUseVar(const Expr& e,
                const std::unordered_set<const Variable*>& vset) {
  ExprUseVSetVisitor visitor(vset);
  visitor.Visit(e);
  return visitor.use_var_;
}

}  // namespace ir
}  // namespace air
