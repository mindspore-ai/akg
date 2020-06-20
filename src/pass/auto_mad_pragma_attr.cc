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
#include <tvm/ir_visitor.h>
#include <tvm/ir_pass.h>

#include "ir_pass.h"
#include "pass/ir_util.h"
#include "emit_insn/insn_info.h"

namespace akg {
namespace ir {
enum MadMNKTYPE { M = 0, N, K };
const char MADL0CSUFFIXSCHEDULE[] = ".local.L0C";
const char MADL0BSUFFIXSCHEDULE[] = ".local.L0B";
const char MADL0CSUFFIXPOLY[] = "_local_L0C";
const char MADL0BSUFFIXPOLY[] = "_local_L0B";
const char MAD_FATAL_LOG[] =
  "MAD check failed, please modify DSL to match correct cube compute pattern:\n"
  "L0C(i1, i2, i3, i4) += L0A(i2, i5, i3, i6) * L0B(i5, i1, i4, i6);\n"
  "Incorrect MAD code:\n";

Stmt emitGemmOutDtype(Stmt stmt) {
  class FindCast : public IRVisitor {
   public:
    void Visit_(const Cast *op) override { cast_p_ = op; }
    const Cast *cast_p_{nullptr};
  };
  auto f = FindCast();
  f.Visit(stmt);
  if (f.cast_p_) {
    Expr out_dtype;
    if (f.cast_p_->type == Float(32)) out_dtype = StringImm::make("float32");
    stmt = AttrStmt::make(make_zero(Int(32)), "pragma_gemm_out_dtype", out_dtype, stmt);
  }
  return stmt;
}

class GenMNKValue : public IRVisitor {
 public:
  GenMNKValue(MadMNKTYPE mnk, bool cubeSchedule)
      : updateMapValue_(false),
        cubeSchedule_(cubeSchedule),
        mnk_(mnk),
        oAxis_(Expr(1)),
        iAxis_(Expr(16)),
        oVAxis_(Expr(0)),
        iVAxis_(Expr(0)) {}
  ~GenMNKValue() override = default;

  void find(const Stmt &stmt) {
    // collect m,n,k index info
    updateMapValue_ = true;
    this->Visit(stmt);
    updateMapValue_ = false;
    // get m, n, k value
    this->Visit(stmt);
  }

  bool isMadCall(const std::string &callName, size_t callArgs) const {
    if (callName == "mad" && callArgs == 2) {
      return true;
    }
    return false;
  }

  void Visit_(const Call *op) final {
    CHECK(op);
    if (!updateMapValue_) return;
    if (isMadCall(op->name, op->args.size())) {
      IRVisitor::Visit_(op);
      return;
    }
    size_t pos = op->name.find(MADL0CSUFFIXPOLY);
    if (cubeSchedule_) pos = op->name.find(MADL0CSUFFIXSCHEDULE);
    if (op->args.size() < 4) return;
    if (pos != std::string::npos) {
      size_t len = op->args.size();
      maps_["no"] = GetVarsInExpr(op->args[len - 4]);
      maps_["mo"] = GetVarsInExpr(op->args[len - 3]);
      maps_["mi"] = GetVarsInExpr(op->args[len - 2]);
      maps_["ni"] = GetVarsInExpr(op->args[len - 1]);
      return;
    }
    pos = op->name.find(MADL0BSUFFIXPOLY);
    if (cubeSchedule_) pos = op->name.find(MADL0BSUFFIXSCHEDULE);
    if (pos != std::string::npos) {
      CHECK_GE(op->args.size(), 4);
      size_t len = op->args.size();
      /*
       * none transpose B ko, no, ni, ki
       * trans B no, ko, ki, ni
       * */
      if (maps_.find("ni") != maps_.end()) {
        if (isSame(maps_["ni"], op->args[len - 2])) {
          // none transpose
          maps_["ko"] = GetVarsInExpr(op->args[len - 4]);
          maps_["ki"] = GetVarsInExpr(op->args[len - 1]);
        } else {
          // transpose
          maps_["ko"] = GetVarsInExpr(op->args[len - 3]);
          maps_["ki"] = GetVarsInExpr(op->args[len - 2]);
        }
      }
    }
    IRVisitor::Visit_(op);
  }

  bool isSame(const Array<VarExpr> &leftArray, const Expr &right) {
    Array<VarExpr> rightArray = GetVarsInExpr(right);

    if (leftArray.size() != rightArray.size()) return false;

    std::unordered_map<std::string, int> leftMaps;
    for (auto left : leftArray) {
      if (leftMaps.count(left.get()->name_hint) == 0) {
        leftMaps[left.get()->name_hint] = 0;
      } else {
        // should not go to this statement
        leftMaps[left.get()->name_hint]++;
      }
    }

    for (auto right_item : rightArray) {
      if (leftMaps.count(right_item.get()->name_hint) == 0) return false;
    }

    return true;
  }

  bool inMNKMaps(const std::string &key, const std::string &name) {
    if (maps_.find(key) != maps_.end()) {
      for (auto var : maps_[key]) {
        if (var.get()->name_hint == name) return true;
      }
    }
    return false;
  }

  void Visit_(const For *op) final {
    CHECK(op->loop_var.as<Variable>());
    std::string name = op->loop_var.as<Variable>()->name_hint;
    switch (mnk_) {
      case MadMNKTYPE::M: {
        if (inMNKMaps("mo", name)) {
          oAxis_ = op->extent;
          oVAxis_ = op->loop_var;
          oName_ = op->loop_var.get()->name_hint;
        }
        if (inMNKMaps("mi", name)) {
          iAxis_ = op->extent;
          iVAxis_ = op->loop_var;
          iName_ = op->loop_var.get()->name_hint;
        }
        break;
      }
      case MadMNKTYPE::N: {
        if (inMNKMaps("no", name)) {
          oAxis_ = op->extent;
          oVAxis_ = op->loop_var;
          oName_ = op->loop_var.get()->name_hint;
        }
        if (inMNKMaps("ni", name)) {
          iAxis_ = op->extent;
          iVAxis_ = op->loop_var;
          iName_ = op->loop_var.get()->name_hint;
        }
        break;
      }
      case MadMNKTYPE::K: {
        if (inMNKMaps("ko", name)) {
          oAxis_ = op->extent;
          oVAxis_ = op->loop_var;
          oName_ = op->loop_var.get()->name_hint;
        }
        if (inMNKMaps("ki", name)) {
          iAxis_ = op->extent;
          iVAxis_ = op->loop_var;
          iName_ = op->loop_var.get()->name_hint;
        }
        break;
      }
      default: {
        CHECK(false);
      }
    }
    IRVisitor::Visit_(op);
  }

  Expr getOAxis() { return oAxis_; }

  Expr getIAxis() { return iAxis_; }

  Expr getOVAxis() { return oVAxis_; }

  Expr getIVAxis() { return iVAxis_; }

  std::string oName() { return oName_; }

  std::string iName() { return iName_; }

 private:
  bool updateMapValue_;
  bool cubeSchedule_;
  MadMNKTYPE mnk_;
  std::unordered_map<std::string, Array<VarExpr> > maps_;
  Expr oAxis_;
  Expr iAxis_;
  Expr oVAxis_;
  Expr iVAxis_;
  std::string oName_;
  std::string iName_;
};

class MadMNKGenerator : public IRMutator {
 public:
  MadMNKGenerator(const Expr &m, const Expr &n, const Expr &k, const std::string &mo, const std::string &mi,
                  const std::string &no, const std::string &ni, const std::string &ko, const std::string &ki,
                  const std::string &oko, const std::string &okoi)
      : m_value_(m),
        n_value_(n),
        k_value_(k),
        m_outer_(mo),
        m_inner_(mi),
        n_outer_(no),
        n_inner_(ni),
        k_outer_(ko),
        k_inner_(ki),
        o_k_outer_(oko),
        o_k_outer_inner_(okoi) {}
  ~MadMNKGenerator() override = default;
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    CHECK(op);
    if (op->attr_key == "pragma_emit_insn" && Equal(op->value, Expr("mad"))) {
      is_mad_ = true;
      static_cast<void>(IRMutator::Mutate_(op, s));
      if (mad_init_) {
        is_mad_init_block = false;
        mad_init_ = false;
        is_mad_ = false;
        return Evaluate::make(Expr(0));
      }

      Stmt stmt = IRMutator::Mutate_(op, s);
      if (stmt.as<AttrStmt>() != nullptr) {
        stmt = stmt.as<AttrStmt>()->body;
      }
      stmt = emitGemmOutDtype(stmt);
      stmt = AttrStmt::make(make_zero(Int(32)), "pragma_mad_n", n_value_, stmt);
      stmt = AttrStmt::make(make_zero(Int(32)), "pragma_mad_k", k_value_, stmt);
      stmt = AttrStmt::make(make_zero(Int(32)), "pragma_mad_m", m_value_, stmt);
      stmt = AttrStmt::make(make_zero(Int(32)), op->attr_key, op->value, stmt);
      is_mad_ = false;
      return stmt;
    }
    return IRMutator::Mutate_(op, s);
  }
  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    CHECK(op);
    if (is_mad_ && is_mad_init_block && op->value.as<FloatImm>() != nullptr) {
      mad_init_ = true;
      mad_init_prod_ = op;
    }
    return IRMutator::Mutate_(op, s);
  }
  Stmt Mutate_(const For *op, const Stmt &s) final {
    CHECK(op);
    if (op->loop_var.get()->name_hint.find(".init") != std::string::npos) {
      is_mad_init_block = true;
    }
    if (op->loop_var.get()->name_hint == (n_outer_ + ".init") ||
        op->loop_var.get()->name_hint == (m_outer_ + ".init") ||
        op->loop_var.get()->name_hint == (m_inner_ + ".init") ||
        op->loop_var.get()->name_hint == (n_inner_ + ".init")) {
      forMaps_[op->loop_var.get()->name_hint] = op->loop_var;
    }
    if (op->loop_var.get()->name_hint == n_outer_ || op->loop_var.get()->name_hint == m_outer_ ||
        op->loop_var.get()->name_hint == m_inner_ || op->loop_var.get()->name_hint == n_inner_) {
      forMaps_[op->loop_var.get()->name_hint] = op->loop_var;
      std::string keyName = op->loop_var.get()->name_hint + ".init";
      std::string valueName = op->loop_var.get()->name_hint;
      varMaps_[forMaps_[keyName].get()] = forMaps_[valueName];
    }
    if (op->loop_var.get()->name_hint == o_k_outer_) {
      forMaps_[o_k_outer_] = op->loop_var;
    }
    if (op->loop_var.get()->name_hint == o_k_outer_inner_) {
      forMaps_[o_k_outer_inner_] = op->loop_var;
    }

    auto res = IRMutator::Mutate_(op, s);
    if (valideMadInitPosition(op->loop_var->name_hint)) {
      if (mad_init_prod_ != nullptr) {
        std::vector<Stmt> stmts;
        Array<Expr> newArgs;
        for (auto arg : mad_init_prod_->args) {
          Expr sub = Substitute(arg, varMaps_);
          newArgs.push_back(sub);
        }
        Stmt init = Provide::make(mad_init_prod_->func, mad_init_prod_->value_index, mad_init_prod_->value, newArgs);
        if (o_k_outer_ != "") {
          if (forMaps_.count(o_k_outer_) >= 1) {
            if (o_k_outer_inner_ != "") {
              if (forMaps_.count(o_k_outer_inner_) >= 1)
                init = IfThenElse::make(
                  And::make(EQ::make(forMaps_[o_k_outer_inner_], Expr(0)), EQ::make(forMaps_[o_k_outer_], Expr(0))),
                  init);
            } else {
              init = IfThenElse::make(EQ::make(forMaps_[o_k_outer_], Expr(0)), init);
            }
          }
        }
        stmts.push_back(init);
        stmts.push_back(res);
        res = Block::make(stmts);
      }
    }
    return res;
  }

  bool valideMadInitPosition(const std::string &varName) {
    if (k_outer_ != "") {
      if (varName == k_outer_) return true;
    } else {
      if (varName == k_inner_) return true;
    }
    return false;
  }

 private:
  bool is_mad_init_block{false};
  bool is_mad_{false};
  bool mad_init_{false};
  const Provide *mad_init_prod_{nullptr};
  std::unordered_map<std::string, VarExpr> forMaps_;
  std::unordered_map<const Variable *, Expr> varMaps_;
  Expr m_value_;
  Expr n_value_;
  Expr k_value_;
  std::string m_outer_;
  std::string m_inner_;
  std::string n_outer_;
  std::string n_inner_;
  std::string k_outer_;
  std::string k_inner_;
  std::string o_k_outer_;
  std::string o_k_outer_inner_;
};

class MadMNKTYPEUpdater : public IRMutator {
 public:
  explicit MadMNKTYPEUpdater(bool cubeSchedule) { cubeSchedule_ = cubeSchedule; }
  ~MadMNKTYPEUpdater() override = default;
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == "pragma_emit_insn" && Equal(op->value, Expr("mad"))) {
      Expr mSize = reComputeMadMNK(MadMNKTYPE::M, op->body);
      Expr kSize = reComputeMadMNK(MadMNKTYPE::K, op->body);
      Expr nSize = reComputeMadMNK(MadMNKTYPE::N, op->body);
      Stmt stmt = op->body;
      stmt = emitGemmOutDtype(stmt);
      stmt = AttrStmt::make(make_zero(Int(32)), "pragma_mad_n", nSize, stmt);
      stmt = AttrStmt::make(make_zero(Int(32)), "pragma_mad_k", kSize, stmt);
      stmt = AttrStmt::make(make_zero(Int(32)), "pragma_mad_m", mSize, stmt);
      stmt = AttrStmt::make(make_zero(Int(32)), op->attr_key, op->value, stmt);
      return stmt;
    }
    return IRMutator::Mutate_(op, s);
  }
  Expr reComputeMadMNK(MadMNKTYPE mnk, const Stmt &s) {
    GenMNKValue f(mnk, cubeSchedule_);
    f.find(s);
    Expr res = f.getOAxis() * f.getIAxis();
    return res;
  }

 private:
  bool cubeSchedule_;
};

class GemmTransposeFuse : public IRMutator {
 public:
  GemmTransposeFuse() {
    pragmaTransposeData1_ = false;
    pragmaTransposeData2_ = false;
    pragmaTransposeWeight1_ = false;
    pragmaTransposeWeight2_ = false;
  }
  ~GemmTransposeFuse() override = default;
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    CHECK(op);
    if (op->attr_key == "pragma_load2d_transpose_weight" && op->value.as<IntImm>() != nullptr) {
      if (op->value.as<IntImm>()->value == 0) {
        offsetV_.clear();
        l1ProvideCall_ = Expr(0);
        pragmaTransposeWeight1_ = true;
        static_cast<void>(this->Mutate(op->body));
        pragmaTransposeWeight1_ = false;
        return Evaluate::make(0);
      } else if (op->value.as<IntImm>()->value == 1) {
        pragmaTransposeWeight2_ = true;
        Stmt stmt = this->Mutate(op->body);
        pragmaTransposeWeight2_ = false;
        return stmt;
      }
    } else if (op->attr_key == "pragma_load2d_transpose_data" && op->value.as<IntImm>() != nullptr) {
      if (op->value.as<IntImm>()->value == 0) {
        offsetV_.clear();
        l1ProvideCall_ = Expr(0);
        pragmaTransposeData1_ = true;
        static_cast<void>(this->Mutate(op->body));
        pragmaTransposeData1_ = false;
        return Evaluate::make(0);
      } else if (op->value.as<IntImm>()->value == 1) {
        pragmaTransposeData2_ = true;
        Stmt stmt = this->Mutate(op->body);
        pragmaTransposeData2_ = false;
        return stmt;
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    if (pragmaTransposeWeight1_ || pragmaTransposeData1_) {
      CHECK(s.as<Provide>());
      Array<Expr> left_args = s.as<Provide>()->args;
      Array<Expr> right_args;
      if (auto right = s.as<Provide>()->value.as<Call>()) {
        l1ProvideCall_ = s.as<Provide>()->value;
        if (right->call_type == Call::Halide) {
          right_args = right->args;
        }
      }
      for (unsigned int i = 0; i < left_args.size(); i++) {
        Expr offset = Simplify_cce(right_args[i] - left_args[i]);
        offsetV_.push_back(offset);
      }
    } else if (pragmaTransposeWeight2_ || pragmaTransposeData2_) {
      CHECK(s.as<Provide>());
      auto right_call = s.as<Provide>()->value.as<Call>();
      Array<Expr> callArgs;
      CHECK(right_call);
      CHECK(offsetV_.size() == right_call->args.size());
      for (size_t i = 0; i < right_call->args.size(); i++) {
        callArgs.push_back(right_call->args[i] + offsetV_[i]);
      }
      auto l1_right_call = l1ProvideCall_.as<Call>();
      CHECK(l1_right_call);
      auto new_call = Call::make(l1_right_call->type, l1_right_call->name, callArgs, Call::CallType::Halide,
                                 l1_right_call->func, l1_right_call->value_index);
      Stmt p = Provide::make(s.as<Provide>()->func, s.as<Provide>()->value_index, new_call, s.as<Provide>()->args);
      return p;
    }
    return s;
  }

 private:
  std::vector<Expr> offsetV_;
  Expr l1ProvideCall_;
  bool pragmaTransposeWeight1_;
  bool pragmaTransposeWeight2_;
  bool pragmaTransposeData1_;
  bool pragmaTransposeData2_;
};

Stmt updateMadMNKValue(const Stmt &stmt, bool manSchedule = false) {
  /************************************
   * This pass is for cube operator
   * 1. gemm m,n,k value fix
   * *********************************/
  return MadMNKTYPEUpdater(manSchedule).Mutate(stmt);
}

class MadAttrMNKExractor : public IRVisitor {
 public:
  explicit MadAttrMNKExractor(bool cubeSchedule) { cubeSchedule_ = cubeSchedule; }
  ~MadAttrMNKExractor() override = default;
  void Visit_(const AttrStmt *op) final {
    CHECK(op);
    if (op->attr_key == "pragma_emit_insn" && Equal(op->value, Expr("mad"))) {
      is_mad_ = true;
      IRVisitor::Visit_(op);
      if (mad_init_) {
        mad_init_ = false;
        is_mad_ = false;
        is_mad_init_block = false;
        return;
      }

      m_value_ = computeMadMNK(MadMNKTYPE::M, op->body, m_outer_, m_inner_);
      k_value_ = computeMadMNK(MadMNKTYPE::K, op->body, k_outer_, k_inner_);
      n_value_ = computeMadMNK(MadMNKTYPE::N, op->body, n_outer_, n_inner_);
      is_mad_ = false;
      return;
    } else if (op->attr_key == "pragma_is_reduce_k_outer") {
      auto forOp = (op->body).as<For>();
      if (forOp != nullptr) {
        if (o_k_outer_ == "")
          o_k_outer_ = forOp->loop_var->name_hint;
        else
          o_k_outer_inner_ = forOp->loop_var->name_hint;
      }
    }
    IRVisitor::Visit_(op);
  }

  Expr computeMadMNK(MadMNKTYPE mnk, const Stmt &s, std::string &oName, std::string &iName) {
    GenMNKValue f(mnk, cubeSchedule_);
    f.find(s);
    Expr res = f.getOAxis() * f.getIAxis();
    oName = f.oName();
    iName = f.iName();
    return res;
  }
  void Visit_(const Provide *op) final {
    if (is_mad_ && is_mad_init_block && op->value.as<FloatImm>() != nullptr) {
      mad_init_ = true;
    }
    return IRVisitor::Visit_(op);
  }

  void Visit_(const For *op) final {
    if (op->loop_var.get()->name_hint.find(".init") != std::string::npos) {
      is_mad_init_block = true;
    }
    return IRVisitor::Visit_(op);
  }

  bool mad_init_{false};
  bool is_mad_{false};
  bool is_mad_init_block{false};
  bool cubeSchedule_;
  Expr m_value_;
  Expr n_value_;
  Expr k_value_;
  std::string m_outer_{""};
  std::string m_inner_{""};
  std::string n_outer_{""};
  std::string n_inner_{""};
  std::string k_outer_{""};
  std::string k_inner_{""};
  std::string o_k_outer_{""};
  std::string o_k_outer_inner_{""};
};

class MadChecker : public IRVisitor {
 private:
  void Visit_(const AttrStmt *op) override {
    if (!in_mad_ && op->attr_key == "pragma_emit_insn" && op->value.as<StringImm>() &&
        op->value.as<StringImm>()->value == "mad") {
      loop_vars_.clear();
      L0C_var_ = "";
      L0C_args_.clear();
      L0B_args_.clear();
      L0A_args_.clear();
      in_mad_ = true;
      mad_node_ = op;
      Visit(op->body);
      in_mad_ = false;
      mad_node_ = nullptr;
    } else {
      Visit(op->body);
    }
  }

  void Visit_(const For *op) override {
    if (in_mad_) {
      loop_vars_.push_back(op->loop_var->name_hint);
      CHECK(loop_vars_.size() <= 7) << "too many nested loops";
      CHECK(op->min.as<IntImm>() && op->min.as<IntImm>()->value == 0) << "loop min must be 0";
      // comment for dynamic shape: what happen when op->extent.as<IntImm>() == nullptr ?
      Visit(op->body);
      loop_vars_.pop_back();
    } else {
      Visit(op->body);
    }
  }

  template <class T>
  std::vector<std::string> GetLastArgs(const T *op) {
    std::vector<std::string> provide_args;
    for (auto arg : op->args) {
      if (arg.template as<Variable>()) {
        provide_args.push_back(arg.template as<Variable>()->name_hint);
      } else if (is_const_int(arg, 0)) {
        // zero args are excluded from check
      } else {
        // arg is a complicated expr, we convert the expr to string for comparison
        std::stringstream expr_str;
        expr_str << arg;
        provide_args.push_back(expr_str.str());
      }
    }
    size_t last_args_size = std::min(provide_args.size(), (size_t)4);
    return std::vector<std::string>(provide_args.begin() + provide_args.size() - static_cast<uint64_t>(last_args_size),
                                    provide_args.end());
  }

  void Visit_(const Provide *op) override {
    if (in_mad_) {
      auto last_L0C_args = GetLastArgs<Provide>(op);
      // comment for dynamic shape: what happen when L0C_var_ == op->func->func_name() and L0C_args_ ==
      // last_L0C_args if !L0C_var_.empty() ?
      if (L0C_var_.empty()) {
        L0C_var_ = op->func->func_name();
        L0C_args_ = last_L0C_args;
      }
      Visit(op->value);
    }
  }

  void Visit_(const Call *op) override {
    if (op->call_type != Call::CallType::Halide) return;
    if (in_mad_) {
      auto last_call_args = GetLastArgs<Call>(op);
      if (last_call_args.size() != 4 || L0C_args_.size() != 4) {
        return;  // do not check
      }
      if (mad_node_ == nullptr) {
        return;
      }
      if (op->func->func_name() == L0C_var_) {  // test L0C
        CHECK(last_call_args == L0C_args_) << MAD_FATAL_LOG << mad_node_->body << "L0C args mismatch";
      } else if (last_call_args[0] == L0C_args_[1] && last_call_args[2] == L0C_args_[2]) {  // test L0A
        CHECK(L0A_args_.empty()) << MAD_FATAL_LOG << mad_node_->body << "duplicate L0A";
        L0A_args_ = last_call_args;
      } else if (last_call_args[1] == L0C_args_[0] && last_call_args[2] == L0C_args_[3]) {  // test L0B
        CHECK(L0B_args_.empty()) << MAD_FATAL_LOG << mad_node_->body << "duplicate L0B";
        L0B_args_ = last_call_args;
      } else {
        CHECK(false) << "Loop var ordering of " << MAD_FATAL_LOG << op->func->func_name()
                     << " does not match L0A, L0B or L0C.";
      }

      if (!L0A_args_.empty() && !L0B_args_.empty()) {
        CHECK(L0A_args_[1] == L0B_args_[0]) << MAD_FATAL_LOG << mad_node_->body << "i5 mismatch";
        CHECK(L0A_args_[3] == L0B_args_[3]) << MAD_FATAL_LOG << mad_node_->body << "i6 mismatch";
      }
    }
  }

 public:
  void run(const Stmt &stmt) {
    in_mad_ = false;
    Visit(stmt);
  }

 private:
  bool in_mad_{false};
  std::vector<std::string> loop_vars_;
  std::string L0C_var_;
  std::vector<std::string> L0C_args_;
  std::vector<std::string> L0B_args_;
  std::vector<std::string> L0A_args_;
  const AttrStmt *mad_node_{nullptr};
};

Stmt transposeGemm(const Stmt &stmt) {
  /************************************
   * This pass is for cube operator
   * 1. support left, right matrix transpose function
   *    remove pragma_load2d_transpose_weight = 0 IR
   *    move pragma_load2d_transpose_weight = 1 to dma_copy
   *    remove pragma_load2d_transpose_data = 0 IR
   *    move pragma_load2d_transpose_data = 1 to dma_copy
   * *********************************/
  return GemmTransposeFuse().Mutate(stmt);
}

Stmt GenerateMadAttr(Stmt stmt, bool manSchedule) {
  if (manSchedule) {
    MadAttrMNKExractor extractor(manSchedule);
    extractor.Visit(stmt);
    stmt = MadMNKGenerator(extractor.m_value_, extractor.n_value_, extractor.k_value_, extractor.m_outer_,
                           extractor.m_inner_, extractor.n_outer_, extractor.n_inner_, extractor.k_outer_,
                           extractor.k_inner_, extractor.o_k_outer_, extractor.o_k_outer_inner_)
             .Mutate(stmt);
  } else {
    stmt = updateMadMNKValue(stmt, manSchedule);
  }
  return stmt;
}

Stmt AutoMadPragmaAttr(Stmt stmt, bool manSchedule) {
  stmt = GenerateMadAttr(stmt, manSchedule);
  stmt = transposeGemm(stmt);
  MadChecker().run(stmt);
  return stmt;
}
}  // namespace ir
}  // namespace akg
