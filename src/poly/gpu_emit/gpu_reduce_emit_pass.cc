/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

/*!
 * \file gpu_reduce_emit_pass.cc
 */

#include "emit_pass.h"
#include "gpu_isl_emitter_reduce.h"

namespace akg {
namespace ir {
namespace poly {

struct ReduceData {
  Expr init_value;
  const Provide *origin_reduce_stmt_;

  std::string reduce_stmt_index_;
  std::string scalar_tensor_name_;
  std::string scalar_kht_name_;
  std::string scalar_khy_name_;
  std::string scalar_khc_name_;
  Expr input_tensor_expr_;
  std::string shared_compute_name_;
  std::string reduce_op_;
  std::string promoted_tensor_name_for_reduce_;
  std::string akg_reduce_api_;
  std::string akg_reduce_template_arg_;
  Type reduce_data_type_info_;
  std::map<std::string, Tensor> scalar_tensor_;
  Tensor shared_tensor_;
  std::vector<Stmt> stmts_;
};

class ReduceInfoCollect : public IRVisitor {
 public:
  explicit ReduceInfoCollect(ScopInfo &scop_info) : scop_info_(scop_info) {}
  using IRVisitor::Visit_;

  void Visit_(const Call *op) final {
    std::string name = op->name;
    if (ScopInfo::IsReduceInit(name)) {
      reduce_valid_ = true;
      in_reduce_area_ = true;
      ReduceData reduce_data;
      std::vector<std::string> strs = common::Split(name, "_");
      CHECK_EQ(strs.size(), REDUCE_FLAG_SIZE) << "red update format is not right!.";

      reduce_data.reduce_stmt_index_ = strs[REDUCE_FLAG_REDUCE_INDEX];
      reduce_data.scalar_tensor_name_ = SCALAR_TENSOR_PREFIX;
      reduce_data.scalar_tensor_name_ += reduce_data.reduce_stmt_index_;

      reduce_data.shared_compute_name_ = SHARED_TENSOR_PREFIX;
      reduce_data.shared_compute_name_ += reduce_data.reduce_stmt_index_;

      if (AkgSupportedReduceOp.count(strs[REDUCE_FLAG_TYPE_POS])) {
        reduce_data.reduce_op_ = AKG_REDUCE_LIB_SPACE;
        reduce_data.reduce_op_ += "::";
        reduce_data.reduce_op_ += strs[REDUCE_FLAG_TYPE_POS];
      }
      CHECK(!reduce_data.reduce_op_.empty()) << "reduce op should not be empty!";
      if (reduce_data.reduce_op_.find("SumOp") != std::string::npos) {
        reduce_data.scalar_kht_name_ = SCALAR_KHT_PREFIX;
        reduce_data.scalar_kht_name_ += reduce_data.reduce_stmt_index_;
        reduce_data.scalar_khy_name_ = SCALAR_KHY_PREFIX;
        reduce_data.scalar_khy_name_ += reduce_data.reduce_stmt_index_;
        reduce_data.scalar_khc_name_ = SCALAR_KHC_PREFIX;
        reduce_data.scalar_khc_name_ += reduce_data.reduce_stmt_index_;
      }
      cur_reduce_stmt_ = strs[REDUCE_FLAG_STMT_PREFIX_POS] + "_" + strs[REDUCE_FLAG_STMT_NUM_POS];

      std::string origin_tensor_name = "";
      for (auto it : scop_info_.analysis_result_.GetReduceTensorInfoMap()) {
        if (it.first.name() == cur_reduce_stmt_) {
          origin_tensor_name = it.second.write_tensor_name;
          reduce_data.reduce_data_type_info_ = it.second.write_dtype;
          break;
        }
      }
      CHECK(!origin_tensor_name.empty()) << "origin_tensor_name should not be empty!";

      for (const auto &buffer : scop_info_.analysis_result_.active_buffer_footprints_) {
        auto cluster_id = buffer.second.cluster_id;
        auto buf_def = scop_info_.analysis_result_.GetBufferDefInfo(cluster_id);
        if (buf_def.tensor_id.name() == origin_tensor_name) {
          reduce_data.promoted_tensor_name_for_reduce_ = cluster_id.name();
          break;
        }
      }

      for (auto it : scop_info_.analysis_result_.GetReduceTensorInfoMap()) {
        if (it.first.name() == cur_reduce_stmt_) {
          reduce_data.init_value = it.second.init_value;
          break;
        }
      }

      MakeAkgReduceFuncName(reduce_data);
      SetScalarTensorBind(reduce_data, reduce_data.scalar_tensor_name_);
      if (reduce_data.reduce_op_.find("SumOp") != std::string::npos) {
        SetScalarTensorBind(reduce_data, reduce_data.scalar_kht_name_);
        SetScalarTensorBind(reduce_data, reduce_data.scalar_khy_name_);
        SetScalarTensorBind(reduce_data, reduce_data.scalar_khc_name_);
      }
      SetSharedTensorBind(reduce_data);

      reduce_datas_[cur_reduce_stmt_] = reduce_data;
    } else if (ScopInfo::IsReduceUpdate(name)) {
      in_reduce_area_ = false;
    }
    IRVisitor::Visit_(op);
  }

  void Visit_(const Provide *op) {
    if (in_reduce_area_) {
      reduce_datas_[cur_reduce_stmt_].origin_reduce_stmt_ = op;
    }
    IRVisitor::Visit_(op);
  }

  void MakeAkgReduceFuncName(ReduceData &reduce_data) {
    auto thread_cfg = scop_info_.user_config_.GetThreadConfig();
    CHECK(thread_cfg) << "thread config is null.";
    auto block_cfg = scop_info_.user_config_.GetBlockConfig();
    CHECK(block_cfg) << "thread config is null.";
    int tx = thread_cfg->GetX().second;
    int ty = thread_cfg->GetY().second;
    int by = block_cfg->GetY().second;
    std::string direction = scop_info_.analysis_result_.GetReduceDirection();
    CHECK(!direction.empty()) << "direction should not be empty!";
    std::string direction_size = "";
    if (direction == X_DIRECTION) {
      direction_size = std::to_string(tx);
    } else {
      direction_size = std::to_string(ty);
    }

    std::string reduce_lib_namespace = "";
    std::string reduce_lib_name = "";
    if (scop_info_.user_config_.GetReduceLibType() == REDUCE_LIB_TYPE_ORIGIN) {
      reduce_lib_namespace = AKG_REDUCE_LIB_SPACE;
      reduce_lib_name = AKG_REDUCE_LIB_NAME;
    } else if (scop_info_.user_config_.GetReduceLibType() == REDUCE_LIB_TYPE_PARIS) {
      reduce_lib_namespace = PARIS_REDUCE_LIB_SPACE;
      reduce_lib_name = PARIS_REDUCE_LIB_NAME;
    } else {
      CHECK(false) << "reduce lib type is invalid!"
                   << "\n";
    }
    std::string ret = reduce_lib_namespace;
    ret += "::";
    ret += reduce_lib_name;

    reduce_data.akg_reduce_api_ = ret;
    ret = "";

    std::string op = reduce_data.reduce_op_;
    ret += op;
    ret += ", ";

    ret += std::to_string(tx);
    ret += ", ";
    ret += std::to_string(ty);
    std::string reduce_type = "";
    if (by == 1 && ty == 1) {
      reduce_type = AKG_ALL_REDUCE;
    } else if (direction == X_DIRECTION) {
      reduce_type = AKG_X_REDUCE;
    } else {
      reduce_type = AKG_Y_REDUCE;
    }
    ret += ", ";
    ret += reduce_type;

    reduce_data.akg_reduce_template_arg_ = ret;
  }
  void SetScalarTensorBind(ReduceData &reduce_data, std::string scalar_tensor_name) {
    Array<Expr> shapes;
    shapes.push_back(Expr(1));
    Type type = reduce_data.reduce_data_type_info_;

    Tensor tensor = placeholder(shapes, type, scalar_tensor_name);
    const Buffer buffer = decl_buffer(shapes, type, scalar_tensor_name);
    reduce_data.scalar_tensor_[scalar_tensor_name] = tensor;
    CHECK(reduce_data.scalar_tensor_[scalar_tensor_name].defined());

    scop_info_.user_config_.SetBind(tensor, buffer);
  }

  void SetSharedTensorBind(ReduceData &reduce_data) {
    auto thread_cfg = scop_info_.user_config_.GetThreadConfig();
    CHECK(thread_cfg) << "thread config is null.";
    int tx = thread_cfg->GetX().second;
    int ty = thread_cfg->GetY().second;

    int size = tx * ty;
    Array<Expr> shapes;
    shapes.push_back(Expr(size));
    Type type = reduce_data.reduce_data_type_info_;
    std::string shared_tensor_name = reduce_data.shared_compute_name_;

    Tensor tensor = placeholder(shapes, type, shared_tensor_name);
    const Buffer buffer = decl_buffer(shapes, type, shared_tensor_name);
    reduce_data.shared_tensor_ = tensor;

    scop_info_.user_config_.SetBind(tensor, buffer);
  }

  bool is_valid_reduce() { return reduce_valid_; }

  friend class ReduceStmtEmit;

 private:
  bool in_reduce_area_{false};
  ScopInfo &scop_info_;
  std::map<std::string, ReduceData> reduce_datas_;
  std::string cur_reduce_stmt_{""};
  bool reduce_valid_{false};
};

class AkgReduceStmtChange : public air::ir::IRMutator {
 public:
  explicit AkgReduceStmtChange(Tensor t, Array<Expr> args, std::string name) : t(t), args(args), name(name) {}
  ~AkgReduceStmtChange() override = default;

  Expr Mutate_(const Call *op, const Expr &e) final {
    if (op->name == name) {
      return Call::make(op->type, t->op->func_name(), args, op->call_type, t->op, op->value_index);
    }
    return IRMutator::Mutate_(op, e);
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    auto stmt = IRMutator::Mutate_(op, s);
    auto new_op = stmt.as<Provide>();
    CHECK(new_op);
    if (new_op->func->func_name() == name) {
      return Provide::make(t->op, new_op->value_index, new_op->value, args);
    }
    return stmt;
  }

 private:
  Tensor t;
  Array<Expr> args;
  std::string name;
};

class ReduceStmtEmit : public IRMutator {
 public:
  explicit ReduceStmtEmit(ReduceInfoCollect &info, ScopInfo &scop_info)
      : reduce_datas_(info.reduce_datas_), scop_info_(scop_info) {}
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) {
    auto key = op->attr_key;
    if (key == REDUCE_AREA_FLAG) {
      Stmt stmt = IRMutator::Mutate_(op, s);
      CHECK(!cur_reduce_stmt_.empty());
      auto reduce_data = reduce_datas_[cur_reduce_stmt_];
      stmt = InsertRealizeWithMemType(stmt, isl::id(scop_info_.ctx_, reduce_data.scalar_tensor_name_), MEM_TYPE_LOCAL);
      if (reduce_data.reduce_op_.find("SumOp") != std::string::npos) {
        stmt = InsertRealizeWithMemType(stmt, isl::id(scop_info_.ctx_, reduce_data.scalar_kht_name_), MEM_TYPE_LOCAL);
        stmt = InsertRealizeWithMemType(stmt, isl::id(scop_info_.ctx_, reduce_data.scalar_khy_name_), MEM_TYPE_LOCAL);
        stmt = InsertRealizeWithMemType(stmt, isl::id(scop_info_.ctx_, reduce_data.scalar_khc_name_), MEM_TYPE_LOCAL);
      }
      stmt =
        InsertRealizeWithMemType(stmt, isl::id(scop_info_.ctx_, reduce_data.shared_compute_name_), MEM_TYPE_SHARED);
      return stmt;
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Evaluate *op, const Stmt &s) {
    Expr value = op->value;
    if (value.as<Call>()) {
      auto call = value.as<Call>();
      auto name = call->name;
      if (ScopInfo::IsReduceInit(name)) {
        in_reduce_area_ = true;
        std::vector<std::string> strs = common::Split(name, "_");
        CHECK_EQ(strs.size(), REDUCE_FLAG_SIZE) << "red init format is not right!.";

        cur_reduce_stmt_ = strs[REDUCE_FLAG_STMT_PREFIX_POS] + "_" + strs[REDUCE_FLAG_STMT_NUM_POS];
        CHECK(reduce_datas_.find(cur_reduce_stmt_) != reduce_datas_.end());
        auto reduce_data = reduce_datas_[cur_reduce_stmt_];

        Array<Expr> args;
        args.push_back(Expr(0));
        Stmt scalar_stmt = Provide::make(reduce_data.scalar_tensor_[reduce_data.scalar_tensor_name_]->op, 0,
                                         reduce_data.init_value, args);
        if (reduce_data.reduce_op_.find("SumOp") != std::string::npos) {
          Stmt scalar_khc = Provide::make(reduce_data.scalar_tensor_[reduce_data.scalar_khc_name_]->op, 0,
                                          reduce_data.init_value, args);
          CHECK(scalar_khc.defined());
          scalar_stmt = Block::make(scalar_khc, scalar_stmt);
        }

        scalar_stmt = AttrStmt::make(Expr("INFO"), name, Expr(""), scalar_stmt);
        return scalar_stmt;
      } else if (ScopInfo::IsReduceUpdate(name)) {
        in_reduce_area_ = false;
        auto reduce_data = reduce_datas_[cur_reduce_stmt_];
        return MakeReduceStmt(reduce_data);
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) {
    if (in_reduce_area_) {
      Array<Expr> args_scalar;
      auto reduce_data = reduce_datas_[cur_reduce_stmt_];
      args_scalar.push_back(Expr(0));

      Stmt stmt = AkgReduceStmtChange(reduce_data.scalar_tensor_[reduce_data.scalar_tensor_name_], args_scalar,
                                      reduce_data.promoted_tensor_name_for_reduce_)
                    .Mutate(s);

      if (reduce_data.reduce_op_.find("SumOp") != std::string::npos) {
        auto pro = stmt.as<Provide>();
        CHECK(pro);
        auto value = pro->value;
        auto add = value.as<Add>();
        CHECK(add);
        auto add_a = add->a;
        auto add_b = add->b;
        reduce_data.input_tensor_expr_ =
          (add->a.as<Call>() && add->a.as<Call>()->name == reduce_data.scalar_tensor_name_) ? add_b : add_a;
        stmt = TransferToKaHanInterface(reduce_data);
      }

      return stmt;
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt InsertRealizeWithMemType(Stmt stmt, const isl::id &var, std::string mem) {
    stmt = FindInnerRealize(var.get_name()).Mutate(stmt);

    Tensor t = scop_info_.FindTensorWithLargestShape(var);
    Region bounds;

    // no isolate
    if (bounds.empty()) {
      for (auto j : t->shape) {
        bounds.push_back(Range::make_by_min_extent(Expr(0), j));
      }
    }

    // If isolate, make a new buffer
    auto buf = scop_info_.user_config_.GetBind().at(t);

    auto tt = placeholder(t->shape, t->dtype, t->op->name);

    stmt = TensorSubstitute(stmt, t->op, tt->op, tt->value_index);
    t = tt;
    if (scop_info_.analysis_result_.CountBufferDefInfo(var)) {
      auto decl = scop_info_.analysis_result_.GetBufferDefInfo(var);
      decl.tensor = t;
    }
    scop_info_.user_config_.SetBind(t, buf);
    stmt = TensorSubstitute2(stmt, t->op->func_name(), t->op, t->value_index);
    stmt = Realize::make(t->op, t->value_index, t->dtype, bounds, const_true(1), stmt);
    stmt = AttrStmt::make(t->op, air::ir::attr::realize_scope, Expr(mem), stmt);

    return stmt;
  }

  Stmt MakeReduceStmt(ReduceData &reduce_data) {
    std::string func_name = reduce_data.akg_reduce_api_;
    std::string op_info = reduce_data.reduce_op_ + "()";

    Expr template_arg0 = make_const(reduce_data.reduce_data_type_info_, 1);
    CHECK(!reduce_data.akg_reduce_template_arg_.empty());
    Expr template_arg1 = StringImm::make(reduce_data.akg_reduce_template_arg_);

    Array<Expr> args_a1;
    Expr a1 = Call::make(Int(32), reduce_data.reduce_op_, args_a1, Call::Extern);

    auto p = reduce_data.origin_reduce_stmt_;
    CHECK(p);
    Expr a2 = Call::make(p->value.type(), p->func->func_name(), p->args, Call::Halide, p->func, 0);
    a2 = Call::make(a2.type(), "&", {a2}, Call::Extern);

    Tensor tensor = scop_info_.FindTensor(reduce_data.shared_compute_name_);
    auto bind = scop_info_.user_config_.GetBind();
    Buffer buffer;
    for (auto &i : bind) {
      if (!i.first.defined()) continue;
      if (i.first == tensor) {
        buffer = i.second;
      }
    }

    CHECK(buffer.defined());

    Tensor tt = reduce_data.scalar_tensor_[reduce_data.scalar_tensor_name_];
    Array<Expr> args;
    args.push_back(Expr(0));
    Expr a4 = Call::make(tt->dtype, tt->op->func_name(), args, Call::Halide, tt->op, 0);

    auto thread_cfg = scop_info_.user_config_.GetThreadConfig();
    CHECK(thread_cfg);
    int tx = thread_cfg->GetX().second;
    int ty = thread_cfg->GetY().second;
    Expr a5 = Expr(tx);

    Stmt stmt = Evaluate::make(
      Call::make(Int(32), func_name, {template_arg0, template_arg1, a1, a2, buffer->data, a4, a5}, Call::Extern));

    stmt = AttrStmt::make(Expr("INFO"), REDUCE_LIB_TYPE_FLAG, scop_info_.user_config_.GetReduceLibType(), stmt);

    int size = tx * ty;
    stmt = AttrStmt::make(buffer->data, air::ir::attr::storage_scope, Expr(MEM_TYPE_SHARED),
                          Allocate::make(buffer->data, buffer->dtype, {Expr(size)}, const_true(), stmt));
    return stmt;
  }

  Stmt TransferToKaHanInterface(ReduceData &reduce_data) {
    std::string func_name = AKG_REDUCE_LIB_SPACE;
    func_name += "::";
    func_name += AKG_KAHAN_LIB_NAME;
    Expr template_arg0 = make_const(reduce_data.reduce_data_type_info_, 1);

    Array<Expr> args;
    args.push_back(Expr(0));

    Tensor tt = reduce_data.scalar_tensor_[reduce_data.scalar_khy_name_];
    Expr a1 = Call::make(tt->dtype, tt->op->func_name(), args, Call::Halide, tt->op, 0);
    a1 = Call::make(a1.type(), "&", {a1}, Call::Extern);

    tt = reduce_data.scalar_tensor_[reduce_data.scalar_kht_name_];
    Expr a2 = Call::make(tt->dtype, tt->op->func_name(), args, Call::Halide, tt->op, 0);
    a2 = Call::make(a2.type(), "&", {a2}, Call::Extern);

    tt = reduce_data.scalar_tensor_[reduce_data.scalar_khc_name_];
    Expr a3 = Call::make(tt->dtype, tt->op->func_name(), args, Call::Halide, tt->op, 0);
    a3 = Call::make(a3.type(), "&", {a3}, Call::Extern);

    tt = reduce_data.scalar_tensor_[reduce_data.scalar_tensor_name_];
    Expr a4 = Call::make(tt->dtype, tt->op->func_name(), args, Call::Halide, tt->op, 0);
    a4 = Call::make(a4.type(), "&", {a4}, Call::Extern);

    CHECK(reduce_data.input_tensor_expr_.defined());
    Stmt stmt = Evaluate::make(
      Call::make(Int(32), func_name, {template_arg0, a1, a2, a3, a4, reduce_data.input_tensor_expr_}, Call::Extern));

    return stmt;
  }

 private:
  std::map<std::string, ReduceData> reduce_datas_;
  bool in_reduce_area_{false};
  bool collect_area_stmt_{false};
  std::string cur_reduce_stmt_{""};
  ScopInfo &scop_info_;
  std::vector<Stmt> block_stmts_;
  int block_depth_{1};
  bool reduce_start_{false};
  bool reduce_end_{false};
  Stmt rest_part_;
};

struct AtomicReturnData {
  std::string reduce_op_;
  std::string akg_atomic_api_;
  std::string akg_atomic_template_arg_;
  Type output_tensor_data_type_info_;
  Expr atomic_rhs_;
  Stmt gm_write_stmt_;
};

class AtomicReturnStmtEmit : public IRMutator {
 public:
  explicit AtomicReturnStmtEmit(ScopInfo &scop_info) : scop_info_(scop_info) {}

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) {
    auto key = op->attr_key;
    if (IsStartsWith(key, REDUCE_ATOMIC_FLAG)) {
      in_atomic_area_ = true;
      std::vector<std::string> strs = common::Split(key, "_");
      CHECK_EQ(strs.size(), REDUCE_ATOMIC_FLAG_SIZE) << "atomic mark format is not right!.";
      atomic_data_.reduce_op_.clear();
      if (AkgSupportedReduceOp.count(strs[REDUCE_ATOMIC_FLAG_TYPE_POS])) {
        atomic_data_.reduce_op_ = AKG_REDUCE_LIB_SPACE;
        atomic_data_.reduce_op_ += "::";
        atomic_data_.reduce_op_ += strs[REDUCE_ATOMIC_FLAG_TYPE_POS];
      } else {
        CHECK(false) << "reduce op type is not supported!";
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) {
    if (in_atomic_area_) {
      in_atomic_area_ = false;
      Stmt stmt = IRMutator::Mutate_(op, s);
      atomic_data_.gm_write_stmt_ = stmt;
      auto op = stmt.as<Provide>();
      CHECK(op);
      atomic_data_.atomic_rhs_ = op->value;
      atomic_data_.output_tensor_data_type_info_ = scop_info_.GetDtypeOf(op->func->func_name());

      ConstructAtomicReturnFuncName();
      return MakeAtomicStmt();
    }
    return IRMutator::Mutate_(op, s);
  }

  void ConstructAtomicReturnFuncName() {
    std::string reduce_lib_namespace = "";
    std::string reduce_return_name = "";
    if (scop_info_.user_config_.GetReduceLibType() == REDUCE_LIB_TYPE_ORIGIN) {
      reduce_lib_namespace = AKG_REDUCE_LIB_SPACE;
      reduce_return_name = AKG_REDUCE_RETURN_NAME;
    } else if (scop_info_.user_config_.GetReduceLibType() == REDUCE_LIB_TYPE_PARIS) {
      reduce_lib_namespace = PARIS_REDUCE_LIB_SPACE;
      reduce_return_name = PARIS_REDUCE_RETURN_NAME;
    } else {
      CHECK(false) << "reduce lib type is invalid!"
                   << "\n";
    }
    std::string ret = "";
    ret += reduce_lib_namespace;
    ret += "::";
    ret += reduce_return_name;

    atomic_data_.akg_atomic_api_ = ret;
    ret = "";

    std::string op = atomic_data_.reduce_op_;
    ret += op;

    atomic_data_.akg_atomic_template_arg_ = ret;
  }

  Stmt MakeAtomicStmt() {
    std::string func_name = atomic_data_.akg_atomic_api_;

    Expr template_arg0 = make_const(atomic_data_.output_tensor_data_type_info_, 1);
    CHECK(!atomic_data_.akg_atomic_template_arg_.empty());
    Expr template_arg1 = StringImm::make(atomic_data_.akg_atomic_template_arg_);

    Expr a1 = atomic_data_.atomic_rhs_;

    auto p = atomic_data_.gm_write_stmt_.as<Provide>();
    CHECK(p);

    Expr a2 = Call::make(p->value.type(), p->func->func_name(), p->args, Call::Halide, p->func, 0);
    a2 = Call::make(a2.type(), "&", {a2}, Call::Extern);

    std::string op_info = atomic_data_.reduce_op_ + "()";

    Array<Expr> args;
    Expr a3 = Call::make(Int(32), atomic_data_.reduce_op_, args, Call::Extern);

    return Evaluate::make(Call::make(Int(32), func_name, {template_arg0, template_arg1, a1, a2, a3}, Call::Extern));
  }

 private:
  ScopInfo &scop_info_;
  AtomicReturnData atomic_data_;
  bool in_atomic_area_{false};
};

class ConditionExprMod : public air::ir::IRMutator {
 public:
  explicit ConditionExprMod(bool &is_found) : is_found_(is_found) {}
  ~ConditionExprMod() override = default;

  Expr Mutate_(const And *op, const Expr &e) override {
    auto o_a = op->a;
    auto o_b = op->b;
    auto a = air::ir::IRMutator::Mutate(op->a);
    auto b = air::ir::IRMutator::Mutate(op->b);
    if (!a.defined() && !b.defined()) return Expr();
    if (!a.defined()) return b;
    if (!b.defined()) return a;
    if (o_a.same_as(a) && o_b.same_as(b)) return e;
    return And::make(a, b);
  }

  Expr Mutate_(const Or *op, const Expr &e) override {
    auto o_a = op->a;
    auto o_b = op->b;
    auto a = air::ir::IRMutator::Mutate(op->a);
    auto b = air::ir::IRMutator::Mutate(op->b);
    if (!a.defined() && !b.defined()) return Expr();
    if (!a.defined()) return b;
    if (!b.defined()) return a;
    if (o_a.same_as(a) && o_b.same_as(b)) return e;
    return Or::make(a, b);
  }

  Expr Mutate_(const EQ *op, const Expr &e) override {
    Expr a = op->a;
    Expr b = op->b;

    bool rh_zero = false;
    bool lh_block = false;
    if (b.as<IntImm>()) {
      auto v = b.as<IntImm>();
      if (v->value == 0) rh_zero = true;
    }

    if (a.as<Variable>()) {
      auto v = a.as<Variable>();
      if (v->name_hint == BLOCK_IDX_X) {
        lh_block = true;
      }
    }

    if (rh_zero && lh_block) {
      is_found_ = true;
      return Expr();
    }
    return e;
  }

 private:
  bool &is_found_;
};

class InitStmtIndexModify : public IRMutator {
 public:
  explicit InitStmtIndexModify(ScopInfo &scop_info) : scop_info_(scop_info) {}

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) {
    auto key = op->attr_key;
    if (key == REDUCE_INIT_FLAG) {
      init_stmt_emit_ = true;
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const IfThenElse *op, const Stmt &s) {
    Stmt stmt = IRMutator::Mutate_(op, s);
    if (init_stmt_emit_) {
      if (scop_info_.user_config_.GetEnableAtomicAdd() && !scop_info_.analysis_result_.GetAtomicMarkers().empty()) {
        bool is_found = false;
        auto op = s.as<IfThenElse>();
        CHECK(op);
        auto condition = op->condition;
        condition = ConditionExprMod(is_found).Mutate(condition);
        if (is_found) {
          init_stmt_emit_ = false;
        }
        return IfThenElse::make(condition, op->then_case, op->else_case);
      }
    }
    return stmt;
  }

 private:
  ScopInfo &scop_info_;
  bool init_stmt_emit_{false};
};

class DeleteComplicatedSync : public IRMutator {
 public:
  DeleteComplicatedSync() {}

  Stmt Mutate_(const Block *op, const Stmt &s) {
    Stmt first = this->Mutate(op->first);
    if (first.as<Evaluate>()) {
      Expr value = first.as<Evaluate>()->value;
      if (value.as<Call>()) {
        auto call = value.as<Call>();
        auto name = call->name;
        if (name == STORAGE_SYNC) {
          emit_sync_ = true;
        } else {
          emit_sync_ = false;
        }
      }
    } else {
      emit_sync_ = false;
    }

    Stmt rest = this->Mutate(op->rest);

    if (!first.defined() && !rest.defined()) {
      return Stmt();
    }

    if (!first.defined() && rest.defined()) {
      return rest;
    }

    if (first.defined() && !rest.defined()) {
      return first;
    }

    if (first.same_as(op->first) && rest.same_as(op->rest)) {
      return s;
    } else {
      return Block::make(first, rest);
    }
  }

  Stmt Mutate_(const Evaluate *op, const Stmt &s) {
    Expr value = op->value;
    if (value.as<Call>()) {
      auto call = value.as<Call>();
      auto name = call->name;
      if (name == STORAGE_SYNC) {
        if (emit_sync_) {
          return Stmt();
        }
      }
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  bool emit_sync_{false};
};

Stmt EmitForReduce(Stmt stmt, ScopInfo &scop_info) {
  ReduceInfoCollect col(scop_info);
  col.Visit(stmt);

  if (!col.is_valid_reduce()) {
    return stmt;
  }

  stmt = ReduceStmtEmit(col, scop_info).Mutate(stmt);
  stmt = AtomicReturnStmtEmit(scop_info).Mutate(stmt);

  if (scop_info.user_config_.GetEnableAtomicAdd()) {
    stmt = InitStmtIndexModify(scop_info).Mutate(stmt);
  }

  stmt = DeleteComplicatedSync().Mutate(stmt);

  return stmt;
}
}  // namespace poly
}  // namespace ir
}  // namespace akg
