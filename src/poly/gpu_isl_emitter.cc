/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "poly/gpu_isl_emitter.h"
#include "pass/utils.h"
#include "gpu_emit/emit_pass.h"
#include <sstream>
#include <algorithm>

namespace akg {
namespace ir {
namespace poly {

Expr GpuIslEmitter::EmitLoad(const isl::ast_expr &expr, const Type type) {
  if (PRINT_EMITTER) {
    LOG(INFO) << ">>>>>>>>>>>>INPUT AST_NODE[LOAD]<<<<<<<<<<<<<<\n" << expr;
  }
  if (auto op = expr.as<isl::ast_expr_op>()) {
    if (auto access = op.as<isl::ast_expr_op_access>()) {
      CHECK(op.get_arg(0).as<isl::ast_expr_id>());
      auto var = op.get_arg(0).as<isl::ast_expr_id>().get_id();
      Array<Expr> local_args;
      for (unsigned int i = 1; i < op.get_n_arg(); ++i) {
        local_args.push_back(Interpret(op.get_arg(i)));
      }

      Tensor t = info_.FindTensor(var);
      auto call = Call::make(type, t->op->name, local_args, Call::CallType::Halide, t->op, t->value_index);
      if (PRINT_EMITTER) {
        LOG(INFO) << ">>>>>>>>>>>>OUTPUT STMT<<<<<<<<<<<<\n" << call;
      }
      return call;
    }
  }
  return Expr();
}

Stmt GpuIslEmitter::EmitRead(const isl::ast_node_user &node) {
  isl::id node_id = node.get_annotation();
  isl::pw_multi_aff iterator_map = node_info_map_.at(node_id).iterator_map;
  isl::pw_multi_aff hoisted = iterator_map.range_factor_range();
  isl::pw_multi_aff original = iterator_map.range_factor_domain().range_factor_range();

  isl::id original_tensor = original.get_tuple_id(isl_dim_out);

  auto build = node_info_map_.at(node_id).build;
  auto lhs = build.access_from(isl::multi_pw_aff(hoisted));
  auto rhs = build.access_from(isl::multi_pw_aff(original));

  Type type = info_.GetDtypeOf(rhs);
  if (auto op = lhs.as<isl::ast_expr_op>()) {
    if (auto access = op.as<isl::ast_expr_op_access>()) {
      Expr value = EmitLoad(rhs, type);
      auto var = op.get_arg(0).as<isl::ast_expr_id>().get_id();

      Array<Expr> local_args;
      for (unsigned int i = 1; i < op.get_n_arg(); ++i) {
        local_args.push_back(Interpret(op.get_arg(i)));
      }

      Tensor t = info_.FindTensor(var);
      CHECK(t.defined());
      return Provide::make(t->op, 0, value, local_args);
    }
  }
  return Stmt();
}

std::string SimplifyName(std::string input) {
  auto pos_local = input.find(LOCAL_SUFFIX);
  auto pos_shared = input.find(SHARE_SUFFIX);
  std::string res = input;
  if (pos_local != std::string::npos) {
    res = input.substr(0, pos_local);
  }
  if (pos_shared != std::string::npos) {
    res = res.substr(0, pos_shared);
  }
  return res;
}

Stmt GpuIslEmitter::EmitReadCore(const isl::ast_node_user &node) {
  isl::id node_id = node.get_annotation();
  isl::pw_multi_aff iterator_map = node_info_map_.at(node_id).iterator_map;
  isl::pw_multi_aff hoisted = iterator_map.range_factor_range();
  isl::pw_multi_aff original = iterator_map.range_factor_domain().range_factor_range();

  isl::id original_tensor = original.get_tuple_id(isl_dim_out);

  auto build = node_info_map_.at(node_id).build;
  auto lhs = build.access_from(isl::multi_pw_aff(hoisted));
  auto rhs = build.access_from(isl::multi_pw_aff(original));

  Type type = info_.GetDtypeOf(rhs);
  if (auto op = lhs.as<isl::ast_expr_op>()) {
    if (auto access = op.as<isl::ast_expr_op_access>()) {
      Expr value = EmitLoad(rhs, type);
      auto var = op.get_arg(0).as<isl::ast_expr_id>().get_id();

      Array<Expr> local_args;
      for (unsigned int i = 1; i < op.get_n_arg(); ++i) {
        local_args.push_back(Interpret(op.get_arg(i)));
      }

      Tensor t = info_.FindTensor(var);
      CHECK(t.defined());
      Stmt s = Provide::make(t->op, 0, value, local_args);

      auto op_new = s.as<Provide>();
      CHECK(op_new);
      const Call *call_value = op_new->value.as<Call>();
      CHECK(call_value != nullptr) << "Can only load fragment from a buffer";

      auto left_expr = MakeLeftCallFromProvide(op_new);
      auto left_call = left_expr.as<Call>();
      CHECK(left_call != nullptr) << "make right part call failed!";

      auto it = tensor_core_info_.strides_.find(call_value->name);
      CHECK(it != tensor_core_info_.strides_.end()) << "Cannot find stride for " << call_value->name;
      auto strides = it->second;
      CHECK_GE(strides.size(), 2);
      Expr stride = strides[strides.size() - 2];

      std::string call_name = op_new->func->func_name();
      Expr src = Call::make(call_value->type, "&", {op_new->value}, Call::Extern);

      Expr matrix_major;
      auto iter2 = tensor_core_info_.matrix_major_.find(SimplifyName(call_name));
      CHECK(iter2 != tensor_core_info_.matrix_major_.end()) << "Can not determine matrix major for " << call_name;
      if (iter2->second == COL_MAJOR) {
        matrix_major = StringImm::make(COL_MAJOR);
      } else if (iter2->second == ROW_MAJOR) {
        matrix_major = StringImm::make(ROW_MAJOR);
      } else {
        LOG(FATAL) << "invalid matrix major for " << call_name;
      }

      NodePtr<BufferNode> buffer_node = make_node<BufferNode>();
      EmitTensorCoreHelper helper(tensor_core_info_);
      helper.SetDataForLoad(src, stride, matrix_major, left_call, op_new, buffer_node);
      return helper.MakeLoadTransform();
    }
  }
  return Stmt();
}

Expr GpuIslEmitter::MakeLeftCallFromProvide(const Provide *op) {
  std::string name = op->func->func_name();
  Type type = info_.GetDtypeOf(name);
  Expr dst = Call::make(type, name, op->args, Call::Halide, op->func, 0);
  return dst;
}

Stmt GpuIslEmitter::EmitWrite(const isl::ast_node_user &node) {
  auto node_id = node.get_annotation();
  CHECK_GT(node_info_map_.count(node_id), 0);
  auto iterator_map = node_info_map_.at(node_id).iterator_map;
  auto hoisted = iterator_map.range_factor_range();
  auto original = iterator_map.range_factor_domain().range_factor_range();

  auto build = node_info_map_.at(node_id).build;
  auto rhs = build.access_from(isl::multi_pw_aff(hoisted));
  auto lhs = build.access_from(isl::multi_pw_aff(original));
  Type type = info_.GetDtypeOf(lhs);

  if (auto op = lhs.as<isl::ast_expr_op>()) {
    if (auto access = op.as<isl::ast_expr_op_access>()) {
      Expr value = EmitLoad(rhs, type);
      auto var = op.get_arg(0).as<isl::ast_expr_id>().get_id();

      Array<Expr> local_args;
      for (unsigned int i = 1; i < op.get_n_arg(); ++i) {
        local_args.push_back(Interpret(op.get_arg(static_cast<int>(i))));
      }

      Tensor t = info_.FindTensor(var);
      CHECK(t.defined());

      return Provide::make(t->op, 0, value, local_args);
    }
  }
  return Stmt();
}

Stmt GpuIslEmitter::EmitWriteCore(const isl::ast_node_user &node) {
  auto node_id = node.get_annotation();
  CHECK_GT(node_info_map_.count(node_id), 0);
  auto iterator_map = node_info_map_.at(node_id).iterator_map;
  auto hoisted = iterator_map.range_factor_range();
  auto original = iterator_map.range_factor_domain().range_factor_range();

  auto build = node_info_map_.at(node_id).build;
  auto rhs = build.access_from(isl::multi_pw_aff(hoisted));
  auto lhs = build.access_from(isl::multi_pw_aff(original));
  Type type = info_.GetDtypeOf(lhs);

  if (auto op = lhs.as<isl::ast_expr_op>()) {
    if (auto access = op.as<isl::ast_expr_op_access>()) {
      Expr value = EmitLoad(rhs, type);
      auto var = op.get_arg(0).as<isl::ast_expr_id>().get_id();

      Array<Expr> local_args;
      for (unsigned int i = 1; i < op.get_n_arg(); ++i) {
        local_args.push_back(Interpret(op.get_arg(static_cast<int>(i))));
      }

      Tensor t = info_.FindTensor(var);
      CHECK(t.defined());

      Stmt s = Provide::make(t->op, 0, value, local_args);

      auto op = s.as<Provide>();
      CHECK(op);

      auto lh_expr = MakeLeftCallFromProvide(op);
      auto lh_call = lh_expr.as<Call>();
      CHECK(lh_call != nullptr) << "make right part call failed!";

      auto it = tensor_core_info_.strides_.find(lh_call->name);
      CHECK(it != tensor_core_info_.strides_.end()) << "Cannot find stride for " << lh_call->name;
      auto strides = it->second;
      CHECK_GE(strides.size(), 2);
      Expr stride = strides[strides.size() - 2];

      Expr dst = lh_expr;
      dst = Call::make(Handle(), "&", {dst}, Call::Extern);

      auto call = op->value.as<Call>();
      NodePtr<BufferNode> buffer_node = make_node<BufferNode>();
      EmitTensorCoreHelper helper(tensor_core_info_);
      helper.SetDataForStore(dst, stride, call, buffer_node);
      return helper.MakeStoreTransform();
    }
  }
  return Stmt();
}

Stmt GpuIslEmitter::EmitWriteAtomic(const isl::ast_node_user &node) {
  auto node_id = node.get_annotation();
  CHECK_GT(node_info_map_.count(node_id), 0);
  auto iterator_map = node_info_map_.at(node_id).iterator_map;
  auto hoisted = iterator_map.range_factor_range();
  auto original = iterator_map.range_factor_domain().range_factor_range();

  auto build = node_info_map_.at(node_id).build;
  auto rhs = build.access_from(isl::multi_pw_aff(hoisted));
  auto lhs = build.access_from(isl::multi_pw_aff(original));

  auto opr = rhs.as<isl::ast_expr_op>();
  reduce_info_.output_promoted_tensor_name_for_atomic_ = opr.get_arg(0).as<isl::ast_expr_id>().get_id().name();
  reduce_info_.atomic_tensors_.insert(reduce_info_.output_promoted_tensor_name_for_atomic_);

  Type type = info_.GetDtypeOf(lhs);
  reduce_info_.output_tensor_data_type_info_ = type;

  if (auto op = lhs.as<isl::ast_expr_op>()) {
    if (auto access = op.as<isl::ast_expr_op_access>()) {
      Expr value = EmitLoad(rhs, type);
      reduce_info_.atomic_rhs_ = value;
      auto var = op.get_arg(0).as<isl::ast_expr_id>().get_id();

      Array<Expr> local_args;
      for (unsigned int i = 1; i < op.get_n_arg(); ++i) {
        Expr arg = Interpret(op.get_arg(static_cast<int>(i)));
        local_args.push_back(arg);
      }

      Tensor t = info_.FindTensor(var);
      CHECK(t.defined());

      return Provide::make(t->op, 0, value, local_args);
    }
  }
  return Stmt();
}

Stmt GpuIslEmitter::EmitSync() {
  return Evaluate::make(Call::make(Int(32), STORAGE_SYNC, {StringImm::make(SYNC_SCOP_SHARED)}, Call::Intrinsic));
}

void GpuIslEmitter::SetScalarTensorBind() {
  Array<Expr> shapes;
  shapes.push_back(Expr(1));
  Type type = reduce_info_.reduce_data_type_info_;
  std::string scalar_tensor_name = reduce_info_.scalar_tensor_name_;
  reduce_info_.added_tensors_.insert(scalar_tensor_name);

  Tensor tensor = placeholder(shapes, type, scalar_tensor_name);
  const Buffer buffer = decl_buffer(shapes, type, scalar_tensor_name);
  reduce_info_.scalar_tensor_ = tensor;

  info_.user_config_.SetBind(tensor, buffer);
}

void GpuIslEmitter::SetSharedTensorBind() {
  auto thread_cfg = info_.user_config_.GetThreadConfig();
  CHECK(thread_cfg) << "thread config is null.";
  int tx = thread_cfg->GetX().second;
  int ty = thread_cfg->GetY().second;

  int size = tx * ty;
  Array<Expr> shapes;
  shapes.push_back(Expr(size));
  Type type = reduce_info_.reduce_data_type_info_;
  std::string shared_tensor_name = reduce_info_.shared_compute_name_;
  reduce_info_.added_tensors_.insert(shared_tensor_name);

  Tensor tensor = placeholder(shapes, type, shared_tensor_name);
  const Buffer buffer = decl_buffer(shapes, type, shared_tensor_name);
  reduce_info_.shared_tensor_ = tensor;

  info_.user_config_.SetBind(tensor, buffer);
}

Stmt GpuIslEmitter::EmitReduceInit(const isl::ast_node_user &node) {
  CHECK(node.get_expr().isa<isl::ast_expr_op>());
  isl::ast_expr_op usr_expr = node.get_expr().as<isl::ast_expr_op>();
  CHECK(usr_expr);
  auto stmt_id = usr_expr.get_arg(0).as<isl::ast_expr_id>().get_id();

  CHECK(!reduce_info_.scalar_tensor_name_.empty()) << "scalar tensor info should not be empty!";

  std::vector<std::string> strs = common::Split(stmt_id.name(), "_");
  CHECK_EQ(strs.size(), REDUCE_FLAG_SIZE) << "red init format is not right!.";

  std::string stmt_name = strs[REDUCE_FLAG_STMT_PREFIX_POS] + "_" + strs[REDUCE_FLAG_STMT_NUM_POS];
  Expr init_value;
  for (auto it : info_.analysis_result_.GetReduceTensorInfoMap()) {
    if (it.first.name() == stmt_name) {
      init_value = it.second.init_value;
      break;
    }
  }

  Array<Expr> args;
  args.push_back(Expr(0));
  Stmt scalar_stmt = Provide::make(reduce_info_.scalar_tensor_->op, 0, init_value, args);

  CHECK(reduce_info_.reduce_area_stmt_.defined());
  reduce_info_.stmts_.insert(reduce_info_.stmts_.begin(), reduce_info_.reduce_area_stmt_);

  CHECK(scalar_stmt.defined());
  reduce_info_.stmts_.insert(reduce_info_.stmts_.begin(), scalar_stmt);

  MakeReduceStmt();

  Stmt stmt = Block::make(reduce_info_.stmts_);
  stmt = InsertRealizeWithMemType(stmt, isl::id(stmt_id.ctx(), reduce_info_.scalar_tensor_name_), MEM_TYPE_LOCAL);
  stmt = InsertRealizeWithMemType(stmt, isl::id(stmt_id.ctx(), reduce_info_.shared_compute_name_), MEM_TYPE_SHARED);

  ResetStatus();
  return stmt;
}

Stmt GpuIslEmitter::EmitUserStmt(const isl::ast_node_user &node) {
  CHECK(node.get_expr().isa<isl::ast_expr_op>());
  isl::ast_expr_op usr_expr = node.get_expr().as<isl::ast_expr_op>();
  stmt_id_ = usr_expr.get_arg(0).as<isl::ast_expr_id>().get_id();
  node_id_ = node.get_annotation();
  const Node *stmt_node = info_.analysis_result_.GetStatementMap().at(stmt_id_);
  CHECK(stmt_node);
  // compute VarMap to replace old iterators
  auto build = node_info_map_.at(node_id_).build;
  auto tuple = info_.analysis_result_.GetOperatorDomainMap().at(stmt_id_).tuple;
  auto iterator_map = node_info_map_.at(node_id_).iterator_map;

  auto ids = info_.analysis_result_.GetReduceInitIds();
  for (auto &i : ids) {
    if (i.get_name() == stmt_id_.get_name()) {
      reduce_info_.init_stmt_emit_ = true;
      break;
    }
  }

  var_map_.clear();
  for (unsigned int i = 0; i < tuple.size(); ++i) {
    isl::id isl_old_iter = tuple.get_id(i);
    auto isl_expr = build.expr_from(iterator_map.get_pw_aff(i));
    Expr halide_new_iter = Interpret(isl_expr);
    var_map_.emplace(isl_old_iter, halide_new_iter);
  }

  return EmitUserStmtContent(stmt_node);
}

void GpuIslEmitter::ResetStatus() {
  reduce_info_.stmts_.clear();
  reduce_info_.reduce_area_stmt_ = Stmt();
  reduce_info_.origin_reduce_stmt_ = Stmt();
  reduce_info_.gm_write_stmt_ = Stmt();
  reduce_info_.atomic_rhs_ = Expr();
  is_out_most_stmt_ = true;
}

Stmt GpuIslEmitter::EmitReduceUpdate(const isl::ast_node_user &node) {
  CHECK(node.get_expr().isa<isl::ast_expr_op>());
  isl::ast_expr_op usr_expr = node.get_expr().as<isl::ast_expr_op>();
  CHECK(usr_expr);
  auto stmt_id = usr_expr.get_arg(0).as<isl::ast_expr_id>().get_id();

  std::vector<std::string> strs = common::Split(stmt_id.name(), "_");
  CHECK_EQ(strs.size(), REDUCE_FLAG_SIZE) << "red update format is not right!.";

  reduce_info_.reduce_stmt_index_ = strs[REDUCE_FLAG_REDUCE_INDEX];
  reduce_info_.scalar_tensor_name_ = SCALAR_TENSOR_PREFIX;
  reduce_info_.scalar_tensor_name_ += reduce_info_.reduce_stmt_index_;

  reduce_info_.shared_compute_name_ = SHARED_TENSOR_PREFIX;
  reduce_info_.shared_compute_name_ += reduce_info_.reduce_stmt_index_;

  if (AkgSupportedReduceOp.count(strs[REDUCE_FLAG_TYPE_POS])) {
    reduce_info_.reduce_op_ = AKG_REDUCE_LIB_SPACE;
    reduce_info_.reduce_op_ += "::";
    reduce_info_.reduce_op_ += strs[REDUCE_FLAG_TYPE_POS];
  }
  CHECK(!reduce_info_.reduce_op_.empty()) << "reduce op should not be empty!";
  std::string stmt_name = strs[REDUCE_FLAG_STMT_PREFIX_POS] + "_" + strs[REDUCE_FLAG_STMT_NUM_POS];
  std::string origin_tensor_name = "";
  for (auto it : info_.analysis_result_.GetReduceTensorInfoMap()) {
    if (it.first.name() == stmt_name) {
      origin_tensor_name = it.second.write_tensor_name;
      reduce_info_.reduce_data_type_info_ = it.second.write_dtype;
      break;
    }
  }
  CHECK(!origin_tensor_name.empty()) << "origin_tensor_name should not be empty!";

  for (const auto &buffer : info_.analysis_result_.active_buffer_footprints_) {
    auto cluster_id = buffer.second.cluster_id;
    auto buf_def = info_.analysis_result_.GetBufferDefInfo(cluster_id);
    if (buf_def.tensor_id.name() == origin_tensor_name) {
      reduce_info_.promoted_tensor_name_for_reduce_ = cluster_id.name();
      break;
    }
  }

  MakeAkgReduceFuncName();
  SetScalarTensorBind();
  SetSharedTensorBind();

  return Stmt();
}

void GpuIslEmitter::MakeReduceStmt() {
  std::string func_name = reduce_info_.akg_reduce_api_;
  std::string op_info = reduce_info_.reduce_op_ + "()";

  Expr template_arg0 = make_const(reduce_info_.reduce_data_type_info_, 1);
  CHECK(!reduce_info_.akg_reduce_template_arg_.empty());
  Expr template_arg1 = StringImm::make(reduce_info_.akg_reduce_template_arg_);

  Array<Expr> args_a1;
  Expr a1 = Call::make(Int(32), reduce_info_.reduce_op_, args_a1, Call::Extern);

  auto p = reduce_info_.origin_reduce_stmt_.as<Provide>();
  CHECK(p);
  Expr a2 = Call::make(p->value.type(), p->func->func_name(), p->args, Call::Halide, p->func, 0);
  a2 = Call::make(a2.type(), "&", {a2}, Call::Extern);

  Tensor tensor = info_.FindTensor(reduce_info_.shared_compute_name_);
  auto bind = info_.user_config_.GetBind();
  Buffer buffer;
  for (auto &i : bind) {
    if (!i.first.defined()) continue;
    if (i.first == tensor) {
      buffer = i.second;
    }
  }

  CHECK(buffer.defined());

  Tensor tt = reduce_info_.scalar_tensor_;
  Array<Expr> args;
  args.push_back(Expr(0));
  Expr a4 = Call::make(tt->dtype, tt->op->func_name(), args, Call::Halide, tt->op, 0);

  auto thread_cfg = info_.user_config_.GetThreadConfig();
  CHECK(thread_cfg);
  int tx = thread_cfg->GetX().second;
  int ty = thread_cfg->GetY().second;
  Expr a5 = Expr(tx);

  Stmt stmt = Evaluate::make(
    Call::make(Int(32), func_name, {template_arg0, template_arg1, a1, a2, buffer->data, a4, a5}, Call::Extern));

  stmt = AttrStmt::make(Expr("INFO"), REDUCE_LIB_TYPE_FLAG, info_.user_config_.GetReduceLibType(), stmt);

  int size = tx * ty;
  stmt = AttrStmt::make(buffer->data, air::ir::attr::storage_scope, Expr(MEM_TYPE_SHARED),
                        Allocate::make(buffer->data, buffer->dtype, {Expr(size)}, const_true(), stmt));
  reduce_info_.stmts_.insert(reduce_info_.stmts_.end(), stmt);
  return;
}

Stmt GpuIslEmitter::MakeAtomicStmt() {
  std::string func_name = reduce_info_.akg_atomic_api_;

  Expr template_arg0 = make_const(reduce_info_.output_tensor_data_type_info_, 1);
  CHECK(!reduce_info_.akg_atomic_template_arg_.empty());
  Expr template_arg1 = StringImm::make(reduce_info_.akg_atomic_template_arg_);

  Expr a1 = reduce_info_.atomic_rhs_;

  auto p = reduce_info_.gm_write_stmt_.as<Provide>();
  CHECK(p);

  Expr a2 = Call::make(p->value.type(), p->func->func_name(), p->args, Call::Halide, p->func, 0);
  a2 = Call::make(a2.type(), "&", {a2}, Call::Extern);

  std::string op_info = reduce_info_.reduce_op_ + "()";

  Array<Expr> args;
  Expr a3 = Call::make(Int(32), reduce_info_.reduce_op_, args, Call::Extern);

  return Evaluate::make(Call::make(Int(32), func_name, {template_arg0, template_arg1, a1, a2, a3}, Call::Extern));
}

Stmt GpuIslEmitter::EmitReduceArea(const isl::ast_node_user &node) {
  bool add_to_reduce_area = false;
  if (in_reduce_area_ && is_out_most_stmt_) {
    add_to_reduce_area = true;
    is_out_most_stmt_ = false;
  }
  CHECK(node.get_expr().isa<isl::ast_expr_op>());
  isl::ast_expr_op usr_expr = node.get_expr().as<isl::ast_expr_op>();
  stmt_id_ = usr_expr.get_arg(0).as<isl::ast_expr_id>().get_id();
  node_id_ = node.get_annotation();
  const Node *stmt_node = info_.analysis_result_.GetStatementMap().at(stmt_id_);
  CHECK(stmt_node);
  // compute VarMap to replace old iterators
  auto build = node_info_map_.at(node_id_).build;
  auto tuple = info_.analysis_result_.GetOperatorDomainMap().at(stmt_id_).tuple;
  auto iterator_map = node_info_map_.at(node_id_).iterator_map;

  var_map_.clear();
  for (unsigned int i = 0; i < tuple.size(); ++i) {
    isl::id isl_old_iter = tuple.get_id(i);
    auto isl_expr = build.expr_from(iterator_map.get_pw_aff(i));
    Expr halide_new_iter = Interpret(isl_expr);
    var_map_.emplace(isl_old_iter, halide_new_iter);
  }

  Stmt stmt = EmitUserStmtContent(stmt_node);

  CHECK(!reduce_info_.promoted_tensor_name_for_reduce_.empty())
    << "promoted_tensor_name_for_reduce_ should not be empty";
  reduce_info_.reduce_stmt_[reduce_info_.promoted_tensor_name_for_reduce_] = stmt;
  reduce_info_.origin_reduce_stmt_ = stmt;

  Array<Expr> args_scalar;
  args_scalar.push_back(Expr(0));

  stmt = AkgReduceStmtChange(reduce_info_.scalar_tensor_, args_scalar, reduce_info_.promoted_tensor_name_for_reduce_)
           .Mutate(stmt);
  if (add_to_reduce_area) {
    reduce_info_.reduce_area_stmt_ = stmt;
    return Stmt();
  }

  return stmt;
}

Stmt GpuIslEmitter::EmitUserStmtCore(const isl::ast_node_user &node) {
  if (tensor_core_info_.matrix_info_[MMA_SYNC]) {
    return EmitUserStmtCoreSync(node);
  }
  return Stmt();
}

Stmt GpuIslEmitter::EmitUserStmtCoreSync(const isl::ast_node_user &node) {
  static int serial_number = MMA_SYNC_STMT_SERIAL;
  CHECK(node.get_expr().isa<isl::ast_expr_op>());
  isl::ast_expr_op usr_expr = node.get_expr().as<isl::ast_expr_op>();
  stmt_id_ = usr_expr.get_arg(0).as<isl::ast_expr_id>().get_id();
  node_id_ = node.get_annotation();
  const Node *stmt_node = info_.analysis_result_.GetStatementMap().at(stmt_id_);
  CHECK(stmt_node);
  // compute VarMap to replace old iterators
  auto build = node_info_map_.at(node_id_).build;
  auto tuple = info_.analysis_result_.GetOperatorDomainMap().at(stmt_id_).tuple;
  auto iterator_map = node_info_map_.at(node_id_).iterator_map;

  var_map_.clear();
  for (unsigned int i = 0; i < tuple.size(); ++i) {
    isl::id isl_old_iter = tuple.get_id(i);
    auto isl_expr = build.expr_from(iterator_map.get_pw_aff(i));
    Expr halide_new_iter = Interpret(isl_expr);
    var_map_.emplace(isl_old_iter, halide_new_iter);
  }

  Stmt s = EmitUserStmtContent(stmt_node);

  if (serial_number == MMA_SYNC_STMT_SERIAL) {
    serial_number = MMA_FILL_STMT_SERIAL;
    auto op = s.as<Provide>();
    auto left_expr = MakeLeftCallFromProvide(op);
    Type type = info_.GetDtypeOf(op->func->func_name());
    auto *add = op->value.as<Add>();
    CHECK(add) << "format error of bmm";
    auto mul = akg::common::SplitCast(add->b, type).as<Mul>();
    CHECK(mul) << "format error of bmm";

    auto load_a_expr = akg::common::SplitCast(mul->a, type);
    auto load_b_expr = akg::common::SplitCast(mul->b, type);

    Expr a = load_a_expr;
    Expr b = load_b_expr;
    Expr c = left_expr;

    NodePtr<BufferNode> buffer_node_a = make_node<BufferNode>();
    NodePtr<BufferNode> buffer_node_b = make_node<BufferNode>();
    NodePtr<BufferNode> buffer_node_c = make_node<BufferNode>();

    EmitTensorCoreHelper helper(tensor_core_info_);
    helper.SetDataForSync(a, b, c, buffer_node_a, buffer_node_b, buffer_node_c);
    return helper.MakeSyncTransform();
  } else if (serial_number == MMA_FILL_STMT_SERIAL) {
    serial_number = MMA_SYNC_STMT_SERIAL;
    auto op = s.as<Provide>();
    auto left_expr = MakeLeftCallFromProvide(op);
    auto left_call = left_expr.as<Call>();
    CHECK(left_call != nullptr) << "make right part call failed";

    if (op->value.as<FloatImm>() != nullptr || op->value.as<IntImm>() != nullptr) {
      NodePtr<BufferNode> buffer_node = make_node<BufferNode>();
      EmitTensorCoreHelper helper(tensor_core_info_);
      helper.SetDataForFill(op, left_call, buffer_node);
      return helper.MakeFillTransform();
    } else {
      CHECK(false) << "mma init stmt format error";
    }
  }

  return Stmt();
}

Stmt GpuIslEmitter::EmitStmt(const isl::ast_node_user &node) {
  CHECK(node.get_expr().isa<isl::ast_expr_op>());
  isl::ast_expr_op usr_expr = node.get_expr().as<isl::ast_expr_op>();
  CHECK(usr_expr);
  auto stmt_id = usr_expr.get_arg(0).as<isl::ast_expr_id>().get_id();
  auto node_id = node.get_annotation();

  if (info_.IsRead(stmt_id)) {
    Stmt s;
    is_sync_before_ = false;
    if (tensor_core_info_.core_area_) {
      s = EmitReadCore(node);
    } else {
      s = EmitRead(node);
      s = AttrStmt::make(Expr(""), GMREAD_FLAG, StringImm::make(GMREAD_FLAG), s);
    }
    return s;
  } else if (info_.IsWrite(stmt_id)) {
    if (info_.IsGMWrite(stmt_id)) {
      if (tensor_core_info_.core_area_) {
        is_sync_before_ = false;
        return EmitWriteCore(node);
      }
      auto iterator_map = node_info_map_.at(node_id).iterator_map;
      auto original = iterator_map.range_factor_domain().range_factor_range();
      auto srcid = original.get_tuple_id(isl_dim_out);
      bool no_need_to_emit = NoNeedToEmitForTempTensor(srcid);
      if (no_need_to_emit) return Stmt();

      if (reduce_info_.is_atomic) {
        reduce_info_.gm_write_stmt_ = EmitWriteAtomic(node);
        ConstructAtomicReturnFuncName();
        is_sync_before_ = false;
        reduce_info_.is_atomic = false;
        return MakeAtomicStmt();
      }
      is_sync_before_ = false;
      if (tensor_core_info_.core_area_) {
        return EmitWriteCore(node);
      } else {
        return EmitWrite(node);
      }
    }
    is_sync_before_ = false;
    return EmitWrite(node);
  } else if (info_.IsSync(stmt_id)) {
    if (is_sync_before_) {
      return Stmt();
    }
    Stmt s = EmitSync();
    is_sync_before_ = true;
    return s;
  } else if (info_.IsReduceInit(stmt_id)) {
    is_sync_before_ = false;
    in_reduce_area_ = false;
    return EmitReduceInit(node);
  } else if (in_reduce_area_) {
    is_sync_before_ = false;
    return EmitReduceArea(node);
  } else if (info_.IsReduceUpdate(stmt_id)) {
    is_sync_before_ = false;
    Stmt s = EmitReduceUpdate(node);
    in_reduce_area_ = true;
    return s;
  } else {
    is_sync_before_ = false;
    Stmt s;
    if (tensor_core_info_.core_area_) {
      s = EmitUserStmtCore(node);
    } else {
      s = EmitUserStmt(node);
    }

    return s;
  }
}

void GpuIslEmitter::MakeAkgReduceFuncName() {
  auto thread_cfg = info_.user_config_.GetThreadConfig();
  CHECK(thread_cfg) << "thread config is null.";
  auto block_cfg = info_.user_config_.GetBlockConfig();
  CHECK(block_cfg) << "thread config is null.";
  int tx = thread_cfg->GetX().second;
  int ty = thread_cfg->GetY().second;
  int by = block_cfg->GetY().second;
  std::string direction = info_.analysis_result_.GetReduceDirection();
  CHECK(!direction.empty()) << "direction should not be empty!";
  std::string direction_size = "";
  if (direction == X_DIRECTION) {
    direction_size = std::to_string(tx);
  } else {
    direction_size = std::to_string(ty);
  }

  std::string reduce_lib_namespace = "";
  std::string reduce_lib_name = "";
  if (info_.user_config_.GetReduceLibType() == REDUCE_LIB_TYPE_ORIGIN) {
    reduce_lib_namespace = AKG_REDUCE_LIB_SPACE;
    reduce_lib_name = AKG_REDUCE_LIB_NAME;
  } else if (info_.user_config_.GetReduceLibType() == REDUCE_LIB_TYPE_PARIS) {
    reduce_lib_namespace = PARIS_REDUCE_LIB_SPACE;
    reduce_lib_name = PARIS_REDUCE_LIB_NAME;
  } else {
    CHECK(false) << "reduce lib type is invalid!"
                 << "\n";
  }
  std::string ret = reduce_lib_namespace;
  ret += "::";
  ret += reduce_lib_name;

  reduce_info_.akg_reduce_api_ = ret;
  ret = "";

  std::string op = reduce_info_.reduce_op_;
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

  reduce_info_.akg_reduce_template_arg_ = ret;
}

void GpuIslEmitter::ConstructAtomicReturnFuncName() {
  std::string reduce_lib_namespace = "";
  std::string reduce_return_name = "";
  if (info_.user_config_.GetReduceLibType() == REDUCE_LIB_TYPE_ORIGIN) {
    reduce_lib_namespace = AKG_REDUCE_LIB_SPACE;
    reduce_return_name = AKG_REDUCE_RETURN_NAME;
  } else if (info_.user_config_.GetReduceLibType() == REDUCE_LIB_TYPE_PARIS) {
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

  reduce_info_.akg_atomic_api_ = ret;
  ret = "";

  std::string op = reduce_info_.reduce_op_;
  ret += op;

  reduce_info_.akg_atomic_template_arg_ = ret;
}

bool GpuIslEmitter::NoNeedToEmitForTempTensor(const isl::id &id) {
  bool no_need = true;
  auto origin_binds = info_.user_config_.GetOriginBind();
  for (auto i : origin_binds) {
    if (!i.first.defined()) continue;
    std::string name = i.first->op->name;
    if (name == id.name()) {
      no_need = false;
      break;
    }
  }
  return no_need;
}

Stmt GpuIslEmitter::EmitBlock(const isl::ast_node_block &block_node) {
  bool add_to_reduce_area = false;
  if (in_reduce_area_ && is_out_most_stmt_) {
    add_to_reduce_area = true;
    is_out_most_stmt_ = false;
  }

  std::vector<Stmt> stmts;

  int num = block_node.get_children().size();
  int last_num = 0;
  for (int i = num - 1; i >= 0; --i) {
    auto child = block_node.get_children().at(i);

    if (auto node = child.as<isl::ast_node_user>()) {
      CHECK(node.get_expr().isa<isl::ast_expr_op>());
      isl::ast_expr_op usr_expr = node.get_expr().as<isl::ast_expr_op>();
      CHECK(usr_expr);
      auto stmt_id = usr_expr.get_arg(0).as<isl::ast_expr_id>().get_id();
      if (info_.IsRealize(stmt_id)) {
        isl::id new_stmt_id = isl::id(stmt_id.ctx(), stmt_id.name().substr(REALIZE_PREFIX_LEN));
        int stmt_num = stmts.size();
        CHECK_NE(stmt_num, 0) << "when stmt_num is zero, no realize should be emitted!.";
        if (stmt_num == 1) {
          stmts[0] = InsertRealize(stmts[0], new_stmt_id);
        } else {
          if (stmt_num - last_num == 1) {
            stmts[0] = InsertRealize(stmts[0], new_stmt_id);
          } else {
            for (int index = stmt_num - 2 - last_num; index >= 0; --index) {
              auto p_index = static_cast<unsigned int>(index);
              stmts[p_index] = Block::make(stmts[p_index], stmts[p_index + 1]);
            }
            stmts[0] = InsertRealize(stmts[0], new_stmt_id);
          }
        }
        last_num = stmt_num - 1;
        continue;
      }
    }

    Stmt body = EmitAst(child);
    if (!body.defined()) continue;
    stmts.insert(stmts.begin(), body);
  }

  int len = stmts.size();

  if (len == 0) {
    return Stmt();
  }

  if (last_num == len - 1) {
    if (add_to_reduce_area) {
      reduce_info_.reduce_area_stmt_ = stmts[0];
      return Stmt();
    }
    return stmts[0];
  } else {
    for (int index = len - 2 - last_num; index >= 0; --index) {
      auto p_index = static_cast<unsigned int>(index);
      stmts[p_index] = Block::make(stmts[p_index], stmts[p_index + 1]);
    }
    if (add_to_reduce_area) {
      reduce_info_.reduce_area_stmt_ = stmts[0];
      return Stmt();
    }
    return stmts[0];
  }
}

Stmt GpuIslEmitter::EmitFor(const isl::ast_node_for &node) {
  bool add_to_reduce_area = false;
  if (in_reduce_area_ && is_out_most_stmt_) {
    add_to_reduce_area = true;
    is_out_most_stmt_ = false;
  }
  isl::id isl_iter_id = node.get_iterator().as<isl::ast_expr_id>().get_id();
  VarExpr iter_expr(isl_iter_id.to_str());
  PushIter(iter_expr.get());

  Expr init_expr = Interpret(node.get_init());

  auto isl_cond = node.get_cond().as<isl::ast_expr_op>();
  CHECK(isl_cond.as<isl::ast_expr_op_lt>() || isl_cond.as<isl::ast_expr_op_le>());
  auto cond_lhs = isl_cond.get_arg(0).as<isl::ast_expr_id>();
  CHECK(cond_lhs);
  CHECK_EQ(cond_lhs.get_id(), isl_iter_id);
  Expr cond_expr = Interpret(isl_cond.get_arg(1));

  int64_t inc = static_cast<int64_t>(WrappedStrtol(node.get_inc().to_C_str()));
  CHECK_NE(inc, 0) << "stride should not be zero!.";

  bool need_to_modify_inc_ = false;
  if (inc != 1) {
    need_to_modify_inc_ = true;
    Expr original_init_expr = init_expr;
    init_expr = ModifyTheInitExpr(init_expr);
    cond_expr = ModifyTheCondExpr(cond_expr, static_cast<int>(inc));
    Expr modify_iter = ModifyTheIterExpr(iter_expr, static_cast<int>(inc), original_init_expr);
    stride_modify_iter_map_[iter_expr.get()] = modify_iter;
  }

  if (isl_cond.as<isl::ast_expr_op_le>()) {
    cond_expr = Simplify(cond_expr + 1);
  }

  cond_expr = Simplify(cond_expr - init_expr);

  // add for tensor core

  if (tensor_core_info_.core_area_) {
    tensor_core_info_.core_area_for_extent_[iter_expr] = cond_expr;
  }

  if (tensor_core_info_.fragment_axis_begin_) {
    if (tensor_core_info_.is_fragment_m_) {
      tensor_core_info_.fragment_m_ = cond_expr;
    } else if (tensor_core_info_.is_fragment_n_) {
      tensor_core_info_.fragment_n_ = cond_expr;
    }
  }

  Stmt body_stmt = EmitAst(node.get_body());

  if (!body_stmt.defined()) {
    PopIter(iter_expr.get());
    if (tensor_core_info_.core_area_) {
      tensor_core_info_.core_area_for_extent_.erase(iter_expr);
    }
    return Stmt();
  }

  if (need_to_modify_inc_) {
    stride_modify_iter_map_.erase(iter_expr.get());
  }
  PopIter(iter_expr.get());
  if (tensor_core_info_.core_area_) {
    tensor_core_info_.core_area_for_extent_.erase(iter_expr);
  }
  Stmt stmt = For::make(iter_expr, init_expr, cond_expr, ForType::Serial, DeviceAPI::None, body_stmt);
  if (add_to_reduce_area) {
    reduce_info_.reduce_area_stmt_ = stmt;
    return Stmt();
  }
  return stmt;
}

Stmt GpuIslEmitter::EmitIf(const isl::ast_node_if &node) {
  bool add_to_reduce_area = false;
  if (in_reduce_area_ && is_out_most_stmt_) {
    add_to_reduce_area = true;
    is_out_most_stmt_ = false;
  }

  Expr cond_expr = Interpret(node.get_cond());
  cur_if_list_.push_back(cond_expr.get());
  Stmt then_case = EmitAst(node.get_then_node());
  if (!then_case.defined()) {
    return Stmt();
  }
  Stmt else_case;
  if (node.has_else_node()) {
    else_case = EmitAst(node.get_else_node());
  }
  cur_if_list_.pop_back();
  if (reduce_info_.init_stmt_emit_) {
    reduce_info_.init_stmt_emit_ = false;
    if (info_.user_config_.GetEnableAtomicAdd()) {
      cond_expr = ConditionExprMod().Mutate(cond_expr);
    }
  }

  Stmt s;
  if (!cond_expr.defined()) {
    s = then_case;
  } else {
    s = IfThenElse::make(cond_expr, then_case, else_case);
  }

  if (add_to_reduce_area) {
    reduce_info_.reduce_area_stmt_ = s;
    return Stmt();
  }

  return s;
}

Expr GpuIslEmitter::ModifyTheInitExpr(const Expr &e) { return 0; }

Expr GpuIslEmitter::ModifyTheCondExpr(const Expr &e, int inc) { return e / Expr(inc); }

Expr GpuIslEmitter::ModifyTheIterExpr(const VarExpr &iter, int inc, const Expr &init) {
  return Simplify(iter * inc + init);
}

int GpuIslEmitter::GetThreadExtent(const std::string &name) {
  if (name == BLOCK_IDX_X || name == BLOCK_IDX_Y || name == BLOCK_IDX_Z) {
    auto block_cfg = info_.user_config_.GetBlockConfig();
    CHECK(block_cfg) << "block config is null.";
    return name == BLOCK_IDX_X ? block_cfg->GetX().second
                               : (name == BLOCK_IDX_Y ? block_cfg->GetY().second : block_cfg->GetZ().second);
  }

  if (name == THREAD_IDX_X || name == THREAD_IDX_Y || name == THREAD_IDX_Z) {
    auto thread_cfg = info_.user_config_.GetThreadConfig();
    CHECK(thread_cfg) << "thread config is null.";
    if (info_.user_config_.GetEnableOneDimThread()) {
      return name == THREAD_IDX_X ? (thread_cfg->GetX().second * thread_cfg->GetY().second * thread_cfg->GetZ().second)
                                  : 1;
    }
    return name == THREAD_IDX_X ? thread_cfg->GetX().second
                                : (name == THREAD_IDX_Y ? thread_cfg->GetY().second : thread_cfg->GetZ().second);
  }
  LOG(WARNING) << "Unrecognized thread name " << name;
  return 1;
}

void GpuIslEmitter::PrepareDataForTensorCore() {
  auto binds = info_.user_config_.GetBind();

  auto thread_cfg = info_.user_config_.GetThreadConfig();
  CHECK(thread_cfg) << "thread config is null";
  int tx = thread_cfg->GetX().second;
  int ty = thread_cfg->GetY().second;
  int tz = thread_cfg->GetZ().second;

  if (info_.user_config_.GetEnableOneDimThread()) {
    tx = tx * ty * tz;
    ty = 1;
    tz = 1;
  }

  for (auto i : binds) {
    if (!i.first.defined()) continue;
    if (!i.second.defined()) continue;
    auto t = i.first;
    auto b = i.second;

    std::string name = t->op->name;

    air::ir::TensorKey key{t->op, t->value_index};
    Region bounds;
    if (bounds.empty()) {
      for (auto j : t->shape) {
        bounds.push_back(Range::make_by_min_extent(Expr(0), j));
      }
    }

    tensor_core_info_.bounds_[key] = bounds;

    Array<Expr> strides;
    for (size_t i = 1; i < b->shape.size(); ++i) {
      Expr stride = IntImm::make(Int(32), 1);
      for (size_t j = b->shape.size() - 1; j >= i; --j) {
        stride = Mul::make(stride, b->shape[j]);
      }
      strides.push_back(stride);
    }
    strides.push_back(make_const(Int(32), 1));
    tensor_core_info_.strides_[name] = strides;
  }

  auto tile_size = info_.analysis_result_.GetTileSizes();
  CHECK_GE(tile_size.size(), 3) << "tile size should be greater to 3";
  int len = tile_size.size();
  tensor_core_info_.warp_tile_.m = tile_size[len - 3].c0_tiling_size;
  tensor_core_info_.warp_tile_.n = tile_size[len - 2].c0_tiling_size;
  tensor_core_info_.warp_tile_.k = tile_size[len - 1].c0_tiling_size;

  bool result = CheckTileValid(tensor_core_info_.warp_tile_);
  CHECK(result) << "tile set is not valid!";

  tensor_core_info_.thread_tile_.m = tensor_core_info_.warp_tile_.m / tx;
  tensor_core_info_.thread_tile_.n = tx / 2;
  tensor_core_info_.thread_tile_.k = tile_size[2].c0_tiling_size / tz;

  tensor_core_info_.matrix_abc_ = info_.analysis_result_.GetMatrixMatmulMap();
  tensor_core_info_.matrix_major_ = info_.analysis_result_.GetMatrixMatmulMajor();

  for (auto &i : tensor_core_info_.matrix_abc_) {
    tensor_core_info_.frag_reg_.insert(i.first + LOCAL_SUFFIX);
  }

  tensor_core_info_.warp_threads_y_ = 32 / tx;
  tensor_core_info_.warp_threads_x_ = tx;
}

bool GpuIslEmitter::CheckTileValid(Tile tile) {
  if (tile.m == 16 && tile.n == 16 && tile.k == 4) {
    tensor_core_info_.wmma_scope_ = "akg";
    return true;
  }
  if (tile.m == 16 && tile.n == 16 && tile.k == 16) {
    tensor_core_info_.wmma_scope_ = "nvcuda";
    return true;
  }
  if (tile.m == 8 && tile.n == 32 && tile.k == 16) {
    tensor_core_info_.wmma_scope_ = "nvcuda";
    return true;
  }
  if (tile.m == 32 && tile.n == 8 && tile.k == 16) {
    tensor_core_info_.wmma_scope_ = "nvcuda";
    return true;
  }
  return false;
}

Stmt GpuIslEmitter::Emit(const isl::ast_node &node) {
  Stmt stmt = EmitAst(node);

  // emit realize for temporary tensor
  stmt = EmitRealizeForGlobalTensor(stmt);

  // iter var node attr emit
  std::map<std::string, VarExpr>::iterator it;
  for (it = iter_name_map_.begin(); it != iter_name_map_.end(); it++) {
    IterVar axis = IterVarNode::make(Range(), it->second, air::kThreadIndex, it->second->name_hint);
    stmt = AttrStmt::make(axis, air::ir::attr::thread_extent, Expr(GetThreadExtent(it->second->name_hint)), stmt);
  }

  // attr for one dimension mapping
  if (info_.user_config_.GetEnableOneDimThread()) {
    auto thread_cfg = info_.user_config_.GetThreadConfig();
    CHECK(thread_cfg) << "thread config is null.";
    int tx = thread_cfg->GetX().second;
    stmt = AttrStmt::make(Expr(""), ORIGIN_THREAD_DIM_X, Expr(tx), stmt);
  }

  // add tensor core plan two attr
  if (info_.user_config_.GetEnableTensorCore()) {
    if (info_.user_config_.GetEnableTensorCoreUsePoly()) {
      stmt = AttrStmt::make(Expr(""), "pragma_tensor_core", StringImm::make(TENSOR_CORE_MODE_TWO), stmt);
      stmt = AttrStmt::make(Expr("INFO"), "wmma_scope", StringImm::make(tensor_core_info_.wmma_scope_), stmt);
    } else {
      stmt = AttrStmt::make(Expr(""), "pragma_tensor_core", StringImm::make(TENSOR_CORE_MODE_ONE), stmt);
    }
  }

  if (tensor_core_info_.is_tensor_core_ && info_.user_config_.GetEnableTensorCoreUsePoly()) {
    stmt = AddMmaAttrFlag(tensor_core_info_).Mutate(stmt);
    stmt = EmitForTensorCore(stmt, tensor_core_info_);
  } else if (info_.user_config_.GetEnableTensorCore()) {
    tensor_core_info_.cast_tensors_ = info_.analysis_result_.GetCastTensors();
    stmt = EmitForTensorCoreDesignOne(stmt, tensor_core_info_);
  }

  return stmt;
}

Stmt GpuIslEmitter::EmitRealizeForGlobalTensor(Stmt stmt) {
  auto binds = info_.user_config_.GetBind();
  auto origin_binds = info_.user_config_.GetOriginBind();
  std::unordered_set<std::string> tensor_name;

  for (auto i : binds) {
    if (!i.first.defined()) continue;
    tensor_name.insert(i.first->op->name);
  }

  for (auto i : binds) {
    if (!i.first.defined()) continue;
    // input and output tensor, no need to emit realize
    if (origin_binds.find(i.first) != origin_binds.end()) {
      continue;
    }

    // promoted tensor, the realize info already emitted before
    std::string name = i.first->op->name;
    if (IsEndsWith(name, MEM_TYPE_SHARED) || IsEndsWith(name, MEM_TYPE_LOCAL)) {
      continue;
    }

    // if the tensor is temporary, but has already promoted, there is no need to emit realize
    if (tensor_name.find(name + "_" + MEM_TYPE_SHARED) != tensor_name.end() ||
        tensor_name.find(name + "_" + MEM_TYPE_LOCAL) != tensor_name.end()) {
      continue;
    }

    if (reduce_info_.added_tensors_.find(name) != reduce_info_.added_tensors_.end()) {
      continue;
    }

    // if the tensor is temporary and it is not promoted, it needs to emit realize
    stmt = InsertRealize(stmt, isl::id(info_.GetCtx(), name));
  }
  return stmt;
}

Stmt GpuIslEmitter::EmitMark(const isl::ast_node_mark &node) {
  bool add_to_reduce_area = false;
  if (in_reduce_area_ && is_out_most_stmt_) {
    add_to_reduce_area = true;
    is_out_most_stmt_ = false;
  }

  std::string mark = node.get_id().get_name();
  if (mark == MIND_TRICKS_SWIZZLE_MARKER) {
    auto stmt = EmitAst(node.get_node());
    stmt = AttrStmt::make(make_zero(Int(32)), MIND_TRICKS_SWIZZLE_PRAGMA, Expr(1), stmt);
    return stmt;
  }

  if (IsStartsWith(mark, REDUCE_ATOMIC_FLAG)) {
    std::vector<std::string> strs = common::Split(mark, "_");
    CHECK_EQ(strs.size(), REDUCE_ATOMIC_FLAG_SIZE) << "atomic mark format is not right!.";
    reduce_info_.reduce_op_.clear();
    if (AkgSupportedReduceOp.count(strs[REDUCE_ATOMIC_FLAG_TYPE_POS])) {
      reduce_info_.reduce_op_ = AKG_REDUCE_LIB_SPACE;
      reduce_info_.reduce_op_ += "::";
      reduce_info_.reduce_op_ += strs[REDUCE_ATOMIC_FLAG_TYPE_POS];
    }
    CHECK(!reduce_info_.reduce_op_.empty()) << "reduce op should not be empty!";

    if (strs[REDUCE_ATOMIC_FLAG_POS] == REDUCE_ATOMIC_FLAG) {
      reduce_info_.is_atomic = true;
    }
  }

  // add for tensor core
  if ((mark == MATRIX_A) || (mark == MATRIX_B) || (mark == MATRIX_C) || (mark == WARP_MARKER)) {
    if (!tensor_core_info_.data_is_set_) {
      PrepareDataForTensorCore();
      tensor_core_info_.data_is_set_ = true;
    }
    tensor_core_info_.fragment_axis_begin_ = false;
    if (mark == WARP_MARKER) {
      mark = MMA_SYNC;
    }
    if (mark == MATRIX_C) {
      mark = MMA_C;
    }

    if (!tensor_core_info_.data_is_set_) {
      PrepareDataForTensorCore();
      tensor_core_info_.data_is_set_ = true;
    }

    tensor_core_info_.is_tensor_core_ = true;
    tensor_core_info_.matrix_info_[mark] = true;
    tensor_core_info_.core_area_ = true;

    Stmt stmt = EmitAst(node.get_node());
    stmt = DeleteUselessFor().Mutate(stmt);
    tensor_core_info_.matrix_info_[mark] = false;
    tensor_core_info_.core_area_ = false;
    return AttrStmt::make(Expr("INFO"), mark, StringImm::make(mark), stmt);
  }

  if ((mark == FRAGMENT_A) || (mark == FRAGMENT_B)) {
    tensor_core_info_.fragment_axis_begin_ = true;
    if (mark == FRAGMENT_A) {
      tensor_core_info_.is_fragment_m_ = true;
    } else if (mark == FRAGMENT_B) {
      tensor_core_info_.is_fragment_n_ = true;
    }
    Stmt stmt = EmitAst(node.get_node());
    tensor_core_info_.fragment_axis_begin_ = false;
    tensor_core_info_.is_fragment_m_ = false;
    tensor_core_info_.is_fragment_n_ = false;
    if (!stmt.defined()) {
      return Stmt();
    }
    return AttrStmt::make(Expr("INFO"), mark, StringImm::make(mark), stmt);
  }

  // add for prefetch pass
  if (mark == PROMOTE_GLOBAL_TO_SHARED_AB) {
    Stmt stmt = EmitAst(node.get_node());
    if (!stmt.defined()) {
      return Stmt();
    }
    return AttrStmt::make(Expr("INFO"), SHARED_MEM_PROMOTED_COMPLETE, StringImm::make(SHARED_MEM_PROMOTED_COMPLETE),
                          stmt);
  }

  Stmt stmt;

  if ((mark == PROMOTE_VECTORIZATION) || (mark == PROMOTE_LOCAL_TO_GLOBAL)) {
    stmt = EmitAst(node.get_node());
    if (!stmt.defined()) {
      return Stmt();
    }
    stmt = AttrStmt::make(Expr("INFO"), mark, StringImm::make(mark), stmt);
  } else {
    stmt = EmitAst(node.get_node());
  }

  if (add_to_reduce_area) {
    reduce_info_.reduce_area_stmt_ = stmt;
    return Stmt();
  }
  return stmt;
}

std::string GpuIslEmitter::FindRealizeScopeToString(const isl::id &var) {
  if (info_.analysis_result_.CountBufferDefInfo(var)) {
    auto tensor_info = info_.analysis_result_.GetBufferDefInfo(var);
    MemType mem_type = tensor_info.DstMemType();

    switch (mem_type) {
      case MemType::SHARED_:
        return MEM_TYPE_SHARED;
      case MemType::LOCAL_:
        return MEM_TYPE_LOCAL;
      default:
        LOG(FATAL) << "unexpected mem_type of var " << var;
        return "ERROR";
    }
  }
  return "";
}

Expr GpuIslEmitter::FindRealizeScope(const isl::id &var) { return Expr(FindRealizeScopeToString(var)); }

Stmt GpuIslEmitter::InsertRealize(Stmt stmt, const isl::id &var) {
  stmt = FindInnerRealize(var.get_name()).Mutate(stmt);

  // A tensor may be defined multiple times in BufferDefInfo due to nested realize.
  // Because we cannot determine which one we actually want, we have to be conservative here
  // and allocate space for the largest shape to avoid overflow.
  Tensor t = info_.FindTensorWithLargestShape(var);
  Region bounds;

  // no isolate
  if (bounds.empty()) {
    for (auto j : t->shape) {
      bounds.push_back(Range::make_by_min_extent(Expr(0), j));
    }
  }

  // If isolate, make a new buffer
  auto buf = info_.user_config_.GetBind().at(t);

  auto tt = placeholder(t->shape, t->dtype, t->op->name);

  stmt = TensorSubstitute(stmt, t->op, tt->op, tt->value_index);
  if (tensor_core_info_.is_tensor_core_) {
    stmt = TensorSubstituteTensorCore(t->op, tt->op, tt->value_index).Mutate(stmt);
  }
  t = tt;
  if (info_.analysis_result_.CountBufferDefInfo(var)) {
    auto decl = info_.analysis_result_.GetBufferDefInfo(var);
    decl.tensor = t;
  }
  info_.user_config_.SetBind(t, buf);
  stmt = TensorSubstitute2(stmt, t->op->func_name(), t->op, t->value_index);
  stmt = Realize::make(t->op, t->value_index, t->dtype, bounds, const_true(1), stmt);
  realized_.insert(t);
  stmt = AttrStmt::make(t->op, air::ir::attr::realize_scope, FindRealizeScope(var), stmt);

  return stmt;
}

Stmt GpuIslEmitter::InsertRealizeWithMemType(Stmt stmt, const isl::id &var, std::string mem) {
  stmt = FindInnerRealize(var.get_name()).Mutate(stmt);

  Tensor t = info_.FindTensorWithLargestShape(var);
  Region bounds;

  // no isolate
  if (bounds.empty()) {
    for (auto j : t->shape) {
      bounds.push_back(Range::make_by_min_extent(Expr(0), j));
    }
  }

  // If isolate, make a new buffer
  auto buf = info_.user_config_.GetBind().at(t);

  auto tt = placeholder(t->shape, t->dtype, t->op->name);

  stmt = TensorSubstitute(stmt, t->op, tt->op, tt->value_index);
  t = tt;
  if (info_.analysis_result_.CountBufferDefInfo(var)) {
    auto decl = info_.analysis_result_.GetBufferDefInfo(var);
    decl.tensor = t;
  }
  info_.user_config_.SetBind(t, buf);
  stmt = TensorSubstitute2(stmt, t->op->func_name(), t->op, t->value_index);
  stmt = Realize::make(t->op, t->value_index, t->dtype, bounds, const_true(1), stmt);
  realized_.insert(t);
  stmt = AttrStmt::make(t->op, air::ir::attr::realize_scope, Expr(mem), stmt);

  return stmt;
}

Expr GpuIslEmitter::IterNameAdaptor(std::string name) {
  if (iter_name_map_.find(name) != iter_name_map_.end()) {
    return iter_name_map_[name];
  } else if (name.find(REPLACE) != std::string::npos) {
    name = name.substr(strlen(REPLACE));
    if (info_.user_config_.GetEnableTileC0()) {
      return SingleConfigToMultiBand(name);
    }
    return AdaptPolyNewVar(name);
  } else {
    return VarExpr(name);
  }
}

Expr GpuIslEmitter::SingleConfigToMultiBand(std::string name) {
  Expr e;
  VarExpr original_id;
  int rep_size = 1;
  auto l0_block_size = info_.user_config_.GetC0BlockSize();
  if (name.find(B0) != std::string::npos) {
    original_id = iter_name_map_[B0];
    rep_size = l0_block_size[0];
  } else if (name.find(B1) != std::string::npos) {
    original_id = iter_name_map_[B1];
    rep_size = l0_block_size[1];
  } else {
    original_id = iter_name_map_[B2];
    rep_size = l0_block_size[2];
  }

  if (rep_size < 0) {
    return e;
  }

  if (name.find(TILE_WITH_C0) != std::string::npos) {
    e = Mod::make(original_id, rep_size);
  } else if (name.find(TILE_WITH_C1) != std::string::npos) {
    e = Div::make(original_id, rep_size);
  } else {
    LOG(FATAL) << "Unexpected binding id: " << name;
  }
  return e;
}

// if new var is added in poly process, modify the logic here.
// another modify pos is IterNameAdaptor interface
Expr GpuIslEmitter::AdaptPolyNewVar(std::string name) {
  Expr e;
  std::string t0_string = T0;
  int suffix_len = t0_string.size() + 1;
  auto tensor_name = name.substr(0, name.size() - suffix_len);
  if (!info_.user_config_.GetReplaceConfig().count(tensor_name)) {
    return e;
  }
  auto mapping_cfg = (info_.user_config_.GetReplaceConfig()[tensor_name]);
  CHECK(mapping_cfg) << "mapping config is null.";
  int mx = mapping_cfg->GetX().second;
  int my = mapping_cfg->GetY().second;
  int mz = mapping_cfg->GetZ().second;
  if (name.find(WARP_COMPUTE) != std::string::npos) {
    if (name.find(T0) != std::string::npos) {
      e = Div::make(iter_name_map_[T0], WARP_SIZE);
      e = Mod::make(e, mx);
      return e;
    } else if (name.find(T1) != std::string::npos) {
      e = Div::make(iter_name_map_[T0], WARP_SIZE);
      e = Div::make(e, mx);
      return e;
    }
  } else {
    if (name.find(T0) != std::string::npos) {
      e = Mod::make(iter_name_map_[T0], mx);
      return e;
    } else if (name.find(T1) != std::string::npos) {
      e = Div::make(iter_name_map_[T0], mx);
      if (mz == 1) {
        return e;
      }
      e = Mod::make(e, my);
      return e;
    } else if (name.find(T2) != std::string::npos) {
      e = Div::make(iter_name_map_[T0], mx);
      e = Div::make(e, my);
      return e;
    }
  }
  return e;
}

Expr GpuIslEmitter::Interpret(const isl::ast_expr &e) {
  if (auto int_expr = e.as<isl::ast_expr_int>()) {
    return Expr(IslExprToSInt(int_expr));
  } else if (auto id_expr = e.as<isl::ast_expr_id>()) {
    // If this variable is defined by loop index, we need sharing it.
    const Variable *var = GetIterByName(id_expr.get_id().get_name());
    if (var) {
      if (stride_modify_iter_map_.find(var) != stride_modify_iter_map_.end()) {
        return stride_modify_iter_map_[var];
      }
      return VarExpr(GetObjPtr(var));
    } else {
      return IterNameAdaptor(id_expr.get_id().to_str());
    }
  } else if (auto op_expr = e.as<isl::ast_expr_op>()) {
    return InterpretOp(op_expr);
  } else {
    LOG(FATAL) << "NYI " << e;
    return 0;
  }
}

Stmt GpuIslEmitter::EmitAccessNodeFromPromoteAcsCall(isl::id var, const Node *node, Array<Expr> &args) {
  const Call *call = static_cast<const Call *>(node);
  Tensor t = info_.FindTensor(var);
  return Evaluate::make(Call::make(call->type, var.get_name(), args, call->call_type, t->op, t->value_index));
}

Stmt GpuIslEmitter::EmitAccessNodeFromPromoteAcsProvide(isl::id var, const Node *node, Array<Expr> &args) {
  const auto provide = static_cast<const Provide *>(node);
  Tensor t = info_.FindTensor(var);
  Stmt s = Provide::make(t->op, 0, provide->value, args);
  return s;
}

void GetNameWithoutShared(isl::id &tensor_id, ScopInfo &info) {
  if (info.user_config_.GetEnableMatmul()) {
    size_t pos = tensor_id.get_name().find(SHARE_SUFFIX);
    std::string substr = tensor_id.get_name().substr(0, pos);
    if (pos != 0) tensor_id = isl::id(tensor_id.ctx(), substr);
  }
}

isl::multi_aff GpuIslEmitter::TensorAccessMultAff(isl::id &tensor_id, const Array<Expr> &tensor_index,
                                                  const isl::id &node_id) {
  GetNameWithoutShared(tensor_id, info_);
  return IslEmitter::TensorAccessMultAff(tensor_id, tensor_index, node_id);
}

Array<Expr> EmitTensorCoreHelper::GetTileSize(const std::string &name) {
  auto it = tensor_core_info_.matrix_abc_.find(name);
  auto it2 = tensor_core_info_.matrix_major_.find(name);
  CHECK(it != tensor_core_info_.matrix_abc_.end() && it2 != tensor_core_info_.matrix_major_.end())
    << "Cannot find matrix info for " << name;
  Expr size0 = make_const(Int(32), 16);
  Expr size1 = make_const(Int(32), 16);
  if (it->second == MMA_A && it2->second == COL_MAJOR) {
    size0 = make_const(Int(32), tensor_core_info_.warp_tile_.k);
    size1 = make_const(Int(32), tensor_core_info_.warp_tile_.m);
  }
  if (it->second == MMA_A && it2->second == ROW_MAJOR) {
    size0 = make_const(Int(32), tensor_core_info_.warp_tile_.m);
    size1 = make_const(Int(32), tensor_core_info_.warp_tile_.k);
  }
  if (it->second == MMA_B && it2->second == ROW_MAJOR) {
    size0 = make_const(Int(32), tensor_core_info_.warp_tile_.k);
    size1 = make_const(Int(32), tensor_core_info_.warp_tile_.n);
  }
  if (it->second == MMA_B && it2->second == COL_MAJOR) {
    size0 = make_const(Int(32), tensor_core_info_.warp_tile_.n);
    size1 = make_const(Int(32), tensor_core_info_.warp_tile_.k);
  }

  if (it->second == MATRIX_C) {
    size0 = make_const(Int(32), tensor_core_info_.warp_tile_.m);
    size1 = make_const(Int(32), tensor_core_info_.warp_tile_.n);
  }
  Array<Expr> tile_size = {size0, size1};
  return tile_size;
}

void EmitTensorCoreHelper::SetDataForLoad(Expr src, Expr stride, Expr major, const Call *call, const Provide *op,
                                          NodePtr<BufferNode> &node) {
  data_for_load_.src = src;
  data_for_load_.stride = stride;
  data_for_load_.major = major;
  data_for_load_.call = call;
  data_for_load_.op = op;
  data_for_load_.node = node;
}
void EmitTensorCoreHelper::SetDataForStore(Expr dst, Expr stride, const Call *call, NodePtr<BufferNode> &node) {
  data_for_store_.dst = dst;
  data_for_store_.stride = stride;
  data_for_store_.call = call;
  data_for_store_.node = node;
}
void EmitTensorCoreHelper::SetDataForFill(const Provide *op, const Call *call, NodePtr<BufferNode> &node) {
  data_for_fill_.call = call;
  data_for_fill_.op = op;
  data_for_fill_.node = node;
}
void EmitTensorCoreHelper::SetDataForSync(Expr a, Expr b, Expr c, NodePtr<BufferNode> &node_a,
                                          NodePtr<BufferNode> &node_b, NodePtr<BufferNode> &node_c) {
  data_for_sync_.a = a;
  data_for_sync_.b = b;
  data_for_sync_.c = c;
  data_for_sync_.node_a = node_a;
  data_for_sync_.node_b = node_b;
  data_for_sync_.node_c = node_c;
}

void EmitTensorCoreHelper::PrepareDataCore() {
  auto it = tensor_core_info_.bounds_.find(key_);
  CHECK(it != tensor_core_info_.bounds_.end());
  Array<Expr> min_bound;
  for (auto i : it->second) {
    min_bound.push_back(i->min);
  }

  CHECK_GE(it->second.size(), 2);
  Array<Expr> shape;
  for (size_t i = 0; i < it->second.size() - 2; ++i) {
    shape.push_back(it->second[i]->extent);
  }
  auto tile_size = GetTileSize(SimplifyName(call_->name));
  shape.push_back(tile_size[0]);
  shape.push_back(tile_size[1]);

  tensor_core_info_.min_bounds_[call_->name] = min_bound;

  Array<Expr> strides;
  for (size_t i = 1; i < shape.size(); ++i) {
    Expr stride = IntImm::make(Int(32), 1);
    for (size_t j = shape.size() - 1; j >= i; --j) {
      stride = Mul::make(stride, shape[j]);
    }
    strides.push_back(stride);
  }
  strides.push_back(make_const(Int(32), 1));

  Expr elem_offset = IntImm::make(Int(32), 0);
  CHECK_EQ(call_->args.size(), min_bound.size());
  for (size_t i = 0; i < min_bound.size(); i++) {
    auto arg = call_->args[i];
    arg = DeleteThreadIdx().Mutate(arg);
    arg = Simplify(arg);
    elem_offset = Add::make(elem_offset, Mul::make(strides[i], Sub::make(arg, min_bound[i])));
  }

  auto it2 = tensor_core_info_.matrix_abc_.find(SimplifyName(call_->name));
  CHECK(it2 != tensor_core_info_.matrix_abc_.end()) << "Cannot find matrix info for " << call_->name;
  buffer_node_->data = Variable::make(Handle(), call_->name);
  buffer_node_->name = call_->name;
  std::string name = it2->second;
  if (name == MATRIX_C) {
    name = MMA_C;
  }
  buffer_node_->scope = "wmma." + name;
  buffer_node_->dtype = data_type_;
  buffer_node_->strides = strides;
  buffer_node_->shape = shape;
  buffer_node_->data_alignment = 1;
  buffer_node_->elem_offset = Simplify(elem_offset);
  buffer_node_->offset_factor = 1;
  Buffer buffer(buffer_node_);

  NodePtr<TensorNode> tensor_node = make_node<TensorNode>();
  tensor_node->value_index = key_.value_index;
  tensor_node->op = Downcast<Operation>(key_.f);
  tensor_node->shape = shape;
  tensor_node->dtype = data_type_;
  Tensor tensor(tensor_node);

  Array<Expr> args;
  for (size_t i = 0; i < call_->args.size(); ++i) {
    auto arg = call_->args[i];
    arg = DeleteThreadIdx().Mutate(arg);
    arg = Simplify(arg);

    args.push_back(arg);
    args.push_back(shape[i]);
  }
  tuple_ = Call::make(Handle(), air::ir::intrinsic::tvm_tuple, args, Call::Intrinsic);
  node_ = {buffer, tensor};
}

Stmt EmitTensorCoreHelper::MakeLoadTransform() {
  key_ = air::ir::TensorKey{data_for_load_.op->func, data_for_load_.op->value_index};
  call_ = data_for_load_.call;
  buffer_node_ = data_for_load_.node;
  data_type_ = call_->type;

  PrepareDataCore();
  Buffer buffer = Downcast<Buffer>(node_[0]);
  Stmt stmt = Evaluate::make(Call::make(
    Handle(), air::ir::intrinsic::tvm_load_matrix_sync,
    {buffer->data, tensor_core_info_.warp_tile_.m, tensor_core_info_.warp_tile_.n, tensor_core_info_.warp_tile_.k,
     Simplify(buffer->elem_offset), data_for_load_.src, data_for_load_.stride, data_for_load_.major},
    Call::Intrinsic));
  return AttrStmt::make(node_, "buffer_bind_scope", tuple_, stmt);
}

Stmt EmitTensorCoreHelper::MakeStoreTransform() {
  key_ = air::ir::TensorKey{data_for_store_.call->func, data_for_store_.call->value_index};
  call_ = data_for_store_.call;
  buffer_node_ = data_for_store_.node;
  data_type_ = call_->type;

  PrepareDataCore();
  Buffer buffer = Downcast<Buffer>(node_[0]);
  Stmt stmt = Evaluate::make(Call::make(
    Handle(), air::ir::intrinsic::tvm_store_matrix_sync,
    {buffer->data, tensor_core_info_.warp_tile_.m, tensor_core_info_.warp_tile_.n, tensor_core_info_.warp_tile_.k,
     buffer->elem_offset, data_for_store_.dst, data_for_store_.stride, StringImm::make(ROW_MAJOR)},
    Call::Intrinsic));
  return AttrStmt::make(node_, "buffer_bind_scope", tuple_, stmt);
}

Stmt EmitTensorCoreHelper::MakeFillTransform() {
  key_ = air::ir::TensorKey{data_for_fill_.call->func, data_for_fill_.call->value_index};
  call_ = data_for_fill_.call;
  buffer_node_ = data_for_fill_.node;
  data_type_ = call_->type;

  PrepareDataCore();
  Buffer buffer = Downcast<Buffer>(node_[0]);
  Stmt stmt = Evaluate::make(Call::make(Handle(), air::ir::intrinsic::tvm_fill_fragment,
                                        {buffer->data, tensor_core_info_.warp_tile_.m, tensor_core_info_.warp_tile_.n,
                                         tensor_core_info_.warp_tile_.k, buffer->elem_offset, data_for_fill_.op->value},
                                        Call::Intrinsic));
  return AttrStmt::make(node_, "buffer_bind_scope", tuple_, stmt);
}

Stmt EmitTensorCoreHelper::MakeSyncTransform() {
  bool is_cast = false;
  if (data_for_sync_.a.as<Call>()) {
    auto call_a = data_for_sync_.a.as<Call>();
    key_ = air::ir::TensorKey{call_a->func, call_a->value_index};
    call_ = call_a;
    buffer_node_ = data_for_sync_.node_a;
    data_type_ = call_->type;
    is_cast = true;
  } else if (data_for_sync_.a.as<Cast>()) {
    auto cast_a = data_for_sync_.a.as<Cast>();
    auto call_a = cast_a->value.as<Call>();
    CHECK(call_a);
    key_ = air::ir::TensorKey{call_a->func, call_a->value_index};
    call_ = call_a;
    buffer_node_ = data_for_sync_.node_a;
    data_type_ = call_->type;
    is_cast = true;
  }

  PrepareDataCore();

  auto tuple_a = tuple_;
  auto node_a = node_;

  if (data_for_sync_.b.as<Call>()) {
    auto call_b = data_for_sync_.b.as<Call>();
    key_ = air::ir::TensorKey{call_b->func, call_b->value_index};
    call_ = call_b;
    buffer_node_ = data_for_sync_.node_b;
    data_type_ = call_->type;
    is_cast = true;
  } else if (data_for_sync_.b.as<Cast>()) {
    auto cast_b = data_for_sync_.b.as<Cast>();
    auto call_b = cast_b->value.as<Call>();
    CHECK(call_b);
    key_ = air::ir::TensorKey{call_b->func, call_b->value_index};
    call_ = call_b;
    buffer_node_ = data_for_sync_.node_b;
    data_type_ = call_->type;
    is_cast = true;
  }

  PrepareDataCore();

  auto tuple_b = tuple_;
  auto node_b = node_;

  auto call_c = data_for_sync_.c.as<Call>();
  CHECK(call_c);
  key_ = air::ir::TensorKey{call_c->func, call_c->value_index};
  call_ = call_c;
  buffer_node_ = data_for_sync_.node_c;
  data_type_ = call_->type;

  PrepareDataCore();

  auto tuple_c = tuple_;
  auto node_c = node_;

  Buffer buffer_a(data_for_sync_.node_a);
  Buffer buffer_b(data_for_sync_.node_b);
  Buffer buffer = Downcast<Buffer>(node_c[0]);

  Stmt stmt = Evaluate::make(Call::make(Handle(), air::ir::intrinsic::tvm_mma_sync,
                                        {buffer->data, buffer->elem_offset, buffer_a->data, buffer_a->elem_offset,
                                         buffer_b->data, buffer_b->elem_offset, buffer->data, buffer->elem_offset},
                                        Call::Intrinsic));

  stmt = AttrStmt::make(node_c, "buffer_bind_scope", tuple_c, stmt);
  stmt = AttrStmt::make(node_b, "buffer_bind_scope", tuple_b, stmt);
  stmt = AttrStmt::make(node_a, "buffer_bind_scope", tuple_a, stmt);

  std::string cast_mode = CAST_MODE_1;
  if (is_cast) {
    stmt = AttrStmt::make(Expr("INFO"), CAST_FLAG, StringImm::make(cast_mode), stmt);
  }

  return stmt;
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
