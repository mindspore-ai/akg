/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "gpu_isl_emitter.h"
#include "emit_pass.h"
#include "ir_pass.h"
#include <sstream>
#include <algorithm>

namespace akg {
namespace ir {
namespace poly {

Expr GpuIslEmitter::EmitLoad(const isl::ast_expr &expr, const Type type) {
  if (PRINT_EMITTER) {
    LOG(INFO) << ">>>>>>>>>>>>INPUT AST_NODE[LOAD]<<<<<<<<<<<<<<\n" << expr;
  }
  auto value = IslEmitter::EmitLoad(expr, type);
  if (PRINT_EMITTER) {
    LOG(INFO) << ">>>>>>>>>>>>OUTPUT STMT<<<<<<<<<<<<\n" << value;
  }
  return value;
}

Stmt GpuIslEmitter::EmitSync() {
  return Evaluate::make(Call::make(Int(32), STORAGE_SYNC, {StringImm::make(SYNC_SCOP_SHARED)}, Call::Intrinsic));
}

Stmt GpuIslEmitter::EmitStmt(const isl::ast_node_user &node) {
  CHECK(node.get_expr().isa<isl::ast_expr_op>());
  isl::ast_expr_op usr_expr = node.get_expr().as<isl::ast_expr_op>();
  CHECK(usr_expr);
  auto stmt_id = usr_expr.get_arg(0).as<isl::ast_expr_id>().get_id();
  auto node_id = node.get_annotation();

  if (info_.IsRead(stmt_id)) {
    Stmt s;
    s = EmitRead(node);
    s = AttrStmt::make(Expr(""), GMREAD_FLAG, StringImm::make(GMREAD_FLAG), s);
    return s;
  } else if (info_.IsWrite(stmt_id)) {
    if (info_.IsGMWrite(stmt_id) || info_.IsGMLWrite(stmt_id)) {
      auto iterator_map = node_info_map_.at(node_id).iterator_map;
      auto original = iterator_map.range_factor_domain().range_factor_range();
      auto srcid = original.get_tuple_id(isl_dim_out);
      bool no_need_to_emit = GpuIslEmitter::NoNeedToEmitForTempTensor(srcid);
      if (no_need_to_emit) return Stmt();
    }
    return EmitWrite(node);
  } else if (info_.IsSync(stmt_id)) {
    return EmitSync();
  } else {
    Stmt stmt = EmitUserStmt(node);
    auto tot = info_.analysis_result_.GetTensorOfTensorStmt();
    auto id_name = stmt_id.get_name();
    if (tot.count(id_name)) {
      std::string marker_name = ATOMIC_MARKER;
      marker_name += "_";
      marker_name += tot[id_name];
      stmt = AttrStmt::make(Expr("INFO"), marker_name, StringImm::make(marker_name), stmt);
    }
    return stmt;
  }
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
        if ((stmt_num != 1) && (stmt_num - last_num != 1)) {
          for (int index = stmt_num - 2 - last_num; index >= 0; --index) {
            auto p_index = static_cast<unsigned int>(index);
            stmts[p_index] = Block::make(stmts[p_index], stmts[p_index + 1]);
          }
        }
        stmts[0] = InsertRealize(stmts[0], new_stmt_id);
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

  if (last_num != len - 1) {
    for (int index = len - 2 - last_num; index >= 0; --index) {
      auto p_index = static_cast<unsigned int>(index);
      stmts[p_index] = Block::make(stmts[p_index], stmts[p_index + 1]);
    }
  }
  return stmts[0];
}

Stmt GpuIslEmitter::EmitFor(const isl::ast_node_for &node) {
  ForType for_type = for_type_;
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

  Stmt body_stmt = EmitAst(node.get_body());

  if (!body_stmt.defined()) {
    PopIter(iter_expr.get());
    return Stmt();
  }

  if (need_to_modify_inc_) {
    stride_modify_iter_map_.erase(iter_expr.get());
  }
  PopIter(iter_expr.get());
  Stmt stmt = For::make(iter_expr, init_expr, cond_expr, for_type, DeviceAPI::None, body_stmt);
  return stmt;
}

Stmt GpuIslEmitter::EmitIf(const isl::ast_node_if &node) {
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

  Stmt s;
  if (!cond_expr.defined()) {
    s = then_case;
  } else {
    s = IfThenElse::make(cond_expr, then_case, else_case);
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

Stmt GpuIslEmitter::EmitTensorOfTensorStmt(const Stmt &s) {
  Stmt stmt = LowerWith(s);
  stmt = AtomicReturnStmtEmit(info_).Mutate(stmt);
  stmt = AttrStmt::make(Expr("INFO"), REDUCE_LIB_TYPE_FLAG, info_.user_config_.GetReduceLibType(), stmt);
  return stmt;
}

void GpuIslEmitter::UpdateGpuIndexDtype() {
  auto read_map = info_.StmtReadMap();
  auto write_map = info_.StmtWriteMap();
  std::set<std::string> id_sets;
  for (auto item : read_map) {
    for (auto item_id : item.second) {
      if (id_sets.count(item_id.get_name()) == 0) {
        id_sets.insert(item_id.get_name());
      }
    }
  }
  for (auto item : write_map) {
    for (auto item_id : item.second) {
      if (id_sets.count(item_id.get_name()) == 0) {
        id_sets.insert(item_id.get_name());
      }
    }
  }

  bool use_int64_idx_gpu = false;
  for (auto tensor_name : id_sets) {
    auto tensor_shape = info_.GetShapeOf(tensor_name);
    int64_t tensor_size = 1;
    for (int v : tensor_shape) {
      tensor_size *= (int64_t)v;
    }
    if (tensor_size >= INT_MAX) {
      use_int64_idx_gpu = true;
      break;
    }
  }

  if (use_int64_idx_gpu) {
    iter_name_map_ = {{B0, VarExpr(BLOCK_IDX_X, Int(64))},  {B1, VarExpr(BLOCK_IDX_Y, Int(64))},
                      {B2, VarExpr(BLOCK_IDX_Z, Int(64))},  {T0, VarExpr(THREAD_IDX_X, Int(64))},
                      {T1, VarExpr(THREAD_IDX_Y, Int(64))}, {T2, VarExpr(THREAD_IDX_Z, Int(64))}};
  }
}

class InitStmtInsertSync : public IRMutator {
  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    if (op->value.as<IntImm>() != nullptr) {
      scop_init_ = true;
    }
    return s;
  }

  Stmt Mutate_(const IfThenElse *op, const Stmt &s) {
    auto stmt = IRMutator::Mutate_(op, s);
    if (scop_init_) {
      scop_init_ = false;
      return Block::make(stmt, Evaluate::make(
        Call::make(Int(int_bit_count_), "tvm_storage_sync", {StringImm::make("shared")}, Call::Intrinsic)));
    }
    return stmt;
  }

  bool scop_init_{false};
  static constexpr int int_bit_count_{32};
};

Stmt GpuIslEmitter::Emit(const isl::ast_node &node) {
  UpdateGpuIndexDtype();

  Stmt stmt = EmitAst(node);

  // emit realize for temporary tensor
  stmt = EmitRealizeForGlobalTensor(stmt);

  if (!info_.analysis_result_.GetTensorOfTensorStmt().empty()) {
    stmt = EmitTensorOfTensorStmt(stmt);
  }

  if (info_.analysis_result_.GetOpTemplate() == Template::COUNT_OP) {
    stmt = InitStmtInsertSync().Mutate(stmt);
  }

  // iter var node attr emit
  std::map<std::string, VarExpr>::iterator it;
  for (it = iter_name_map_.begin(); it != iter_name_map_.end(); ++it) {
    IterVar axis = IterVarNode::make(Range(), it->second, air::kThreadIndex, it->second->name_hint);
    stmt = AttrStmt::make(axis, air::ir::attr::thread_extent, Expr(GetThreadExtent(it->second->name_hint)), stmt);
  }

  // attr for one dimension mapping
  if (info_.user_config_.GetEnableOneDimThread()) {
    auto thread_cfg = info_.user_config_.GetThreadConfig();
    CHECK(thread_cfg) << "thread config is null.";
    int tx = thread_cfg->GetX().second;
    int warp_number = tx;

    if (info_.user_config_.GetReplaceConfig().count(WARP_COMPUTE) != 0) {
      int ty = thread_cfg->GetY().second;
      int tz = thread_cfg->GetZ().second;
      warp_number = (tx * ty * tz) / 32;
    }
    stmt = AttrStmt::make(Expr(""), ORIGIN_THREAD_DIM_X, Expr(warp_number), stmt);
  }

  if (info_.user_config_.GetMindTrickWasUsed() && info_.user_config_.GetMindTrickGpuHasSwizzle()) {
    stmt = AttrStmt::make(make_zero(Int(32)), MIND_TRICKS_SWIZZLE_PRAGMA, Expr(1), stmt);
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

    // if the tensor is temporary and it is not promoted, it needs to emit realize
    stmt = InsertRealize(stmt, isl::id(info_.GetCtx(), name));
  }
  return stmt;
}

Stmt GpuIslEmitter::EmitMark(const isl::ast_node_mark &node) {
  std::string mark = node.get_id().get_name();

  Stmt stmt;

  if ((mark == FOR_VECTORIZED) || (mark == PROMOTE_REGISTER_TO_GLOBAL) || (mark == PROMOTE_REGISTER_TO_SHARED) ||
      (mark == PROMOTE_SHARED_TO_GLOBAL) || (mark == SHARED_MEM_PROMOTED_COMPLETE) ||
      IsStartsWith(mark, REDUCE_ATOMIC_FLAG)) {
    bool is_specific_for = (AkgSupportedForType.find(mark) != AkgSupportedForType.end());
    if (is_specific_for) {
      for_type_ = AkgSupportedForType.at(mark);
    }
    stmt = EmitAst(node.get_node());
    for_type_ = ForType::Serial;
    if (!stmt.defined()) {
      return Stmt();
    }
    if (!is_specific_for) {
      stmt = AttrStmt::make(Expr("INFO"), mark, StringImm::make(mark), stmt);
    }
  } else {
    stmt = EmitAst(node.get_node());
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

Stmt GpuIslEmitter::SubstituteTensorStmt(const Stmt &s, Tensor origin, Tensor replaced) {
  auto stmt = TensorSubstitute(s, origin->op, replaced->op, replaced->value_index);
  stmt = TensorStringSubstitute(stmt, replaced->op->func_name(), replaced->op, replaced->value_index);
  return stmt;
}

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

  stmt = SubstituteTensorStmt(stmt, t, tt);
  t = tt;
  if (info_.analysis_result_.CountBufferDefInfo(var)) {
    auto decl = info_.analysis_result_.GetBufferDefInfo(var);
    decl.tensor = t;
  }
  info_.user_config_.SetBind(t, buf);
  stmt = Realize::make(t->op, t->value_index, t->dtype, bounds, const_true(1), stmt);
  stmt = AttrStmt::make(t->op, air::ir::attr::realize_scope, FindRealizeScope(var), stmt);

  return stmt;
}

Expr GpuIslEmitter::IterNameAdaptor(std::string name) {
  if (iter_name_map_.find(name) != iter_name_map_.end()) {
    return iter_name_map_[name];
  } else if (name.find(REPLACE) != std::string::npos) {
    name = name.substr(strlen(REPLACE));
    return AdaptPolyNewVar(name);
  } else {
    return VarExpr(name);
  }
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
  if (mapping_cfg->type == MappingType::REPLACE_THREADS) {
    e = AdaptThreadNewVar(name, mapping_cfg);
  } else {
    e = AdaptOneConfigForMulAxis(mapping_cfg, name, false);
  }
  CHECK(e.defined()) << "new var is null";
  return e;
}

Expr GpuIslEmitter::AdaptThreadNewVar(const std::string &name, MappingCfg *mapping_cfg) {
  Expr e;
  int mx = mapping_cfg->GetX().second;
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
    e = AdaptOneConfigForMulAxis(mapping_cfg, name, true);
  }
  return e;
}

Expr GpuIslEmitter::AdaptOneConfigForMulAxis(MappingCfg *mapping_cfg, const std::string &orig_name,
                                             const bool is_thread) {
  std::string config_name = T0;
  std::string repeated_name = REPEATED_MAPPING;
  std::string name = orig_name;
  if (name.find(repeated_name) != std::string::npos) {
    config_name = name.substr(repeated_name.size(), config_name.size());
    int suffix_len = repeated_name.size() + config_name.size();
    name = name.substr(suffix_len + 1, name.size() - suffix_len);
  }

  Expr e;
  for (size_t i = 0; i < mapping_cfg->bound; ++i) {
    std::string config_id_name = is_thread ? THREAD_STR : BLOCK_STR;
    config_id_name += std::to_string(i);
    if (name.find(config_id_name) == std::string::npos) {
      continue;
    }

    e = iter_name_map_[config_name];
    int config_id_number = mapping_cfg->GetAt(i).second;

    if (i == 0) {
      e = Mod::make(e, config_id_number);
      return e;
    }

    for (size_t j = 0; j < i; ++j) {
      config_id_number = mapping_cfg->GetAt(j).second;
      e = Div::make(e, config_id_number);
    }

    config_id_number = mapping_cfg->GetAt(i).second;
    e = Mod::make(e, config_id_number);
    return e;
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

Stmt AtomicReturnStmtEmit::Mutate_(const AttrStmt *op, const Stmt &s) {
  auto key = op->attr_key;
  if (IsStartsWith(key, REDUCE_ATOMIC_FLAG)) {
    in_atomic_area_ = true;
    std::vector<std::string> strs = common::Split(key, "_");
    CHECK_EQ(strs.size(), REDUCE_ATOMIC_FLAG_SIZE) << "atomic mark format is not right!.";
    atomic_data_.reduce_op.clear();
    if (AkgSupportedReduceOp.count(strs[REDUCE_ATOMIC_FLAG_TYPE_POS])) {
      atomic_data_.reduce_op = AKG_REDUCE_LIB_SPACE;
      atomic_data_.reduce_op += "::";
      atomic_data_.reduce_op += strs[REDUCE_ATOMIC_FLAG_TYPE_POS];
    } else {
      CHECK(false) << "reduce op type is not supported!";
    }
  }
  return IRMutator::Mutate_(op, s);
}

Stmt AtomicReturnStmtEmit::Mutate_(const Provide *op, const Stmt &s) {
  if (in_atomic_area_) {
    in_atomic_area_ = false;
    Stmt stmt = IRMutator::Mutate_(op, s);
    atomic_data_.gm_write_stmt = stmt;
    auto op = stmt.as<Provide>();
    CHECK(op);
    auto value = op->value;
    auto value_call = value.as<Call>();
    auto value_add = value.as<Add>();
    if (value_call) {
      atomic_data_.atomic_rhs = op->value;
    }
    if (value_add) {
      auto a = value_add->a.as<Call>();
      auto b = value_add->b.as<Call>();
      if (a && a->name == op->func->func_name()) {
        atomic_data_.atomic_rhs = value_add->b;
      } else if (b && b->name == op->func->func_name()) {
        atomic_data_.atomic_rhs = value_add->a;
      } else {
        CHECK(false) << "no support atomic return type";
      }
    }
    if (!atomic_data_.atomic_rhs.defined()) {
      CHECK(ContainsHalideCall(op->args)) << "atomic_data_.atomic_rhs_ is not defined";
      atomic_data_.atomic_rhs = value;
    }
    atomic_data_.output_tensor_data_type_info = scop_info_.GetDtypeOf(op->func->func_name());

    ConstructAtomicReturnFuncName(scop_info_.user_config_.GetReduceLibType(), atomic_data_.reduce_op,
                                  atomic_data_.akg_atomic_api, atomic_data_.akg_atomic_template_arg);
    return MakeAtomicStmt(atomic_data_);
  }
  return IRMutator::Mutate_(op, s);
}
}  // namespace poly
}  // namespace ir
}  // namespace akg
