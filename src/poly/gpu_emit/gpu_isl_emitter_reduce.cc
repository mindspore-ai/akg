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

#include "emit_pass.h"
#include "gpu_isl_emitter_reduce.h"

namespace akg {
namespace ir {
namespace poly {

Stmt GpuIslEmitterReduce::Emit(const isl::ast_node &node) {
  Stmt stmt = GpuIslEmitter::Emit(node);

  stmt = EmitForReduce(stmt, info_);

  return stmt;
}

Stmt GpuIslEmitterReduce::EmitMark(const isl::ast_node_mark &node) {
  std::string mark = node.get_id().get_name();
  if (IsStartsWith(mark, REDUCE_ATOMIC_FLAG) || mark == REDUCE_AREA_FLAG) {
    Stmt stmt = EmitAst(node.get_node());
    if (!stmt.defined()) {
      return Stmt();
    }
    return AttrStmt::make(Expr("INFO"), mark, StringImm::make(mark), stmt);
  }
  return GpuIslEmitter::EmitMark(node);
}

Stmt GpuIslEmitterReduce::EmitStmt(const isl::ast_node_user &node) {
  CHECK(node.get_expr().isa<isl::ast_expr_op>());
  isl::ast_expr_op usr_expr = node.get_expr().as<isl::ast_expr_op>();
  CHECK(usr_expr);
  auto stmt_id = usr_expr.get_arg(0).as<isl::ast_expr_id>().get_id();
  auto node_id = node.get_annotation();

  if (info_.IsWrite(stmt_id)) {
    if (info_.IsGMWrite(stmt_id)) {
      auto iterator_map = node_info_map_.at(node_id).iterator_map;
      auto original = iterator_map.range_factor_domain().range_factor_range();
      auto srcid = original.get_tuple_id(isl_dim_out);
      bool no_need_to_emit = GpuIslEmitter::NoNeedToEmitForTempTensor(srcid);
      if (no_need_to_emit) return Stmt();
    }
  } else if (info_.IsReduceInit(stmt_id) || info_.IsReduceUpdate(stmt_id)) {
    return EmitFilter(stmt_id.get_name());
  }
  return GpuIslEmitter::EmitStmt(node);
}

Stmt GpuIslEmitterReduce::EmitFilter(std::string name) {
  return Evaluate::make(Call::make(Int(32), name, {}, Call::Extern));
}

Stmt GpuIslEmitterReduce::EmitUserStmt(const isl::ast_node_user &node) {
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

  bool init_stmt_emit = false;
  auto ids = info_.analysis_result_.GetReduceInitIds();
  for (auto &i : ids) {
    if (i.get_name() == stmt_id_.get_name()) {
      init_stmt_emit = true;
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

  Stmt stmt = EmitUserStmtContent(stmt_node);

  if (init_stmt_emit) {
    stmt = AttrStmt::make(Expr("INFO"), REDUCE_INIT_FLAG, StringImm::make(""), stmt);
  }
  return stmt;
}

}  // namespace poly
}  // namespace ir
}  // namespace akg