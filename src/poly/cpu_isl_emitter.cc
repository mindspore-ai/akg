/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "poly/cpu_isl_emitter.h"
#include <regex>

namespace akg {
namespace ir {
namespace poly {

static constexpr auto REDUCE_FUNCTION = "Reduce";
static constexpr auto PACK_A_ATTR = "pack_a";
static constexpr auto PACK_B_ATTR = "pack_b";
static constexpr auto MATRIX_TRANSPOSE_FUNCTION = "MatrixTranspose";
static constexpr auto LOCAL_MEMORY = "local";
static std::unordered_map<std::string, std::function<Expr(Expr, Expr)>> maker_map = {
  {"SumOp", Add::make}, {"AddOp", Add::make}, {"SubOp", Sub::make}, {"MulOp", Mul::make}, {"DivOp", Div::make},
  {"MinOp", Min::make}, {"MaxOp", Max::make}, {"AndOp", And::make}, {"OrOp", Or::make}};

Stmt CpuIslEmitter::EmitterPostProcess(Stmt &stmt) {
  stmt = EmitRealizeForGlobalTensor(stmt);
  stmt = EmitInfo(stmt);
  return IslEmitter::EmitterPostProcess(stmt);
}

Stmt CpuIslEmitter::EmitInfo(const Stmt &stmt) {
  Stmt result = stmt;

  if (info_.analysis_result_.GetOpTemplate() == Template::REDUCTION || info_.analysis_result_.GetCsr()) {
    auto len = info_.analysis_result_.GetVectorizedLoopSize();
    result = AttrStmt::make(Expr("INFO"), "vector_length", Expr(len), result);
    auto vectorized_bits = info_.analysis_result_.GetVectorizedLength();
    result = AttrStmt::make(Expr("INFO"), "vectorized_bits", Expr(vectorized_bits), result);
  }

  if (info_.analysis_result_.GetOpTemplate() == Template::MATMUL) {
    PackBlockSize pack_block_size = info_.analysis_result_.GetPackBlockSize();
    result = AttrStmt::make(Expr("INFO"), PACK_A_ATTR, Expr(pack_block_size.pack_a_size), result);
    result = AttrStmt::make(Expr("INFO"), PACK_B_ATTR, Expr(pack_block_size.pack_b_size), result);
    if (!info_.user_config_.NeedPackMatrixB()) {
      auto tensor_name = info_.analysis_result_.GetMatrixName(MATRIX_B);
      auto major_map = info_.analysis_result_.GetMatrixMatmulMajor();
      std::string major;
      if (major_map.count(tensor_name) != 0) {
        major = major_map[tensor_name];
      }
      result = AttrStmt::make(Expr("INFO"), PREPARE_PACK, Expr(tensor_name + "_" + major), result);
    }
  }

  return result;
}

Stmt CpuIslEmitter::EmitMark(const isl::ast_node_mark &node) {
  std::string mark = node.get_id().get_name();

  auto it = AkgSupportedForType.find(mark);
  if (it != AkgSupportedForType.end()) {
    for_type_ = it->second;
    Stmt stmt = EmitAst(node.get_node());
    for_type_ = ForType::Serial;
    return stmt;
  } else {
    Stmt stmt = EmitAst(node.get_node());
    return AttrStmt::make(Expr("INFO"), mark, StringImm::make(mark), stmt);
  }
}

Stmt CpuIslEmitter::EmitFor(const isl::ast_node_for &node) {
  isl::id isl_iter_id = node.get_iterator().as<isl::ast_expr_id>().get_id();
  VarExpr iter_expr(isl_iter_id.to_str());
  PushIter(iter_expr.get());

  Expr init_expr = Interpret(node.get_init());

  auto isl_cond = node.get_cond().as<isl::ast_expr_op>();
  CHECK(isl_cond.as<isl::ast_expr_op_lt>() || isl_cond.as<isl::ast_expr_op_le>());
  auto cond_lhs = isl_cond.get_arg(0).as<isl::ast_expr_id>();
  CHECK(cond_lhs);
  CHECK_EQ(cond_lhs.get_id(), isl_iter_id);
  Expr cond_expr = Interpret(isl_cond.get_arg(1)) - init_expr;
  if (isl_cond.as<isl::ast_expr_op_le>()) {
    cond_expr = Simplify_cce(cond_expr + 1);
  }

  int64_t inc = static_cast<int64_t>(WrappedStrtol(node.get_inc().to_C_str()));
  CHECK_EQ(inc, 1) << "We guarantee stride=1 by making scale=false in poly.";
  ForType for_type = for_type_;
  for_type_ = ForType::Serial;
  Stmt body_stmt = EmitAst(node.get_body());
  PopIter(iter_expr.get());
  if (auto imm = cond_expr.as<IntImm>()) {
    if (imm->value == 1) {
      Map<Var, Expr> replace_var;
      replace_var.Set(iter_expr, init_expr);
      return air::ir::Substitute(body_stmt, replace_var);
    }
  }
  return For::make(iter_expr, init_expr, cond_expr, for_type, DeviceAPI::None, body_stmt);
}

Stmt CpuIslEmitter::EmitBlock(const isl::ast_node_block &block_node) {
  std::vector<Stmt> stmts;

  int num = block_node.get_children().size();
  int last_num = 0;
  ForType for_type = for_type_;
  for (int i = num - 1; i >= 0; --i) {
    for_type_ = for_type;
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

Stmt CpuIslEmitter::EmitUserStmt(const isl::ast_node_user &node) {
  CHECK(node.get_expr().isa<isl::ast_expr_op>());
  isl::ast_expr_op usr_expr = node.get_expr().as<isl::ast_expr_op>();
  stmt_id_ = usr_expr.get_arg(0).as<isl::ast_expr_id>().get_id();
  node_id_ = node.get_annotation();
  const Node *stmt_node = info_.analysis_result_.GetStatementMap().at(stmt_id_);
  CHECK(stmt_node);
  // compute VarMap to replace old iterators
  auto cpu_build = node_info_map_.at(node_id_).build;
  auto cpu_tuple = info_.analysis_result_.GetOperatorDomainMap().at(stmt_id_).tuple;
  auto cpu_iterator_map = node_info_map_.at(node_id_).iterator_map;

  var_map_.clear();
  for (unsigned int i = 0; i < cpu_tuple.size(); i++) {
    isl::id isl_old_iter = cpu_tuple.get_id(i);
    auto isl_expr = cpu_build.expr_from(cpu_iterator_map.get_pw_aff(i));
    Expr halide_new_iter = Interpret(isl_expr);
    var_map_.emplace(isl_old_iter, halide_new_iter);
  }

  Stmt stmt = EmitUserStmtContent(stmt_node);
  if (stmt_node->IsInstance<Provide>()) {
    const auto op = static_cast<const Provide *>(stmt_node);
    if (info_.analysis_result_.GetReduceMap().count(op) != 0) {
      return AttrStmt::make(Expr("INFO"), "reduce_provide", Expr("reduce_provide"), stmt);
    }
  }
  return stmt;
}

Stmt CpuIslEmitter::EmitRealizeForGlobalTensor(const Stmt &from) {
  auto binds = info_.user_config_.GetBind();
  auto origin_binds = info_.user_config_.GetOriginBind();
  Stmt stmt = from;
  for (auto bind : binds) {
    if (!bind.first.defined()) {
      continue;
    }
    // input and output tensor, no need to emit realize
    if (origin_binds.find(bind.first) != origin_binds.end()) {
      continue;
    }
    // promoted tensor, the realize info already emitted before
    std::string name = bind.first->op->name;
    if (IsEndsWith(name, LOCAL_MEMORY)) {
      continue;
    }
    // if the tensor is temporary and it is not promoted, it needs to emit realize
    stmt = InsertRealize(stmt, isl::id(info_.GetCtx(), bind.first->op->name));
  }
  return stmt;
}

Stmt CpuIslEmitter::InsertRealize(const Stmt &from, const isl::id &var) {
  Stmt stmt = FindInnerRealize(var.get_name()).Mutate(from);

  Tensor t = info_.FindTensorWithLargestShape(var);
  Region bounds;

  if (bounds.empty()) {
    for (auto j : t->shape) {
      bounds.push_back(Range::make_by_min_extent(Expr(0), j));
    }
  }

  auto buf = info_.user_config_.GetBind().at(t);
  auto tt = placeholder(t->shape, t->dtype, t->op->name);
  stmt = TensorSubstitute(stmt, t->op, tt->op, tt->value_index);
  t = tt;
  stmt = TensorStringSubstitute(stmt, t->op->func_name(), t->op, t->value_index);
  stmt = Realize::make(t->op, t->value_index, t->dtype, bounds, const_true(1), stmt);
  std::string realize_name = LOCAL_MEMORY;

  stmt = AttrStmt::make(t->op, air::ir::attr::realize_scope, Expr(realize_name), stmt);
  return stmt;
}

Stmt CpuIslEmitter::EmitCall(const isl::ast_node_user &node) {
  CHECK(node.get_expr().isa<isl::ast_expr_op>());
  isl::ast_expr_op usr_expr = node.get_expr().as<isl::ast_expr_op>();
  CHECK(usr_expr);
  auto id_name = usr_expr.get_arg(0).as<isl::ast_expr_id>().get_id().get_name();
  std::regex delimiters("__");
  std::vector<std::string> args(std::sregex_token_iterator(id_name.begin(), id_name.end(), delimiters, -1),
                                std::sregex_token_iterator());
  CHECK_GT(args.size(), 2) << "Emit call must have function name";
  if (args[1] == MATRIX_TRANSPOSE_FUNCTION) {
    return EmitMatrixTranspose(args);
  } else if (args[1] == REDUCE_FUNCTION) {
    return EmitReduce(args);
  }
  return IslEmitter::EmitCall(node);
}

Stmt CpuIslEmitter::EmitMatrixTranspose(const std::vector<std::string> &names) {
  Tensor t = info_.FindTensor(names[3]);
  Array<Expr> indices;
  Array<Expr> args;
  args.push_back(make_zero(Int(INT_32)));
  for (size_t i = 0; i < t.ndim(); i++) {
    indices.push_back(make_zero(Int(INT_32)));
    args.push_back(t->shape[i]);
  }
  Expr addr = Call::make(Handle(), air::ir::intrinsic::tvm_address_of, {t(indices)}, Call::PureIntrinsic);
  args.Set(0, addr);
  Expr matrix_trans = Call::make(Handle(), names[1], args, Call::Intrinsic);
  return Evaluate::make(matrix_trans);
}

Stmt CpuIslEmitter::EmitReduce(const std::vector<std::string> &names) {
  Tensor t = info_.FindTensor(names[3]);
  Array<Var> vars;
  Array<Expr> indices;
  for (size_t i = 0; i < t.ndim(); i++) {
    auto loop_var = Variable::make(Int(INT_32), "loop" + std::to_string(i));
    vars.push_back(loop_var);
    indices.push_back(loop_var);
  }
  indices.Set(t.ndim() - 1, make_zero(Int(INT_32)));
  Expr value = maker_map[names[2]](t(indices), t(vars));
  Stmt body = Provide::make(t->op, t->value_index, value, indices);
  for (int i = t.ndim() - 1; i >= 0; i--) {
    body = For::make(vars[i], make_zero(Int(INT_32)), t->shape[i], air::ir::ForType::Serial, DeviceAPI::None, body);
  }
  return body;
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
