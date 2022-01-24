/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "poly/npu_isl_emitter.h"

#include "ir_pass.h"
#include "poly/dma_inject.h"
#include "pass/utils.h"
#include "poly/spec_gemm_builder.h"
#include "poly/dsa_utils.h"

namespace akg {
namespace ir {
namespace poly {
class MadMarker : public IRMutator {
 public:
  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    Stmt stmt = ParseStmtOps(op, s);
    return stmt;
  }

  void InsertInsnAttr(const Provide *p, const std::string &str) {
    for (const auto &i : insn_attrs_) {
      if (i.first == p) LOG(WARNING) << "Provide* " << p << " was registered in insn_attrs_ with " << str;
    }
    insn_attrs_.emplace_back(p, str);
  }

  Stmt ParseStmtOps(const Provide *op, const Stmt &s) {
    if (isImm(op->value)) {
      InsertInsnAttr(op, std::string("broadcast"));
      return s;
    }
    ParseMad(op->value, op);
    return s;
  }

  void ParseMad(const Expr &val, const Provide *pop) {
    auto op = val.as<Call>();
    if (op && op->name == "mad") InsertInsnAttr(pop, std::string("mad"));
  }

  Stmt Run(Stmt stmt) {
    stmt = this->Mutate(stmt);
    if (insn_attrs_.empty()) return stmt;

    for (auto i = insn_attrs_.begin(); i != insn_attrs_.end(); ++i) {
      if (i->second == "broadcast") {
        auto j = i;
        ++j;
        if (j != insn_attrs_.end() && j->second == "mad") {
          LOG(INFO) << "There is a mmu in MultiInstSplitter";
          i = j;
        }
      }
      if (i->second == "mad") {
        stmt = AttrStmt::make(make_zero(Int(32)), "pragma_emit_insn", Expr(i->second), stmt);
      }
    }

    return stmt;
  }

  MadMarker() = default;
  ~MadMarker() override = default;

 private:
  std::vector<std::pair<const Provide *, std::string>> insn_attrs_;
};

class GatherVar : public air::ir::IRVisitor {
 public:
  explicit GatherVar(Map<Tensor, Buffer> binds) : binds_(std::move(binds)) {}
  void Visit_(const Variable *op) final {
    if (visit_var_) {
      vars_.insert(op);
    }
    IRVisitor::Visit_(op);
  }
  ~GatherVar() override = default;

  void Visit_(const Provide *op) final {
    if (std::any_of(binds_.begin(), binds_.end(),
                    [=](const std::pair<Tensor, Buffer> &i) { return (op->func->func_name() == i.first->op->name); })) {
      Array<Expr> left_args = op->args;
      Array<Expr> right_args;
      if (auto right = op->value.as<Call>()) {
        if (right->call_type == Call::Halide) {
          right_args = right->args;
        }
      }
      visit_var_ = true;
      for (unsigned int i = 0; i < left_args.size(); i++) {
        this->Visit(Simplify_cce(left_args[i] - right_args[i]));
      }
      visit_var_ = false;
    }
    IRVisitor::Visit_(op);
  }

  std::unordered_set<const Variable *> vars_;
  const Map<Tensor, Buffer> binds_;
  bool visit_var_{false};
};

class HoistC0Write : public IRMutator {
 public:
  HoistC0Write(const Map<Tensor, Buffer> &binds, const Stmt &write) : write_(write) {
    auto f = GatherVar(binds);
    f.Visit(write);
    vars_ = f.vars_;
  }
  ~HoistC0Write() override = default;

  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (!mutate_) {
      for (const auto &v : vars_) {
        if (op->loop_var->name_hint == v->name_hint) {
          found_ = true;
          innermost_for_ = op;
        }
      }
    }
    if (mutate_) {
      if (op == innermost_for_ || op->body.get() == innermost_for_) {
        for (const auto &v : vars_) {
          if (op->loop_var->name_hint == v->name_hint) {
            vmap_.emplace(v, op->loop_var);
            mutate_write_ = true;
            write_ = this->Mutate(write_);
            mutate_write_ = false;
            vmap_.clear();
          }
        }
      }
      if (op == innermost_for_) {
        return For::make(op->loop_var, op->min, op->extent, op->for_type, op->device_api,
                         Block::make(op->body, write_));
      }
    }
    return IRMutator::Mutate_(op, s);
  }
  Expr Mutate_(const Variable *op, const Expr &e) final {
    if (mutate_write_ && vmap_.count(op)) {
      return vmap_.at(op);
    } else {
      return e;
    }
  }
  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    if (mutate_write_) {
      Array<Expr> args;
      for (const auto &arg : op->args) {
        args.push_back(this->Mutate(arg));
      }
      auto value = this->Mutate(op->value);
      return Provide::make(op->func, op->value_index, value, args);
    }
    return IRMutator::Mutate_(op, s);
  }
  Expr Mutate_(const Call *op, const Expr &e) final {
    if (mutate_write_) {
      Array<Expr> args;
      for (const auto &arg : op->args) {
        args.push_back(this->Mutate(arg));
      }
      return Call::make(op->type, op->name, args, op->call_type, op->func, op->value_index);
    }
    return IRMutator::Mutate_(op, e);
  }

  bool found_{false};
  bool mutate_{false};
  bool mutate_write_{false};

 private:
  Stmt write_;
  const For *innermost_for_{nullptr};
  std::unordered_set<const Variable *> vars_;
  std::unordered_map<const Variable *, VarExpr> vmap_;
};

/*
 * for i
 *   for j
 *     if (cond)
 *        S0
 *     else
 *        S1
 *
 * transform to
 *
 * for i
 *   for j
 *     if (cond)
 *        S0
 * for i
 *   for j
 *     if (!cond)
 *        S1
 *
 * Note: we need to ensure that block statements are not split, otherwise the split is not safe.
 */
class IfThenElseSplitter {
 public:
  Stmt Run(const Stmt &stmt) {
    std::vector<Stmt> split_stmts = DescendOrSplit(stmt);
    return JoinSplittedStmts(split_stmts);
  }

 private:
  static Stmt JoinSplittedStmts(const std::vector<Stmt> &split_stmts) {
    if (split_stmts.empty()) {
      return Evaluate::make(0);
    } else if (split_stmts.size() == 1) {
      return split_stmts[0];
    } else {
      Stmt block_stmt = Block::make(split_stmts[0], split_stmts[1]);
      for (size_t i = 2; i < split_stmts.size(); i++) {
        block_stmt = Block::make(block_stmt, split_stmts[i]);
      }
      return block_stmt;
    }
  }

  std::vector<Stmt> DescendOrSplit(const Stmt &body) {
    if (auto for_stmt = body.as<For>()) {
      return Mutate_(for_stmt);
    } else if (auto if_stmt = body.as<IfThenElse>()) {
      return Mutate_(if_stmt);
    } else if (auto attr = body.as<AttrStmt>()) {
      return Mutate_(attr);
    } else {
      std::vector<Stmt> stmts;
      stmts.push_back(body);
      return stmts;
    }
  }

  std::vector<Stmt> Mutate_(const For *op) {
    auto stmts = DescendOrSplit(op->body);
    for (auto &stmt : stmts) {
      stmt = For::make(op->loop_var, op->min, op->extent, op->for_type, op->device_api, stmt);
    }
    return stmts;
  }

  std::vector<Stmt> Mutate_(const IfThenElse *op) {
    std::vector<Stmt> merged_stmts;
    if (op->then_case.defined()) {
      auto stmts = DescendOrSplit(op->then_case);
      for (const auto &stmt : stmts) {
        Stmt new_if = IfThenElse::make(op->condition, stmt, Stmt());
        merged_stmts.push_back(new_if);
      }
    }
    if (op->else_case.defined()) {
      auto stmts = DescendOrSplit(op->else_case);
      for (const auto &stmt : stmts) {
        Stmt new_if = IfThenElse::make(Not::make(op->condition), stmt, Stmt());
        merged_stmts.push_back(new_if);
      }
    }
    return merged_stmts;
  }

  std::vector<Stmt> Mutate_(const AttrStmt *op) {
    auto stmts = DescendOrSplit(op->body);
    for (auto &stmt : stmts) {
      stmt = AttrStmt::make(op->node, op->attr_key, op->value, stmt);
    }
    return stmts;
  }
};

class TransposeLoopVarOrderInMad : public IRMutator {
 private:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) override {
    if (op->attr_key == "gemm_transposed" && op->value.as<StringImm>() &&
        op->value.as<StringImm>()->value == var_name_key_) {
      return AttrStmt::make(op->node, op->attr_key, op->value, op->body);
    } else if (op->attr_key == "pragma_emit_insn" && op->value.as<StringImm>() &&
               op->value.as<StringImm>()->value == "mad") {
      inside_mad_ = true;
      Stmt stmt = IRMutator::Mutate_(op, s);
      inside_mad_ = false;
      return AttrStmt::make(op->node, "gemm_transposed", Expr(var_name_key_), stmt);
    } else {
      return IRMutator::Mutate_(op, s);
    }
  }

  Expr Mutate_(const Call *op, const Expr &e) override {
    Expr expr = IRMutator::Mutate_(op, e);
    std::string var_name = op->name;
    if (inside_mad_ && var_name.find(var_name_key_) != std::string::npos) {
      op = expr.as<Call>();
      CHECK(op != nullptr);
      CHECK(op->args.defined());
      CHECK_GE(op->args.size(), 4);
      size_t arg_base = op->args.size() - 4;
      auto new_args = op->args;
      if (transpose_outer_) {
        new_args.Set(arg_base + 1, op->args[arg_base]);
        new_args.Set(arg_base, op->args[arg_base + 1]);
      }
      if (transpose_inner_) {
        new_args.Set(arg_base + 2, op->args[arg_base + 3]);
        new_args.Set(arg_base + 3, op->args[arg_base + 2]);
      }
      expr = Call::make(op->type, op->name, new_args, op->call_type, op->func, op->value_index);
      return expr;
    } else {
      return expr;
    }
  }

 public:
  Stmt Run(const Stmt &stmt, const std::string &var_name_key, bool transpose_outer, bool transpose_inner) {
    var_name_key_ = var_name_key;
    inside_mad_ = false;
    transpose_outer_ = transpose_outer;
    transpose_inner_ = transpose_inner;
    return IRMutator::Mutate(stmt);
  }

 private:
  bool inside_mad_{false};
  bool transpose_outer_{false};
  bool transpose_inner_{false};
  std::string var_name_key_;
};

class FindStmt {
 public:
  void FindAst(const isl::ast_node &node) {
    if (auto for_node = node.as<isl::ast_node_for>()) {
      FindAst(for_node.get_body());
    } else if (auto if_node = node.as<isl::ast_node_if>()) {
      FindAst(if_node.get_then_node());

      if (if_node.has_else_node()) {
        FindAst(if_node.get_else_node());
      }
    } else if (auto block_node = node.as<isl::ast_node_block>()) {
      for (auto child : block_node.get_children()) {
        FindAst(child);
      }
    } else if (auto mark_node = node.as<isl::ast_node_mark>()) {
      FindAst(mark_node.get_node());
    } else if (auto user_node = node.as<isl::ast_node_user>()) {
      if (user_node.to_str().find("S_") != std::string::npos) usernodes.push_back(user_node);
    } else {
      LOG(FATAL) << "NYI " << node << "\n";
    }
  }

  std::vector<isl::ast_node_user> usernodes;
};

std::vector<isl::id> GetLhsAllArgs(const NPUIslEmitter *emitter, const isl::ast_node_user &node) {
  CHECK(emitter);
  CHECK(node.get_expr().isa<isl::ast_expr_op>());
  isl::ast_expr_op usr_expr = node.get_expr().as<isl::ast_expr_op>();
  CHECK(usr_expr);
  CHECK(usr_expr.get_arg(0).as<isl::ast_expr_id>());
  auto stmt_id = usr_expr.get_arg(0).as<isl::ast_expr_id>().get_id();
  auto node_id = node.get_annotation();
  isl::ast_expr_op node_op;
  std::vector<isl::id> arg_ids;
  if (!emitter->info_.IsRead(stmt_id) && !emitter->info_.IsWrite(stmt_id)) {
    node_op = node.get_expr().as<isl::ast_expr_op>();
    if (!node_op) return arg_ids;
  } else {
    isl::ast_expr node_expr;
    auto iterator_map = emitter->node_info_map_.at(node_id).iterator_map;
    auto hoisted = iterator_map.range_factor_range();
    auto original = iterator_map.range_factor_domain().range_factor_range();
    auto build = emitter->node_info_map_.at(node_id).build;
    if (emitter->info_.IsRead(stmt_id)) {
      node_expr = build.access_from(isl::multi_pw_aff(hoisted));
    } else if (emitter->info_.IsWrite(stmt_id)) {
      node_expr = build.access_from(isl::multi_pw_aff(original));
    }
    node_op = node_expr.as<isl::ast_expr_op>();
    if (node_op && !node_op.as<isl::ast_expr_op_access>()) return arg_ids;
  }

  for (unsigned int i = 1; i < node_op.get_n_arg(); ++i) {
    if (auto expr_id = node_op.get_arg(i).as<isl::ast_expr_id>()) {
      arg_ids.push_back(expr_id.get_id());
    } else if (node_op.get_arg(i).as<isl::ast_expr_op>()) {
      isl::ast_expr_op in_op = node_op.get_arg(i).as<isl::ast_expr_op>();
      if (auto add_op = in_op.as<isl::ast_expr_op_add>()) {
        for (unsigned int j = 0; j < add_op.get_n_arg(); ++j) {
          if (auto add_id = add_op.get_arg(j).as<isl::ast_expr_id>()) {
            arg_ids.push_back(add_id.get_id());
          }
        }
      } else if (auto minus_op = in_op.as<isl::ast_expr_op_minus>()) {
        for (unsigned int j = 0; j < minus_op.get_n_arg(); ++j) {
          if (auto minus_id = minus_op.get_arg(j).as<isl::ast_expr_id>()) {
            arg_ids.push_back(minus_id.get_id());
          }
        }
      }
    }
  }
  return arg_ids;
}

bool ForShouldPassDown(const NPUIslEmitter *const emitter, const isl::ast_node &node, const isl::id &isl_iter_id) {
  std::queue<isl::ast_node> nodes;
  nodes.push(node);

  unsigned int user_node_cnt = 0;

  while (!nodes.empty()) {
    auto node_tmp = nodes.front();
    nodes.pop();
    if (auto block_node = node_tmp.as<isl::ast_node_block>()) {
      for (auto child : block_node.get_children()) {
        nodes.push(child);
      }
    } else if (auto for_node = node_tmp.as<isl::ast_node_for>()) {
      if (auto node_in_for = for_node.get_body()) {
        nodes.push(node_in_for);
      }
    } else if (auto if_node = node_tmp.as<isl::ast_node_if>()) {
      if (auto then_in_if = if_node.get_then_node()) {
        nodes.push(then_in_if);
      }
      if (if_node.has_else_node()) {
        if (auto else_in_if = if_node.get_else_node()) {
          nodes.push(else_in_if);
        }
      }
    } else if (auto user_node = node_tmp.as<isl::ast_node_user>()) {
      user_node_cnt++;
      bool is_contain = false;
      auto arg_ids = GetLhsAllArgs(emitter, user_node);
      auto rit = arg_ids.rbegin();
      if (rit != arg_ids.rend() && *rit == isl_iter_id) {
        is_contain = true;
      }
      if (!is_contain) return false;
    } else if (auto mark_node = node_tmp.as<isl::ast_node_mark>()) {
      nodes.push(mark_node.get_node());
    }
  }
  return true;
}

bool NPUIslEmitter::InjectMulticore(const std::string &iter) {
  bool should_insert_multi_core = false;
  if (multicore_info.enabled) {
    // coincident member is X in iterator "ccX"
    if (iter.substr(0, 2) == "cc") {
      const int radix = 10;
      CHECK_GE(iter.size(), 3);
      auto IsNumber = [](const std::string &str) -> bool {
        return !str.empty() &&
               std::find_if(str.begin(), str.end(), [](char c) { return !std::isdigit(c); }) == str.end();
      };
      CHECK(IsNumber(iter.substr(2)));
      size_t coincident_member = std::strtol(iter.substr(2).c_str(), nullptr, radix);
      if (multicore_info.coincident_map_depth.count(coincident_member)) {
        multicore_info.multicore_depth = multicore_info.coincident_map_depth.at(coincident_member);
        return true;
      }
      bool is_loop_in_multicore_band = (coincident_member < multicore_info.coincidence.size());
      if (is_loop_in_multicore_band) {
        should_insert_multi_core = multicore_info.coincidence[coincident_member];
        if (should_insert_multi_core) {
          ++multicore_info.multicore_depth;
          multicore_info.coincident_map_depth[coincident_member] = multicore_info.multicore_depth;
        }
      }
    } else {
      LOG(WARNING) << "multicore: unrecognized loop var " << iter;
    }
  }
  return should_insert_multi_core;
}

Stmt NPUIslEmitter::EmitFor(const isl::ast_node_for &node) {
  std::string iter = node.get_iterator().to_C_str();

  // get iterator
  isl::id isl_iter_id = node.get_iterator().as<isl::ast_expr_id>().get_id();
  VarExpr iter_expr(isl_iter_id.to_str());
  PushIter(iter_expr.get());

  // get init
  Expr init_expr = Interpret(node.get_init());

  // get condition
  auto isl_cond = node.get_cond().as<isl::ast_expr_op>();
  CHECK(isl_cond && (isl_cond.as<isl::ast_expr_op_lt>() || isl_cond.as<isl::ast_expr_op_le>()))
    << "unexpected isl ast cond: " << node.get_cond();
  auto cond_lhs = isl_cond.get_arg(0).as<isl::ast_expr_id>();
  CHECK(cond_lhs);
  CHECK_EQ(cond_lhs.get_id(), isl_iter_id);
  Expr cond_expr = Simplify_cce(Interpret(isl_cond.get_arg(1)) - init_expr);
  if (isl_cond.as<isl::ast_expr_op_le>()) {
    cond_expr = Simplify_cce(cond_expr + 1);
  }

  auto original_multicore_info = multicore_info;
  bool should_insert_multi_core = info_.user_config_.GetEnableMulticore() != 0 && InjectMulticore(iter);

  // emit body
  Stmt body_stmt = EmitAst(node.get_body());
  Stmt stmt;
  if (body_stmt.defined()) {
    // For the multi-core loop, keep the for whose extent is 1 to facilitate the processing of subsequent
    // the multi-core loop merging.
    if (!should_insert_multi_core && Equal(cond_expr, Expr(1))) {
      Map<Var, Expr> replace_var;
      replace_var.Set(iter_expr, init_expr);
      stmt = air::ir::Substitute(body_stmt, replace_var);
    } else {
      stmt = For::make(iter_expr, init_expr, cond_expr, ForType::Serial, DeviceAPI::None, body_stmt);
    }
    if (info_.user_config_.GetOptimizeForNPU()) {
      const int NPUC0SIZE = 16;
      // need to find the last axis
      if (Equal(cond_expr, Expr(NPUC0SIZE)) && ForShouldPassDown(this, node, isl_iter_id)) {
        stmt = AttrStmt::make(make_zero(Int(32)), "pass_down", NPUC0SIZE, stmt);
      }
    }
  } else {
    stmt = Evaluate::make(0);
  }

  PopIter(iter_expr.get());

  if (should_insert_multi_core) {
    stmt =
      AttrStmt::make(Expr(multicore_info.id), "pragma_multi_core_depth", Expr(multicore_info.multicore_depth), stmt);
    --multicore_info.multicore_depth;
  }

  return stmt;
}

Expr NPUIslEmitter::EmitLoad(const isl::ast_expr &expr, const Type type) {
  if (PRINT_NPU_ISL_EMITTER) {
    LOG(INFO) << ">>>>>>>>>>>>INPUT AST_NODE[LOAD]<<<<<<<<<<<<<<\n" << expr;
  }
  if (auto op = expr.as<isl::ast_expr_op>()) {
    if (auto access = op.as<isl::ast_expr_op_access>()) {
      // make buffer, index
      CHECK(op.get_arg(0).as<isl::ast_expr_id>());
      auto var = op.get_arg(0).as<isl::ast_expr_id>().get_id();
      Array<Expr> local_args;
      for (unsigned int i = 1; i < op.get_n_arg(); ++i) {
        local_args.push_back(Interpret(op.get_arg(i)));
      }
      if (info_.analysis_result_.CountBufferDefInfo(var)) {
        realize_use_.insert(var);
        if (!if_map_.count(var) || !AOutThanB(if_map_.at(var), cur_if_list_)) {
          realize_use_with_may_def_.insert(var);
        }
      }
      Tensor t = info_.FindTensor(var);
      if (info_.mmu_info_.IsIm2col()) {
        // compute_local_BUF find compute
        std::string name = t->op->name;
        for (const auto &updateTensor : info_.analysis_result_.GetUpdateTensor()) {
          if (updateTensor->op->name == name) {
            auto call = Call::make(type, updateTensor->op->name, local_args, Call::CallType::Halide, updateTensor->op,
                                   updateTensor->value_index);
            if (PRINT_NPU_ISL_EMITTER) {
              LOG(INFO) << ">>>>>>>>>>>>OUTPUT STMT<<<<<<<<<<<<\n" << call;
            }
            return call;
          }
        }
      }
      auto call = Call::make(type, t->op->name, local_args, Call::CallType::Halide, t->op, t->value_index);
      if (PRINT_NPU_ISL_EMITTER) {
        LOG(INFO) << ">>>>>>>>>>>>OUTPUT STMT<<<<<<<<<<<<\n" << call;
      }
      return call;
    }
  }
  return Expr();
}

/* Assign "aff" to *user and return isl_stat_error, effectively extracting
 * the first (and presumably only) affine expression in the isl_pw_aff
 * on which this function is used.
 */
static isl_stat ExtractSinglePiece(__isl_take isl_set *set, __isl_take isl_aff *aff, void *user) {
  auto p = reinterpret_cast<isl_aff **>(user);
  CHECK(p != nullptr);

  *p = aff;
  static_cast<void>(isl_set_free(set));

  return isl_stat_error;
}

static isl::pw_multi_aff ComputeNewBufferFootprint(const std::shared_ptr<TensorFootprintCluster> &fp_cluster,
                                                   const isl::pw_multi_aff &buffer_footprint) {
  if (!fp_cluster->UnWriteable()) return buffer_footprint;
  if (!fp_cluster->foot_print_.is_valid) return buffer_footprint;
  unsigned num_dims = fp_cluster->foot_print_.GetBoxDim();

  isl::pw_multi_aff new_buffer_footprint = buffer_footprint;
  for (unsigned dim = 0; dim < num_dims; ++dim) {
    isl::aff lower_bound = fp_cluster->foot_print_.GetBoxLowerBound(dim);
    isl::pw_aff dim_buf_fp = buffer_footprint.get_pw_aff(dim);
    if (dim_buf_fp.n_piece() != 1) return buffer_footprint;
    // there is only one piece, but we have to use the foreach API
    dim_buf_fp.foreach_piece([&lower_bound, &new_buffer_footprint, &dim](const isl::set &set,
                                                                         const isl::aff &aff) -> void {
      if (IsAffVarPlusOffset(lower_bound) && IsAffNonZeroConst(aff)) {
        isl::pw_aff zero = isl::pw_aff(isl::manage(isl_aff_set_constant_si(aff.copy(), 0)));
        new_buffer_footprint = isl::manage(isl_pw_multi_aff_set_pw_aff(new_buffer_footprint.copy(), dim, zero.copy()));
      }
    });
  }
  return new_buffer_footprint;
}

/*
 * Remove the constant offset from provide args, e.g. input_1_local_BUF(32, 7, cc2, cc3) = input_1(...)
 * Check the footprint cluster of the hoisted var to confirm this input tensor has multiple accesses
 * from shifted tiles. This should be improved by computing the new footprint with footprint_per_access(),
 * but from isl AST we do not know the footprint ID that corresponds to the GM -> BUF copy.
 */
isl::pw_multi_aff RemoveConstOffsetFromBufferFootprint(
  const isl::pw_multi_aff &buffer_footprint,
  const std::vector<std::pair<isl::union_set, BufferedFootPrintInfo>> &active_buffer_footprints) {
  const isl::id buffer_id = buffer_footprint.get_tuple_id(isl_dim_out);
  for (const auto &act_buf : active_buffer_footprints) {
    if (act_buf.second.cluster_id == buffer_id) {
      const auto &footprint_cluster = act_buf.second.cluster;
      return ComputeNewBufferFootprint(footprint_cluster, buffer_footprint);
    }
  }
  return buffer_footprint;
}

Stmt NPUIslEmitter::EmitRead(const isl::ast_node_user &node) {
  isl::id node_id = node.get_annotation();
  isl::pw_multi_aff iterator_map = node_info_map_.at(node_id).iterator_map;
  isl::pw_multi_aff hoisted = iterator_map.range_factor_range();
  isl::pw_multi_aff original = iterator_map.range_factor_domain().range_factor_range();

  isl::id original_tensor = original.get_tuple_id(isl_dim_out);
  bool isInputTensor = info_.FindTensorInOrig(original_tensor).defined();
  if (isInputTensor)
    hoisted = RemoveConstOffsetFromBufferFootprint(hoisted, info_.analysis_result_.ActiveBufferFootprints());

  auto build = node_info_map_.at(node_id).build;
  auto lhs = build.access_from(isl::multi_pw_aff(hoisted));
  auto rhs = build.access_from(isl::multi_pw_aff(original));

  size_t pos = info_.mmu_info_.GetBName().find("_local");
  std::string b_name =
    pos == std::string::npos ? info_.mmu_info_.GetBName() : info_.mmu_info_.GetBName().substr(0, pos);
  auto b_c1_name = b_name + LOCAL_C1;

  if (info_.user_config_.GetMatBDimH() > 0 && info_.user_config_.GetMatBDimW() > 0 &&
      original.get_tuple_id(isl_dim_out).get_name() == b_c1_name) {
    auto h_size = info_.user_config_.GetMatBDimH();
    auto w_size = info_.user_config_.GetMatBDimW();

    auto mpa = isl::multi_pw_aff(original);
    auto size = mpa.size();
    auto list = isl::aff_list(original.ctx(), size);

    CHECK_EQ(size, 4);
    isl_aff *affptr0 = nullptr;
    isl_aff *affptr1 = nullptr;
    isl_aff *affptr2 = nullptr;
    isl_aff *affptr3 = nullptr;
    isl_aff *affptr4 = nullptr;
    isl_aff *affptr5 = nullptr;
    CHECK(isl_pw_aff_foreach_piece(mpa.get_pw_aff(0).get(), &ExtractSinglePiece, &affptr0) == isl_stat_error);
    CHECK(affptr0 != nullptr);
    auto aff0 = (isl::manage(affptr0) / (h_size * w_size)).floor();

    CHECK(isl_pw_aff_foreach_piece(mpa.get_pw_aff(0).get(), &ExtractSinglePiece, &affptr1) == isl_stat_error);
    CHECK(affptr1 != nullptr);
    auto aff1 = (h_size - 1 - ((isl::manage(affptr1) / w_size).floor().mod(isl::val(original.ctx(), h_size)))) * w_size;

    CHECK(isl_pw_aff_foreach_piece(mpa.get_pw_aff(0).get(), &ExtractSinglePiece, &affptr2) == isl_stat_error);
    CHECK(affptr2 != nullptr);
    auto aff2 = w_size - 1 - (isl::manage(affptr2).mod(isl::val(original.ctx(), w_size)));

    CHECK(isl_pw_aff_foreach_piece(mpa.get_pw_aff(1).get(), &ExtractSinglePiece, &affptr3) == isl_stat_error);
    CHECK(affptr3 != nullptr);
    auto aff3 = isl::manage(affptr3) * (h_size * w_size);

    CHECK(isl_pw_aff_foreach_piece(mpa.get_pw_aff(2).get(), &ExtractSinglePiece, &affptr4) == isl_stat_error);
    CHECK(affptr4 != nullptr);
    auto aff4 = isl::manage(affptr4);

    CHECK(isl_pw_aff_foreach_piece(mpa.get_pw_aff(3).get(), &ExtractSinglePiece, &affptr5) == isl_stat_error);
    CHECK(affptr5 != nullptr);
    auto aff5 = isl::manage(affptr5);

    list = list.add(aff3 + aff1 + aff2).add(aff0).add(aff5).add(aff4);
    auto ma = isl::multi_aff(mpa.get_space(), list);

    rhs = build.access_from(isl::multi_pw_aff(ma));
  }

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
      if (info_.analysis_result_.CountBufferDefInfo(var)) {
        realize_may_def_.insert(var);
        if_map_.emplace(var, cur_if_list_);
        if (cur_if_list_.empty()) {
          realize_must_def_.insert(var);
        }
      }
      hoisted_read_.insert(var);
      if (info_.mmu_info_.IsIm2col() && !info_.analysis_result_.GetUpdateTensor().empty()) {
        return Provide::make(info_.analysis_result_.GetUpdateTensor()[0]->op, 0, value, local_args);
      }
      return Provide::make(t->op, 0, value, local_args);
    }
  }
  return Stmt();
}

Stmt NPUIslEmitter::EmitWrite(const isl::ast_node_user &node, AtomicType atomic) {
  auto node_id = node.get_annotation();
  CHECK_GT(node_info_map_.count(node_id), 0);
  auto iterator_map = node_info_map_.at(node_id).iterator_map;
  auto hoisted = iterator_map.range_factor_range();
  auto original = iterator_map.range_factor_domain().range_factor_range();

  // refine atomic from reduce op
  bool doatomic = false;
  if (atomic == AtomicType::Add) {
    auto srcid = original.get_tuple_id(isl_dim_out);
    for (const auto &i : info_.analysis_result_.GetStatementMap()) {
      std::set<const Variable *> rmv;
      const auto provide = static_cast<const Provide *>(i.second);
      if (provide == nullptr || info_.analysis_result_.GetReduceMap().count(provide) != 1) continue;
      if (provide->func->func_name() != srcid.get_name()) continue;
      doatomic = true;
      if (!stmt_var_map_.count(i.first)) continue;
      VarMap vmap = stmt_var_map_.at(i.first);
      for (const auto &j : vmap) {
        for (auto reduce_iter_var : info_.analysis_result_.GetReduceMap().at(provide)) {
          if (reduce_iter_var->var.get()->name_hint != j.first.get_name()) continue;
          std::vector<const Variable *> iters = ExtractIterfromExpr().Run(j.second);
          for (auto v : iters)
            if (FindIter(v)) rmv.insert(v);
        }
      }

      for (auto j : rmv) rmif_.insert(j);
    }
  }

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
      if (info_.analysis_result_.CountBufferDefInfo(var)) {
        realize_may_def_.insert(var);
        if_map_.emplace(var, cur_if_list_);
        if (cur_if_list_.empty()) {
          realize_must_def_.insert(var);
        }
      }
      hoisted_write_.insert(var);

      if (doatomic) {
        auto call = Call::make(type, t->op->name, local_args, Call::CallType::Halide, t->op, t->value_index);
        value = Add::make(call, value);
        return AttrStmt::make(make_zero(Int(32)), ATTR_ATOMIC_ADD, Expr(1), Provide::make(t->op, 0, value, local_args));
      }

      // remove original copy out promotion statement because it is sinked into if stmt of computation
      if (info_.analysis_result_.GetConditionalWriteBufferFootprints().count(t->op->name)) return Evaluate::make(0);

      return Provide::make(t->op, 0, value, local_args);
    }
  }
  return Stmt();
}

Stmt NPUIslEmitter::EmitUserStmt(const isl::ast_node_user &node) {
  if (is_old_gemm_c1write_) {
    LOG(INFO) << "don't emit conv origin user stmt.";
    return Evaluate::make(Expr(0));
  } else {
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
      std::string replace_id = isl_old_iter.get_name() + "_";
      std::vector<const Variable *> vec = ExtractIterfromExpr().Run(halide_new_iter);
      for (auto item : vec) {
        std::string new_name = item->name_hint;
        auto iter_prefix = info_.user_config_.GetIterPrefix(info_.mmu_info_.IsSpecGemm());
        size_t pos = new_name.find(iter_prefix);
        if (pos != std::string::npos) {
          new_name = new_name.replace(pos, iter_prefix.size(), replace_id);
          iters_old_name_.emplace(item, item->name_hint);
          iters_new_name_.emplace(item, new_name);
        }
      }
    }

    VarMap vmap = var_map_;
    stmt_var_map_[stmt_id_] = vmap;
    auto user_stmt = EmitUserStmtContent(stmt_node);

    // fix conv prefusion dma if condition
    bool add_attr = false;
    std::string type_key = std::string(stmt_node->GetTypeKey());
    if (!info_.mmu_info_.IsSpecGemm() && (type_key == "IfThenElse")) {
      isl::union_set transfer_stmt = info_.analysis_result_.GetTransferStmt();
      if (!transfer_stmt.is_empty()) {
        transfer_stmt.foreach_set([&add_attr, this](const isl::set &s) -> void {
          if (s.get_tuple_name() == stmt_id_.get_name()) {
            add_attr = true;
          }
        });
      }
    }
    if (add_attr) {
      user_stmt = AttrStmt::make(make_zero(Int(32)), "pragma_fix_ifcondition", Expr(1), user_stmt);
    }
    return user_stmt;
  }
}

AtomicType GetAtomicWrite(const isl::id &id, const StatementMap &statements) {
  for (const auto &i : statements) {
    const Node *stmt_node = i.second;
    if (stmt_node->IsInstance<Provide>()) {
      auto provide = static_cast<const Provide *>(stmt_node);
      if (const auto cop = provide->func.as<ComputeOpNode>()) {
        if (cop->attrs.count(ATTR_ATOMIC_ADD) != 0) {
          if (auto str_op = cop->attrs.at(ATTR_ATOMIC_ADD).as<StringImm>()) {
            auto str = str_op->value;
            if (str == id.get_name()) return AtomicType::Add;
          }
        }
      }
    }
  }
  return AtomicType::Equ;
}

AtomicType GetAtomicWrite(const isl::id &id, const std::unordered_set<std::string> &tensors) {
  return tensors.count(id.get_name()) > 0 ? AtomicType::Add : AtomicType::Equ;
}

Stmt NPUIslEmitter::EmitStmt(const isl::ast_node_user &node) {
  CHECK(node.get_expr().isa<isl::ast_expr_op>());
  isl::ast_expr_op usr_expr = node.get_expr().as<isl::ast_expr_op>();
  CHECK(usr_expr);
  auto stmt_id = usr_expr.get_arg(0).as<isl::ast_expr_id>().get_id();
  auto node_id = node.get_annotation();

  if (info_.IsRead(stmt_id)) {
    auto s = EmitRead(node);
    if (PRINT_NPU_ISL_EMITTER) {
      LOG(INFO) << ">>>>>>>>>>>>INPUT AST_NODE[READ]<<<<<<<<<<<<<<\n" << node;
      LOG(INFO) << ">>>>>>>>>>>>OUTPUT STMT<<<<<<<<<<<<\n" << s;
    }
    return s;
  } else if (info_.IsWrite(stmt_id)) {
    auto s = Stmt();
    if (info_.IsGMWrite(stmt_id)) {
      auto iterator_map = node_info_map_.at(node_id).iterator_map;
      auto original = iterator_map.range_factor_domain().range_factor_range();
      auto srcid = original.get_tuple_id(isl_dim_out);

      /* ******************************************************************************************
       * when "enable_atomic_add" is True, akg will use new atomic add opt flow on NPU;
       * when "enable_atomic_add" is False, akg will choose the old atomic add opt flow on NPU,
       * to find 'atomic_add' attribue and corresponding value, tensor name, in the compute node.
       *********************************************************************************************/
      AtomicType type = info_.user_config_.GetEnableAtomicAdd()
                          ? GetAtomicWrite(srcid, info_.analysis_result_.GetReduceOutTensors())
                          : GetAtomicWrite(srcid, info_.analysis_result_.GetStatementMap());
      s = EmitWrite(node, type);
    } else {
      s = EmitWrite(node, AtomicType::Equ);
    }
    if (PRINT_NPU_ISL_EMITTER) {
      LOG(INFO) << ">>>>>>>>>>>>INPUT AST_NODE[WRITE]<<<<<<<<<<<<<<\n" << node;
      LOG(INFO) << ">>>>>>>>>>>>OUTPUT STMT<<<<<<<<<<<<\n" << s;
    }
    return s;
  } else {
    SetMMU(stmt_id);
    return EmitUserStmt(node);
  }
}

void NPUIslEmitter::SetMMU(const isl::id &stmt_id) {
  auto cur_op = info_.analysis_result_.GetStmtOpInfoMap().at(stmt_id);
  opinfo_.isMMU = cur_op.isMMU || opinfo_.isMMU;
  opinfo_.ops.insert(opinfo_.ops.end(), cur_op.ops.begin(), cur_op.ops.end());
  is_mmu_ = true;
}

std::string NPUIslEmitter::ReplaceAxis(const std::string &old_axis) {
  for (const auto &i : iters_old_name_) {
    if (i.second == old_axis) {
      return iters_new_name_.at(i.first);
    }
  }
  return old_axis;
}

std::vector<std::string> NPUIslEmitter::ConstructPrefix() {
  std::vector<std::string> prefix;
  PartitionSingle *single = PartitionSingle::getInstance();
  if (single != nullptr && PartitionSingle::getTimes() == 1) {
    // m isolate first gemm or special gemm
    std::map<std::string, Expr> fractalInfo = PartitionSingle::getFractalInfo();
    std::vector<std::pair<std::string, std::string>> axis;
    axis.emplace_back(std::pair<std::string, std::string>("b", ATTR_CONV_BATCH));
    axis.emplace_back(std::pair<std::string, std::string>("no_", ATTR_CONV_TILE_N));
    axis.emplace_back(std::pair<std::string, std::string>("mo_", ATTR_CONV_TILE_M));
    axis.emplace_back(std::pair<std::string, std::string>("mi", ATTR_CONV_M_INNER));
    axis.emplace_back(std::pair<std::string, std::string>("ni", ATTR_CONV_N_INNER));
    axis.emplace_back(std::pair<std::string, std::string>("ko_", ATTR_CONV_TILE_K));
    for (const auto &i : axis) {
      CHECK(fractalInfo.find(i.second) != fractalInfo.end());
      if (!is_const_int(fractalInfo[i.second], 1)) {
        prefix.push_back(i.first);
      }
    }
  } else if (single != nullptr && PartitionSingle::getTimes() == 2) {
    /********************
     * m isolate second gemm
     * for gemm axis size is larger than 1, prefix.push_back(*)
     * mo_ is 1, so prefix.push_back at last
     * ***********************/
    std::map<std::string, Expr> fractalInfo = PartitionSingle::getFractalInfo();
    std::vector<std::pair<std::string, std::string>> axis;
    axis.emplace_back(std::pair<std::string, std::string>("b", ATTR_CONV_BATCH));
    axis.emplace_back(std::pair<std::string, std::string>("no_", ATTR_CONV_TILE_N));
    axis.emplace_back(std::pair<std::string, std::string>("mi", ATTR_CONV_M_INNER));
    axis.emplace_back(std::pair<std::string, std::string>("ni", ATTR_CONV_N_INNER));
    axis.emplace_back(std::pair<std::string, std::string>("ko_", ATTR_CONV_TILE_K));
    for (const auto &i : axis) {
      CHECK(fractalInfo.find(i.second) != fractalInfo.end());
      if (!is_const_int(fractalInfo[i.second], 1)) {
        prefix.push_back(i.first);
      }
    }
    prefix.emplace_back("mo_");
  }
  return prefix;
}

Stmt NPUIslEmitter::EmitGemmRangeInfoBackPropFilter(const Stmt &stmt) {
  PartitionSingle *single = PartitionSingle::getInstance();
  CHECK(single != nullptr);
  std::map<std::string, Expr> fractal_int_info = PartitionSingle::getFractalInfo();
  int c0_range_idx = range_idx_++;

  int K = fractal_int_info[ATTR_SPEC_GEMM_K].as<IntImm>()->value;
  int MO = fractal_int_info[ATTR_SPEC_GEMM_M_ALIGN].as<IntImm>()->value;
  int KO = fractal_int_info[ATTR_SPEC_GEMM_K_ALIGN].as<IntImm>()->value;
  int NO = fractal_int_info[ATTR_SPEC_GEMM_N_ALIGN].as<IntImm>()->value;
  int MI = fractal_int_info[ATTR_SPEC_GEMM_M_INNER].as<IntImm>()->value;
  int KI = fractal_int_info[ATTR_SPEC_GEMM_K_INNER].as<IntImm>()->value;
  int NI = fractal_int_info[ATTR_SPEC_GEMM_N_INNER].as<IntImm>()->value;
  int tile_m = fractal_int_info[ATTR_SPEC_GEMM_TILE_M].as<IntImm>()->value;
  int tile_k = fractal_int_info[ATTR_SPEC_GEMM_TILE_K].as<IntImm>()->value;
  int tile_n = fractal_int_info[ATTR_SPEC_GEMM_TILE_N].as<IntImm>()->value;

  // PLZ make sure axis order:
  // for (n_isolate) {
  //   for (m_isolate) {
  //     for (k_isolate) {
  //     }
  //   }
  // }stmtOpInfo

  int m_isolate = MO * MI % tile_m;
  int k_isolate = KO * KI % tile_k;
  int n_isolate = NO * NI % tile_n;

  int range_idx_max = 1;
  int k_base = 1;

  int m_base = k_base;
  if (k_isolate) {
    m_base *= 2;
    range_idx_max *= 2;
  }

  int n_base = m_base;
  if (m_isolate) {
    n_base *= 2;
    range_idx_max *= 2;
  }

  if (n_isolate) {
    range_idx_max *= 2;
  }

  CHECK(c0_range_idx < range_idx_max) << c0_range_idx << ":" << range_idx_max;

  Map<std::string, Range> range_map;

  if (KO * KI < tile_k) {
    tile_k = KO * KI;
  }
  int ko_min = k_isolate ? ((c0_range_idx / k_base % BINARY_FACTOR) ? (KO * KI / tile_k) : (0)) : (0);
  int ko_ext = k_isolate ? ((c0_range_idx / k_base % BINARY_FACTOR) ? (1) : (KO * KI / tile_k)) : (KO * KI / tile_k);
  range_map.Set("ko_", Range(Expr(ko_min), Expr(ko_ext)));
  if (k_isolate) {
    if (c0_range_idx / k_base % BINARY_FACTOR) {
      range_map.Set("k_size", Range(Expr(0), Expr(K - KO * KI / tile_k * tile_k)));
    } else {
      range_map.Set("k_size", Range(Expr(0), Expr(tile_k)));
    }
  } else {
    if (KO * KI == K) {
      range_map.Set("k_size", Range(Expr(0), Expr(tile_k)));
    } else {
      range_map.Set("k_tail_size", Range(Expr(0), Expr(K - (KO * KI / tile_k - 1) * tile_k)));
      range_map.Set("k_tail", Range(Expr(0), Expr(KO * KI / tile_k - 1)));
      range_map.Set("k_size", Range(Expr(0), Expr(tile_k)));
    }
  }
  range_map.Set(K_C1, Range(Expr(0), Expr(K)));

  if (NO * NI < tile_n) {
    tile_n = NO * NI;
  }
  int no_min = n_isolate ? ((c0_range_idx / n_base % BINARY_FACTOR) ? (NO * NI / tile_n) : (0)) : (0);
  int no_ext = n_isolate ? ((c0_range_idx / n_base % BINARY_FACTOR) ? (1) : (NO * NI / tile_n)) : (NO * NI / tile_n);
  range_map.Set("no_", Range(Expr(no_min), Expr(no_ext)));

  if (MO * MI < tile_m) {
    tile_m = MO * MI;
  }
  int mo_min = m_isolate ? ((c0_range_idx / m_base % BINARY_FACTOR) ? (MO * MI / tile_m) : (0)) : (0);
  int mo_ext = m_isolate ? ((c0_range_idx / m_base % BINARY_FACTOR) ? (1) : (MO * MI / tile_m)) : (MO * MI / tile_m);
  range_map.Set("mo_", Range(Expr(mo_min), Expr(mo_ext)));

  return AttrStmt::make(range_map, PRAGMA_GEMM_C0, Expr(c0_range_idx), stmt);
}

void NPUIslEmitter::CollectGemmRangeInfoNewAxis(std::vector<Range> &range, std::vector<std::string> &prefix,
                                                std::unordered_map<std::string, bool> &outerAxis, Range &axisMRange,
                                                Map<std::string, Range> &range_map,
                                                Map<std::string, VarExpr> &axis_map) {
  for (unsigned int i = 0; i < range.size(); i++) {
    std::stringstream ss;
    ss << "ee" << i;
    VarExpr oldName = VarExpr(ss.str());
    std::string newAxis = ReplaceAxis(ss.str());
    Range curRange = range[i];

    if (newAxis == ss.str()) {
      if (prefix[i] != "mi" && prefix[i] != "ni") {
        newAxis = prefix[i];
      } else {
        if (curRange->min.as<IntImm>() && curRange->min.as<IntImm>()->value == 0 && curRange->extent.as<IntImm>() &&
            curRange->extent.as<IntImm>()->value == 1) {
          newAxis = "";
        }
      }
    }

    if (!newAxis.empty()) {
      size_t pos = newAxis.find('_');
      CHECK(pos != std::string::npos);
      std::string tmp = newAxis.substr(0, pos + 1);
      outerAxis[tmp] = true;
      if (tmp == "mo_") axisMRange = curRange;
    }
    axis_map.Set(newAxis, oldName);
    range_map.Set(newAxis, curRange);
  }
}

Stmt NPUIslEmitter::EmitGemmRangeInfo(Stmt stmt) {
  /********************
   * this function is to emit gemm rangeInfo with pragma_gemm_c0 attribute
   *
      // attr [{"m_size": range(min=0, ext=49), "": range(min=0, ext=1), "no_0": range(min=0, ext=16), "m_lager_size":
  range(min=0, ext=64), "ko_4": range(min=0, ext=10), "mo_": range(min=0, ext=1)}] pragma_gemm_c0 = 0
      // attr [placeholder(input1_local_C1_local_C0B, 0x5654b8abca60)] realize_scope = DOT_LOCAL_C0B
      realize input1_local_C1_local_C0B([0, 6], [0, 8], [0, 16], [0, 16]) {
        // attr [0] pragma_bypath_filter_c0 = 0
        produce input1_local_C1_local_C0B {
          // attr [0] pragma_emit_insn = "dma_copy"
          for (ee10, 0, 6) {
            for (ee11, 0, 8) {
              for (ee12, 0, 16) {
                for (ee13, 0, 16) {
                  input1_local_C1_local_C0B(ee10, ee11, ee12, ee13) =input1_local_C1(((6*ko_4) + ee10), ((8*no_0) +
  ee11), ee12, ee13)
                }
              }
            }
          }
        }
      }
    }
  }
}

   * all the rangeInfo is outer outer axis in spec gemm
   *
   * explanation of: prefix and range
     CHECK(prefix.size() == range.size());

   * =========== Poly spec_gem input HalideIR ============
  produce output0_local_BUF {
    for (b, 0, 1) {
      for (no, 0, 128) {
        for (mo, 0, 4) {
          for (mi, 0, 16) {
            for (ni, 0, 16) {
              output0_local_BUF(b, no, mo, mi, ni) =0.000000h
              for (ko, 0, 64) {
                for (ki, 0, 16) {
                  output0_local_BUF(b, no, mo, mi, ni) =(output0_local_BUF(b, no, mo, mi, ni) + (input0_fractal_C1(b,
  mo, ko, mi, ki)*input1_local_C1(ko, no, ni, ki)))
                }
              }
            }
          }
        }
      }
    }
  }
}
}
}
set dim
No: 0
num_band_members: 5, tiling_flag: 1
index: 0, head: 0, body: 0, tail: 0, c1_size: 8, c1_flag: 1, c0_size: 65535, c0_flag: 0, seq: 0
index: 1, head: 0, body: 0, tail: 0, c1_size: 4, c1_flag: 1, c0_size: 65535, c0_flag: 0, seq: 1
index: 2, head: 0, body: 0, tail: 0, c1_size: 16, c1_flag: 1, c0_size: 65535, c0_flag: 0, seq: 2
index: 3, head: 0, body: 0, tail: 0, c1_size: 16, c1_flag: 1, c0_size: 65535, c0_flag: 0, seq: 3
index: 4, head: 0, body: 0, tail: 0, c1_size: 6, c1_flag: 1, c0_size: 65535, c0_flag: 0, seq: 4

main
prefix : no_   range: Range(min=0, extent=16)
prefix : mo_   range: Range(min=0, extent=1)
prefix : mi    range: Range(min=0, extent=1)
prefix : ni    range: Range(min=0, extent=1)
prefix : ko_   range: Range(min=0, extent=10)

isolate
prefix : no_   range: Range(min=0, extent=16)
prefix : mo_   range: Range(min=0, extent=1)
prefix : mi    range: Range(min=0, extent=1)
prefix : ni    range: Range(min=0, extent=1)
prefix : ko_   range: Range(min=0, extent=4)
***********************/
  // spec gemm set dim outer outer range
  std::vector<Range> range;
  if (info_.user_config_.GetTileSizeIsVar()) {
    // must equal to scop.cc
    const int t0_mo = 11;
    const int t0_ko = 13;
    const int t0_no = 17;
    range.emplace_back(Expr(0), floordiv(Var("NO") + t0_no - 1, t0_no));
    range.emplace_back(Expr(0), floordiv(Var("MO") + t0_mo - 1, t0_mo));
    range.emplace_back(Expr(0), Expr(1));
    range.emplace_back(Expr(0), Expr(1));
    range.emplace_back(Expr(0), floordiv(Var("KO") + t0_ko - 1, t0_ko));
  } else {
    range = info_.mmu_info_.GetRange(range_idx_);
  }
  Map<std::string, Range> range_map;
  Map<std::string, VarExpr> axis_map;
  std::unordered_map<std::string, bool> outer_axis;
  // spec gemm outer outer axis prefix name
  std::vector<std::string> prefix = ConstructPrefix();
  // same with gemm construct IR
  CHECK(prefix.size() == range.size());
  Range axis_m_range;

  CollectGemmRangeInfoNewAxis(range, prefix, outer_axis, axis_m_range, range_map, axis_map);

  std::vector<std::string> all_axis;
  all_axis.emplace_back("mo_");
  all_axis.emplace_back("no_");
  all_axis.emplace_back("ko_");
  for (const auto &i : all_axis) {
    if (outer_axis.find(i) == outer_axis.end()) {
      Range one(Expr(0), Expr(1));
      range_map.Set(i, one);
      if (i == "mo_") {
        axis_m_range = one;
      }
    }
  }
  /*************************************************
   * add m_size & m_larger_size
   * m_size: 49 = 3 * 16 + 1
   * m_larger_size: 64 = (3 + 1) * 16
   * **********************************/
  PartitionSingle *single = PartitionSingle::getInstance();
  if (single != nullptr) {
    if (!info_.user_config_.GetIsDynamic()) {
      CollectGemmMWSize(axis_m_range, range_map);
    } else {
      CollectGemmMWSizeDynamic(range_map);
    }
  }
  stmt = AttrStmt::make(axis_map, "pragma_spec_gemm_attr", Expr(0), stmt);
  stmt = AttrStmt::make(range_map, PRAGMA_GEMM_C0, Expr(range_idx_), stmt);
  range_idx_++;
  return stmt;
}

void NPUIslEmitter::CollectGemmMWSize(Range &axis_m_range, Map<std::string, Range> &range_map) {
  std::map<std::string, Expr> fractal_int_info = PartitionSingle::getFractalInfo();
  CHECK(fractal_int_info.find(ATTR_CONV_GMM_M) != fractal_int_info.end());
  CHECK(fractal_int_info.find(ATTR_CONV_TILE_M) != fractal_int_info.end());
  CHECK(fractal_int_info.find(ATTR_CONV_M_INNER) != fractal_int_info.end());
  CHECK(fractal_int_info[ATTR_CONV_GMM_M].as<IntImm>());
  CHECK(fractal_int_info[ATTR_CONV_TILE_M].as<IntImm>());
  CHECK(fractal_int_info[ATTR_CONV_M_INNER].as<IntImm>());
  if (fractal_int_info[ATTR_CONV_GMM_M].as<IntImm>()->value <
      fractal_int_info[ATTR_CONV_TILE_M].as<IntImm>()->value *
        fractal_int_info[ATTR_CONV_M_INNER].as<IntImm>()->value) {
    int64_t size = fractal_int_info[ATTR_CONV_GMM_M].as<IntImm>()->value;
    int64_t larger =
      fractal_int_info[ATTR_CONV_TILE_M].as<IntImm>()->value * fractal_int_info[ATTR_CONV_M_INNER].as<IntImm>()->value;
    CHECK(axis_m_range->min.as<IntImm>() != nullptr);
    CHECK(axis_m_range->extent.as<IntImm>() != nullptr);
    int64_t m_min = axis_m_range->min.as<IntImm>()->value;
    int64_t m_ext = axis_m_range->extent.as<IntImm>()->value;
    int m_CutSize = PartitionSingle::getCutM();
    m_CutSize = static_cast<int>(m_CutSize / fractal_int_info[ATTR_CONV_M_INNER].as<IntImm>()->value);
    int64_t cur_size = 0;
    int64_t cur_larger = 0;
    if (m_min == 0) {
      cur_larger = m_ext * m_CutSize * fractal_int_info[ATTR_CONV_M_INNER].as<IntImm>()->value;
      cur_size = cur_larger < size ? cur_larger : size;
    } else if (m_min > 0) {
      cur_size = size - m_min * m_CutSize * fractal_int_info[ATTR_CONV_M_INNER].as<IntImm>()->value;
      cur_larger = larger - m_min * m_CutSize * fractal_int_info[ATTR_CONV_M_INNER].as<IntImm>()->value;
    }
    range_map.Set("m_size", Range(Expr(0), Expr(cur_size)));
    range_map.Set("m_lager_size", Range(Expr(0), Expr(cur_larger)));
  } else {
    range_map.Set("m_size", Range(Expr(0), Expr(fractal_int_info[ATTR_CONV_GMM_M])));
    range_map.Set("m_lager_size", Range(Expr(0), Expr(fractal_int_info[ATTR_CONV_GMM_M])));
  }
  range_map.Set("w_size", Range(Expr(0), Expr(fractal_int_info[ATTR_CONV_M_CUT_SIZE])));
}

void NPUIslEmitter::CollectGemmMWSizeDynamic(Map<std::string, Range> &range_map) {
  std::map<std::string, Expr> fractal_int_info = PartitionSingle::getFractalInfo();
  CHECK(fractal_int_info.find(ATTR_CONV_GMM_M) != fractal_int_info.end());
  CHECK(fractal_int_info.find(ATTR_CONV_TILE_M) != fractal_int_info.end());
  CHECK(fractal_int_info.find(ATTR_CONV_M_INNER) != fractal_int_info.end());
  CHECK(fractal_int_info.find(ATTR_CONV_TILE_H) != fractal_int_info.end());
  CHECK(fractal_int_info.find(ATTR_CONV_TILE_W) != fractal_int_info.end());

  auto tile_h = fractal_int_info[ATTR_CONV_TILE_H];
  auto kernel_h = fractal_int_info[ATTR_CONV_KERNEL_H];
  auto stride_h = fractal_int_info[ATTR_CONV_STRIDE_H];
  auto win_h = floordiv(tile_h - kernel_h, stride_h) + 1;

  auto tile_w = fractal_int_info[ATTR_CONV_TILE_W];
  auto kernel_w = fractal_int_info[ATTR_CONV_KERNEL_W];
  auto stride_w = fractal_int_info[ATTR_CONV_STRIDE_W];
  auto win_w = floordiv(tile_w - kernel_w, stride_w) + 1;

  auto w = fractal_int_info[ATTR_CONV_FEATURE_W];
  auto pad_left = fractal_int_info[ATTR_CONV_PAD_LEFT];
  auto pad_right = fractal_int_info[ATTR_CONV_PAD_RIGHT];
  auto win_w_gm = floordiv(w + pad_left + pad_right - kernel_w, stride_w) + 1;

  range_map.Set("m_size", Range(Expr(0), win_h * win_w));
  range_map.Set("m_lager_size", Range(Expr(0), floordiv(win_h * win_w + 15, 16) * 16));
  range_map.Set("w_size", Range(Expr(0), win_w_gm));
}

std::string NPUIslEmitter::FindRealizeScopeToString(const isl::id &var) {
  if (info_.analysis_result_.CountBufferDefInfo(var)) {
    auto tensor_info = info_.analysis_result_.GetBufferDefInfo(var);
    MemType mem_type = tensor_info.DstMemType();

    switch (mem_type) {
      case MemType::C1_:
        if (var.get_name().find(FRACTAL_C1) != std::string::npos) return DOT_LOCAL_C1_TMP;
        return DOT_LOCAL_C1;
      case MemType::C0A_:
        return DOT_LOCAL_C0A;
      case MemType::C0B_:
        return DOT_LOCAL_C0B;
      case MemType::C0C_:
        return DOT_LOCAL_C0C;
      case MemType::BUF_:
        return DOT_LOCAL_BUF;
      case MemType::BUF_C0_:
        return DOT_LOCAL_BUF;
      case MemType::BUF_C1_:
        return DOT_LOCAL_BUF;
      default:
        LOG(FATAL) << "unexpected mem_type of var " << var;
        return "ERROR";
    }
  }
  // GEMM C_local_BUF is global
  if (var.get_name().find(LOCAL_BUF) != std::string::npos) {
    return DOT_LOCAL_BUF;
  }

  return "";
}

Expr NPUIslEmitter::FindRealizeScope(const isl::id &var) { return Expr(FindRealizeScopeToString(var)); }

Stmt NPUIslEmitter::InsertRealize(Stmt stmt, const isl::id &var, bool is_C0) {
  stmt = FindInnerRealize(var.get_name()).Mutate(stmt);

  // A tensor may be defined multiple times in BufferDefInfo due to nested realize.
  // Because we cannot determine which one we actually want, we have to be conservative here
  // and allocate space for the largest shape to avoid overflow.
  Tensor t = info_.FindTensorWithLargestShape(var);
  Region bounds;

  if (info_.mmu_info_.IsCUB(var.get_name())) {
    auto ct = info_.FindTensor(var.get_name() + LOCAL_C0C);
    for (auto j : ct->shape) {
      bounds.push_back(Range::make_by_min_extent(Expr(0), j));
    }
  }
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

  if (!info_.mmu_info_.IsIm2col()) {
    stmt = TensorStringSubstitute(stmt, t->op->func_name(), t->op, t->value_index);
  }

  // If pragma_fuse_inst is next, this realize may be inst realize. We move it inside of pragma_fuse_inst.
  if (auto attrstmt = stmt.as<AttrStmt>()) {
    if (attrstmt && attrstmt->attr_key == "pragma_fuse_vector") {
      stmt = attrstmt->body;
      stmt = Realize::make(t->op, t->value_index, t->dtype, bounds, const_true(1), stmt);
      realized_.insert(t);
      stmt = AttrStmt::make(t->op, air::ir::attr::realize_scope, FindRealizeScope(var), stmt);

      return AttrStmt::make(make_zero(Int(32)), "pragma_fuse_vector", Expr(1), stmt);
    }
  }

  if (info_.mmu_info_.IsIm2col()) {
    for (const auto &curTensor : info_.analysis_result_.GetUpdateTensor()) {
      // find the updateTensor with same name and make Realize and AttrStmt
      if (curTensor->op->name == t->op->name) {
        stmt = Realize::make(curTensor->op, t->value_index, t->dtype, bounds, const_true(1), stmt);
        realized_.insert(t);
        stmt = AttrStmt::make(curTensor->op, air::ir::attr::realize_scope, FindRealizeScope(var), stmt);
        return stmt;
      }
    }
  }
  stmt = Realize::make(t->op, t->value_index, t->dtype, bounds, const_true(1), stmt);
  realized_.insert(t);
  stmt = AttrStmt::make(t->op, air::ir::attr::realize_scope, FindRealizeScope(var), stmt);

  return stmt;
}

Stmt HoistC0write(ScopInfo &info, const Stmt &body, std::vector<Stmt> &c0write) {
  Stmt stmt = body;
  if (!c0write.empty()) {
    if (info.mmu_info_.IsGemm()) {
      auto f = HoistC0Write(info.user_config_.GetOriginBind(), c0write.back());
      static_cast<void>(f.Mutate(body));
      f.mutate_ = true;
      stmt = f.Mutate(body);
      if (!f.found_) stmt = Block::make(body, c0write.back());
      // each time we use the first stmt in c0write to construct <c0_write, c1_write> block,
      // so we need to clean vector to emit correct block for tail part.
      c0write.clear();
    } else if (info.mmu_info_.IsSpecGemm()) {
      stmt = Block::make(body, c0write.back());
    }
  }
  return stmt;
}

void NPUIslEmitter::ProcBypathC1(const ScopInfo &info) {
  if (0 == bypathC1_) {
    bypathC1_ = info.user_config_.GetByPathC1();
  }
}

Stmt NPUIslEmitter::EmitSpecGemC1write(const isl::ast_node_mark &node, const Stmt &stmt) {
  is_old_gemm_c1write_ = true;
  static_cast<void>(EmitAst(node.get_node()));
  is_old_gemm_c1write_ = false;
  if (!info_.mmu_info_.IsSpecGemm() && !info_.mmu_info_.GetOldC1Write().empty()) {
    return Block::make(stmt, info_.mmu_info_.GetOldC1Write().back());
  }
  return stmt;
}

void NPUIslEmitter::EmitAttrStmtAfterRealize(bool is_C1, bool is_C0, std::vector<Stmt> &stmts) {
  // Emit attrs of provide
  if (is_C1) {
    for (const auto &i : info_.analysis_result_.GetStmtOpInfoMap()) {
      if (!i.second.isMMU) continue;
      const Node *stmt_node = info_.analysis_result_.GetStatementMap().at(i.first);
      if (!stmt_node->IsInstance<Provide>()) continue;
      const auto provide = static_cast<const Provide *>(stmt_node);
      if (!info_.mmu_info_.GetConvAttrInfo().empty()) {
        stmts[0] = AttrStmt::make(info_.mmu_info_.GetConvAttrInfo(), "pragma_attrs", Expr(1), stmts[0]);
      } else if (const auto cop = provide->func.as<ComputeOpNode>()) {
        stmts[0] = AttrStmt::make(cop->attrs, "pragma_attrs", Expr(1), stmts[0]);
      }
      stmts[0] = AttrStmt::make(make_zero(Int(32)), "isolated_idx", Expr(tile_idx_++), stmts[0]);
      break;
    }
  }

  if (info_.mmu_info_.IsSpecGemm() && is_C0) {
    if (info_.user_config_.GetConvBackPropFilter()) {
      stmts[0] = EmitGemmRangeInfoBackPropFilter(stmts[0]);
    } else {
      stmts[0] = EmitGemmRangeInfo(stmts[0]);
    }
  }
}

void NPUIslEmitter::GemmTranspose(std::vector<Stmt> &stmts) {
  if (info_.mmu_info_.IsGemmDataTranspose()) {
    bool transBlock = !info_.mmu_info_.IsGemmDataTransposeInnerBlock();
    bool transIn = !info_.mmu_info_.IsGemmDataTransposeBlock();
    stmts[0] = TransposeLoopVarOrderInMad().Run(stmts[0], C1_LOCAL_C0A, transBlock, transIn);
  }
  if (info_.mmu_info_.IsGemmWeightTranspose()) {
    bool transBlock = !info_.mmu_info_.IsGemmWeightTransposeInnerBlock();
    bool transIn = !info_.mmu_info_.IsGemmWeightTransposeBlock();
    stmts[0] = TransposeLoopVarOrderInMad().Run(stmts[0], C1_LOCAL_C0B, transBlock, transIn);
  }
}

void NPUIslEmitter::EmitReadAttrAtC0(std::vector<Stmt> &stmts, int i, Tensor &t) {
  bool is_im2col = false;
  bool is_filter_c0 = false;
  bool is_gemm_data_trans = false;
  bool is_gemm_weight_trans = false;
  if (info_.mmu_info_.IsSpecGemm()) {
    // this case is conv gemm
    if (t->op->name.find(FRACTAL_C1_LOCAL_C0A) != std::string::npos ||
        t->op->name.find(FRACTAL_C1_LOCAL_C0B) != std::string::npos) {
      is_im2col = true;
    }

    if (t->op->name.find(LOCAL_C1_LOCAL_C0B) != std::string::npos ||
        t->op->name.find(LOCAL_C1_LOCAL_C0A) != std::string::npos) {
      is_filter_c0 = true;
    }
  } else {
    // this case is ordinary gemm
    std::string data_trans = info_.mmu_info_.ExtractStringFromAttrsAndInfo(ATTR_GEMM_DATA_TRANSPOSE);
    std::string weight_trans = info_.mmu_info_.ExtractStringFromAttrsAndInfo(ATTR_GEMM_WEIGHT_TRANSPOSE);
    size_t pos1 = t->op->name.find(C1_LOCAL_C0A);
    size_t pos2 = t->op->name.find(C1_LOCAL_C0B);
    if (data_trans == "Y" && pos1 != std::string::npos) {
      is_gemm_data_trans = true;
    }
    if (weight_trans == "Y" && pos2 != std::string::npos) {
      is_gemm_weight_trans = true;
    }

    if (bypathC1_ == 2) {
      //  left matrix by pass C1
      if (pos1 != std::string::npos) is_filter_c0 = true;
    } else if (bypathC1_ == 1) {
      // right matrix by pass C1
      if (pos2 != std::string::npos) is_filter_c0 = true;
    }
  }

  if (is_im2col) {
    stmts[i] = AttrStmt::make(make_zero(Int(INT_32)), "pragma_im2col", Expr(1), stmts[i]);
  } else if (is_gemm_data_trans) {
    stmts[i] =
      AttrStmt::make(make_zero(Int(INT_32)), "pragma_load2d_transpose_data", Expr(gemm_transpose_index_), stmts[i]);
    gemm_transpose_index_++;
    gemm_transpose_index_ = gemm_transpose_index_ % BINARY_FACTOR;
  } else if (is_gemm_weight_trans) {
    stmts[i] =
      AttrStmt::make(make_zero(Int(INT_32)), "pragma_load2d_transpose_weight", Expr(gemm_transpose_index_), stmts[i]);
    gemm_transpose_index_++;
    gemm_transpose_index_ = gemm_transpose_index_ % BINARY_FACTOR;
  }
  stmts[i] = ProducerConsumer::make(t->op, true, stmts[i]);
  if (bypathC1_ > 0) {
    if (is_filter_c0) {
      stmts[i] = AttrStmt::make(make_zero(Int(INT_32)), PRAGMA_BYPATH_FILTER_C0, Expr(0), stmts[i]);
    }
  }
}

void NPUIslEmitter::EmitReadAttrAtC1(std::vector<Stmt> &stmts, int i, Tensor &t) {
  bool is_fractal = false;
  bool is_filter_c1 = false;
  std::string fractal_str = info_.mmu_info_.ExtractStringFromAttrs(ATTR_CONV_FEATURE_NAME) + _FRACTAL_C1;
  std::string filter_str = info_.mmu_info_.ExtractStringFromAttrs(ATTR_CONV_FILTER_NAME) + LOCAL_C1;

  if (fractal_str == t->op->name) {
    is_fractal = true;
  }

  if (filter_str == t->op->name) {
    is_filter_c1 = true;
  }

  std::string data_str = info_.mmu_info_.ExtractStringFromAttrs(ATTR_CONV_GMM_FEATURE) + LOCAL_C1;
  std::string weight_str = info_.mmu_info_.ExtractStringFromAttrs(ATTR_CONV_GMM_WEIGHT) + LOCAL_C1;
  if ((bypathC1_ == 2 && data_str == t->op->name) || (bypathC1_ == 1 && weight_str == t->op->name)) {
    is_filter_c1 = true;
  }

  if (is_fractal) {
    stmts[i] = AttrStmt::make(make_zero(Int(32)), "pragma_fractal", Expr(1), stmts[i]);
  }
  stmts[i] = ProducerConsumer::make(t->op, true, stmts[i]);
  if (bypathC1_ > 0) {
    if (is_filter_c1) {
      stmts[i] = AttrStmt::make(make_zero(Int(32)), PRAGMA_BYPATH_FILTER_C1, Expr(0), stmts[i]);
    }
  }
}

void NPUIslEmitter::EmitReadAttr(const std::vector<IslIdSet> &read, std::vector<Stmt> &stmts, int i, bool is_C1,
                                 bool is_C0) {
  for (const auto &id : read[i]) {
    Tensor t = info_.FindTensor(id);
    if (is_C1) {
      EmitReadAttrAtC1(stmts, i, t);
    }

    if (is_C0) {
      EmitReadAttrAtC0(stmts, i, t);
    }
  }
}

void NPUIslEmitter::EmitWriteAttr(const std::vector<IslIdSet> &write, std::vector<Stmt> &stmts, int i, bool is_C1) {
  for (const auto &id : write[i]) {
    if (is_C1 && info_.mmu_info_.IsCUB(id.get_name())) continue;
    if (is_old_gemm_c1write_ && info_.mmu_info_.IsC(id.get_name())) {
      stmts[i] = AttrStmt::make(make_zero(Int(32)), PRAGMA_MMU_C1WRITE, Expr(1), stmts[i]);
      info_.mmu_info_.OldC1WriteInsert(stmts[i]);
    }
    if (info_.mmu_info_.IsSpecGemm() && info_.mmu_info_.IsC(id.get_name())) {
      stmts[i] = AttrStmt::make(make_zero(Int(32)), PRAGMA_MMU_C0WRITE, Expr(1), stmts[i]);
      mmu_c0write_.emplace_back(stmts[i]);
      stmts[i] = Evaluate::make(0);
    }
    if (info_.mmu_info_.IsGemm() && !info_.mmu_info_.IsSpecGemm() && info_.mmu_info_.IsCUB(id.get_name())) {
      stmts[i] = AttrStmt::make(make_zero(Int(32)), PRAGMA_MMU_C0WRITE, Expr(1), stmts[i]);
      mmu_c0write_.emplace_back(stmts[i]);
      stmts[i] = Evaluate::make(0);
    }
    if (info_.mmu_info_.IsGemm() && !info_.mmu_info_.IsSpecGemm() && info_.mmu_info_.IsC(id.get_name())) {
      stmts[i] = AttrStmt::make(make_zero(Int(32)), PRAGMA_MMU_C1WRITE, Expr(1), stmts[i]);
      if (!mmu_c0write_.empty()) {
        mmu_c0write_.emplace_back(Block::make(mmu_c0write_[0], stmts[i]));
        stmts[i] = Evaluate::make(0);
      }
    }
  }
}

void NPUIslEmitter::EmitAttrStmt(const isl::ast_node_block &block_node, const Liveness &liveness, bool is_C1,
                                 bool is_C0, std::vector<Stmt> &stmts) {
  for (unsigned int i = 0; i < block_node.get_children().size(); ++i) {
    EmitReadAttr(liveness.read_, stmts, i, is_C1, is_C0);
    EmitWriteAttr(liveness.write_, stmts, i, is_C1);
  }
}

void NPUIslEmitter::CollectLiveness(const Liveness &liveness_info, bool is_C1, std::vector<IslIdSet> &real,
                                    std::unordered_map<isl::id, std::set<int>, isl::IslIdIslHash> &liveness,
                                    std::function<bool(const std::string &id)> const &CheckGoOut) {
  for (unsigned int i = 0; i < liveness_info.read_.size(); i++) {
    IslIdSet idset;
    real.push_back(idset);
    auto v = static_cast<int>(i);
    // add read
    for (const auto &j : liveness_info.read_[i]) {
      if (!liveness.count(j)) {
        std::set<int> jset;
        liveness.emplace(j, jset);
      }
      liveness.at(j).insert(v);
    }

    // add from inner out
    for (const auto &j : liveness_info.out_[i]) {
      if (!liveness.count(j)) {
        std::set<int> jset;
        liveness.emplace(j, jset);
      }
      liveness.at(j).insert(v);
    }

    // add vectors' def
    for (const auto &j : liveness_info.may_def_[i]) {
      // for i
      //   if i == 0
      //     C=0     // def C
      //   for j
      //     C=C+A*B  // WAR C
      // cause we do not know C=0 cannot dominate the following WAR
      // The C' liveness will cover the whole loops and realize out of loops.
      // Now we just judge whole loop's liveness from existing WAR.
      // It is correct in gemm, conv etc. but may be wrong in other cases.

      std::string tensor_name = info_.GetOriginTensorId(j).get_name();
      if (info_.MayWriteAfterRead(tensor_name) && CheckGoOut(j.get_name())) {
        realize_out_.insert(j);
      }
      if (!liveness.count(j)) {
        std::set<int> jset;
        liveness.emplace(j, jset);
      }
      liveness.at(j).insert(v);
    }
    for (const auto &j : liveness_info.write_[i]) {
      if (!info_.IsInBinds(j) && CheckGoOut(j.get_name())) realize_out_.insert(j);
    }

    // isolated part, may reuse def in full tile. We realize them out
    for (const auto &j : liveness_info.use_with_may_def_[i]) {
      if (CheckGoOut(j.get_name())) global_realize_out_.insert(j);
    }
  }
  /// amazing and fusing control: which should be realized out
  if (is_C1) realize_out_.clear();

  for (const auto &i : liveness) {
    if (realize_out_.count(i.first)) continue;
    real[(unsigned int)*i.second.begin()].insert(i.first);
  }
}

// add realize
// so far we do not think about flow analysis for liveness
// we hack gemm C+=A*B and make C's liveness in the whole loop
void NPUIslEmitter::EmitRealize(const isl::ast_node_block &block_node, const Liveness &liveness_info, bool is_C1,
                                bool is_C0, std::vector<Stmt> &stmts) {
  auto c_buf = info_.mmu_info_.IsSpecGemm() ? info_.mmu_info_.GetCName() : info_.mmu_info_.GetCName() + LOCAL_BUF;
  auto c_c0c = c_buf + LOCAL_C0C;
  auto CheckGoOut = [&c_buf, &c_c0c](const std::string &id) -> bool { return !(id == c_buf || id == c_c0c); };

  std::vector<IslIdSet> real;
  std::unordered_map<isl::id, std::set<int>, isl::IslIdIslHash> liveness;
  CollectLiveness(liveness_info, is_C1, real, liveness, CheckGoOut);

  size_t last = block_node.get_children().size() - 1;
  for (const auto &var : real[last]) {
    /// so far our alloc_C is only designed for specgemm
    if (info_.mmu_info_.IsSpecGemm() || info_.mmu_info_.IsConv()) {
      if (!CheckGoOut(var.get_name())) continue;
    }

    stmts[last] = InsertRealize(stmts[last], var, is_C0);
  }

  for (int i = block_node.get_children().size() - 2; i >= 0; --i) {
    auto p = static_cast<unsigned int>(i);
    stmts[p] = Block::make(stmts[p], stmts[p + 1]);

    for (const auto &var : real[p]) {
      /// so far our alloc_C is only designed for specgemm
      if (info_.mmu_info_.IsSpecGemm() || info_.mmu_info_.IsConv()) {
        if (!CheckGoOut(var.get_name())) continue;
      }

      stmts[p] = InsertRealize(stmts[p], var, is_C0);

      if (!DELETE_FRACTAL) continue;
      std::string feature_str = info_.mmu_info_.ExtractStringFromAttrs(ATTR_CONV_FEATURE_NAME) + LOCAL_C1;
      if (feature_str == var.get_name()) {
        std::string fractal_str = info_.mmu_info_.ExtractStringFromAttrs(ATTR_CONV_FEATURE_NAME) + _FRACTAL_C1;
        stmts[p] = InsertRealize(stmts[p], isl::id(var.ctx(), fractal_str), is_C0);
      }
    }
  }
}

Stmt NPUIslEmitter::EmitBlock(const isl::ast_node_block &block_node) {
  if (!args_) return IslEmitter::EmitBlock(block_node);
  bool is_C0 = *static_cast<const bool *>(args_);
  bool is_C1 = *(static_cast<const bool *>(args_) + 1);
  args_ = nullptr;

  realize_must_def_.clear();
  realize_may_def_.clear();
  realize_use_.clear();
  for (const auto &i : realize_out_) {
    global_realize_out_.insert(i);
  }
  realize_out_.clear();
  hoisted_read_.clear();
  hoisted_write_.clear();

  std::vector<Stmt> stmts;
  auto liveness = Liveness();

  // collect info for each child node
  for (auto child : block_node.get_children()) {
    is_mmu_ = false;
    Stmt body = (EmitAst(child));

    if (is_mmu_) {
      body = MadMarker().Run(body);
      body = IfThenElseSplitter().Run(body);
      opinfo_.ops.clear();
      opinfo_.isMMU = false;
    }
    stmts.push_back(body);

    liveness.may_def_.push_back(realize_may_def_);
    liveness.must_def_.push_back(realize_must_def_);
    liveness.use_.push_back(realize_use_);
    liveness.use_with_may_def_.push_back(realize_use_with_may_def_);
    liveness.out_.push_back(realize_out_);
    liveness.read_.push_back(hoisted_read_);
    liveness.write_.push_back(hoisted_write_);

    realize_must_def_.clear();
    realize_may_def_.clear();
    realize_use_.clear();
    realize_out_.clear();
    hoisted_read_.clear();
    hoisted_write_.clear();
  }

  EmitAttrStmt(block_node, liveness, is_C1, is_C0, stmts);
  EmitRealize(block_node, liveness, is_C1, is_C0, stmts);
  EmitAttrStmtAfterRealize(is_C1, is_C0, stmts);
  GemmTranspose(stmts);
  args_ = nullptr;
  return stmts[0];
}

void NPUIslEmitter::ConvBackPropFilterFixMadInit(const isl::ast_node_mark &node, Expr &mad_init_cond) {
  if (info_.mmu_info_.IsConvBackpropFilter()) {
    /// find reduce k;
    /// correct axles' name
    FindStmt fs = FindStmt();
    fs.FindAst(node.get_node());

    for (const auto &i : fs.usernodes) {
      CHECK(i.get_expr().isa<isl::ast_expr_op>());
      isl::ast_expr_op usr_expr = i.get_expr().as<isl::ast_expr_op>();
      CHECK(usr_expr.get_arg(0).isa<isl::ast_expr_id>());
      isl::id curstmtid = usr_expr.get_arg(0).as<isl::ast_expr_id>().get_id();
      isl::id curnodeid = i.get_annotation();
      const Node *stmt_node = info_.analysis_result_.GetStatementMap().at(curstmtid);
      CHECK(stmt_node != nullptr);

      // stmt_node should not have if stmt
      if (stmt_node->IsInstance<Provide>()) {
        auto build = node_info_map_.at(curnodeid).build;
        auto tuple = info_.analysis_result_.GetOperatorDomainMap().at(curstmtid).tuple;
        auto iterator_map = node_info_map_.at(curnodeid).iterator_map;

        for (unsigned int n = 0; n < tuple.size(); n++) {
          isl::id isl_old_iter = tuple.get_id(n);
          bool is_red = false;
          for (const auto &reds : info_.analysis_result_.GetReduceMap()) {
            for (auto j : reds.second) {
              // when support atomic add, "no" should not init in each core
              if (isl_old_iter.get_name() == j->var->name_hint && isl_old_iter.get_name() != "no") {
                is_red = true;
                break;
              }
            }
          }
          if (!is_red) continue;
          auto isl_expr = build.expr_from(iterator_map.get_pw_aff(n));
          Expr halide_new_iter = Interpret(isl_expr);
          std::vector<const Variable *> vv = ExtractIterfromExpr().Run(halide_new_iter);

          for (auto v : vv) {
            if (std::find(iters_.begin(), iters_.end(), v) == iters_.end()) continue;
            if (mad_init_cond.defined()) {
              mad_init_cond = And::make(mad_init_cond, EQ::make(Expr(GetObjPtr(v)), Expr(0)));
            } else {
              mad_init_cond = EQ::make(Expr(GetObjPtr(v)), Expr(0));
            }
          }
        }
      } else {
        LOG(WARNING) << "stmt_node has if stmt";
      }
    }
  }
}

Stmt NPUIslEmitter::EmitMarkFuseInst(const isl::ast_node_mark &node) {
  auto stmt = AttrStmt::make(make_zero(Int(32)), "pragma_fuse_vector", Expr(1), EmitAst(node.get_node()));
  if (info_.mmu_info_.IsGemm() && !info_.mmu_info_.IsSpecGemm() && !mmu_c0write_.empty()) {
    mmu_c0write_.emplace_back(Block::make(mmu_c0write_[0], stmt));
    stmt = Evaluate::make(0);
  }
  return stmt;
}

Stmt NPUIslEmitter::EmitMarkAllocRealizeOut(const isl::ast_node_mark &node) {
  Stmt body = EmitAst(node.get_node());
  for (const auto &i : realize_out_) {
    body = InsertRealize(body, i, false);
  }
  realize_out_.clear();
  body = AttrStmt::make(make_zero(Int(32)), ALLOC_REALIZE_OUT, Expr(1), body);
  return body;
}

Stmt NPUIslEmitter::EmitMarkAllocC(const isl::ast_node_mark &node) {
  Stmt body = EmitAst(node.get_node());
  body = RemoveNoOp(body);
  body = HoistC0write(info_, body, mmu_c0write_);

  auto c_buf = info_.mmu_info_.IsSpecGemm() ? info_.mmu_info_.GetCName() : info_.mmu_info_.GetCName() + LOCAL_BUF;
  auto c_c0c = c_buf + LOCAL_C0C;
  body = InsertRealize(body, isl::id(info_.GetCtx(), c_c0c), false);
  body = InsertRealize(body, isl::id(info_.GetCtx(), c_buf), false);
  body = AttrStmt::make(make_zero(Int(32)), ALLOC_C, Expr(1), body);
  return body;
}

Stmt NPUIslEmitter::EmitMarkSpecGemm(const isl::ast_node_mark &node) {
  info_.mmu_info_.UpdateFractalIntInfo(++isolate_idx_);
  Expr mad_init_cond;
  ConvBackPropFilterFixMadInit(node, mad_init_cond);
  if (info_.mmu_info_.GetOutReduceInit() == 0) {
    mad_init_cond = Expr(0);
  }
  Stmt stmt = SpecGemmBuilder(info_).Build(mad_init_cond);
  return EmitSpecGemC1write(node, stmt);
}

Stmt NPUIslEmitter::EmitMark(const isl::ast_node_mark &node) {
  auto original_multicore_info = multicore_info;

  std::string mark = node.get_id().get_name();
  bool is_outer_band = (mark.find("multicore_coincident_") == 0);
  if (is_outer_band) {
    multicore_info.enabled = true;
    size_t coinLen = strlen("multicore_coincident_");
    CHECK_GE(mark.size(), coinLen);
    std::string coincidence_str = mark.substr(coinLen);
    multicore_info.coincidence = SplitString(coincidence_str, "_");
    multicore_info.id = ++multicore_info_count_;
    CHECK_GT(multicore_info.coincidence.size(), 0) << "invalid multicore mark: " << mark;
  }

  bool is_inner_band = (mark.find("realize") == 0);
  if (is_inner_band) {
    multicore_info.enabled = false;
  }

  Stmt stmt = EmitMarkMulticore(node);

  multicore_info = original_multicore_info;
  return stmt;
}

void NPUIslEmitter::RealizeOut() {
  // add vectors' def
  for (const auto &j : realize_must_def_) {
    // we lack CFG or SSA here. Such as Gemm,
    // for i
    //   if i == 0
    //     C=0     // def C
    //   for j
    //     C=C+A*B  // WAR C
    // cause we do not know C=0 cannot dominate the following WAR
    // The C' liveness will cover the whole loops and realize out of loops.
    // Now we just judge whole loop's liveness from existing WAR.
    // It is correct in gemm, conv etc. but may be wrong in other cases.

    std::string tensor_name = info_.GetOriginTensorId(j).get_name();
    if (info_.MayWriteAfterRead(tensor_name)) {
      bool do_out = true;
      auto c_buf = info_.mmu_info_.IsSpecGemm() ? info_.mmu_info_.GetCName() : info_.mmu_info_.GetCName() + LOCAL_BUF;
      auto c_c0c = c_buf + LOCAL_C0C;
      if (j.get_name() == c_buf || j.get_name() == c_c0c) {
        do_out = false;
      }
      if (do_out) realize_out_.insert(j);
    }
  }

  /// isolated part, may reuse def in full tile. We realize them out
  for (const auto &j : realize_use_with_may_def_) {
    global_realize_out_.insert(j);
  }

  realize_must_def_.clear();
  realize_may_def_.clear();
  realize_use_.clear();
  realize_use_with_may_def_.clear();
  realize_out_.clear();
  hoisted_read_.clear();
  hoisted_write_.clear();
};

Stmt NPUIslEmitter::EmitMarkMulticore(const isl::ast_node_mark &node) {
  auto mark_name = node.get_id().get_name();
  if (mark_name == FUSE_VECTOR) return EmitMarkFuseInst(node);
  if (mark_name == ALLOC_REALIZE_OUT) return EmitMarkAllocRealizeOut(node);
  if (mark_name == ALLOC_C) return EmitMarkAllocC(node);
#if SPEC_GEMM
  if (mark_name == CONV_GEMM) return EmitMarkSpecGemm(node);
#endif
  if (node.get_node().as<isl::ast_node_mark>()) return EmitAst(node.get_node());
  if (node.get_node().as<isl::ast_node_for>()) {
    is_mmu_ = false;
    return EmitAst(node.get_node());
  }

  if (auto block_node = node.get_node().as<isl::ast_node_block>()) {
    bool is_C0 = (node.get_id().get_name() == REALIZE_C0);
    bool is_C1 = (node.get_id().get_name() == REALIZE_C1);
    bool is_BUF = (node.get_id().get_name() == REALIZE_BUF);
    std::unique_ptr<bool[]> args_tmp(new bool[3]{is_C0, is_C1, is_BUF});
    args_ = args_tmp.get();
    return EmitBlock(block_node);
  } else if (node.get_node().as<isl::ast_node_if>()) {
    auto stmt = EmitAst(node.get_node());
    for (const auto &var : realize_must_def_) {
      Tensor t = info_.FindTensor(var);
      Region bounds;
      for (auto j : t->shape) {
        bounds.push_back(Range::make_by_min_extent(Expr(0), j));
      }
      stmt = Realize::make(t->op, t->value_index, t->dtype, bounds, const_true(1), stmt);
      stmt = AttrStmt::make(t->op, air::ir::attr::realize_scope, Expr(DOT_LOCAL_C1), stmt);
    }
    return stmt;
  } else {
    is_mmu_ = false;
    Stmt body = EmitAst(node.get_node());
    if (is_mmu_) {
      body = MadMarker().Run(body);
      body = IfThenElseSplitter().Run(body);
      opinfo_.ops.clear();
      opinfo_.isMMU = false;
    }
    RealizeOut();
    return body;
  }
  return EmitAst(node.get_node());
}

class FindUsingTensor : public IRVisitor {
 public:
  explicit FindUsingTensor(Stmt &stmt) : stmt_(stmt) {}
  ~FindUsingTensor() override = default;

  void Visit_(const Call *op) final {
    if (op->call_type == Call::Halide && op->func->func_name() == name_) {
      found_ = true;
    }
    IRVisitor::Visit_(op);
  }

  void Visit_(const Provide *op) final {
    if (op->func->func_name() == name_) {
      found_ = true;
    }
    IRVisitor::Visit_(op);
  }

  bool found(const std::string &name) {
    name_ = name;
    IRVisitor::Visit(stmt_);
    return found_;
  }

 private:
  Stmt &stmt_;
  std::string name_;
  bool found_{false};
};

class FindNotRealizedTensors : public air::ir::IRVisitor {
 public:
  void Visit_(const Call *op) final {
    if (op->call_type == Call::Halide && realized_in_scope.count(op->func->func_name()) == 0) {
      non_realized_.insert(op->func->func_name());
    }
    IRVisitor::Visit_(op);
  }

  void Visit_(const Provide *op) final {
    if (realized_in_scope.count(op->func->func_name()) == 0) {
      non_realized_.insert(op->func->func_name());
    }
    IRVisitor::Visit_(op);
  }

  void Visit_(const Realize *op) final {
    std::string realize_var = op->func->func_name();
    realized_in_scope.insert(realize_var);
    IRVisitor::Visit_(op);
    realized_in_scope.erase(realize_var);
  }

  void Visit_(const AttrStmt *op) final {
    if (op->attr_key == "pragma_emit_insn" && op->value.as<StringImm>() && op->value.as<StringImm>()->value == "mad") {
      has_mmu_ = true;
    }
    IRVisitor::Visit_(op);
  }

  std::unordered_set<std::string> Find(const Stmt &stmt) {
    non_realized_.clear();
    realized_in_scope.clear();
    has_mmu_ = false;
    Visit(stmt);
    if (has_mmu_) {
      non_realized_.clear();
    }
    return non_realized_;
  }

 private:
  std::unordered_set<std::string> non_realized_;
  std::unordered_set<std::string> realized_in_scope;
  bool has_mmu_{false};
};

class RmCondwithVar : public IRMutator {
 public:
  explicit RmCondwithVar(std::set<const Variable *> &rmif) : rmif_(rmif) {}
  ~RmCondwithVar() override = default;
  Expr Mutate_(const And *op, const Expr &e) final {
    Expr a = this->Mutate(op->a);
    Expr b = this->Mutate(op->b);
    if (!a.defined() && !b.defined()) return Expr();
    if (!a.defined()) return b;
    if (!b.defined()) return a;
    return IRMutator::Mutate_(op, e);
  }
  Expr Mutate_(const Or *op, const Expr &e) final {
    Expr a = this->Mutate(op->a);
    Expr b = this->Mutate(op->b);
    if (!a.defined() && !b.defined()) return Expr();
    if (!a.defined()) return b;
    if (!b.defined()) return a;
    return IRMutator::Mutate_(op, e);
  }
  Expr Mutate_(const EQ *op, const Expr &e) final {
    Expr a = this->Mutate(op->a);
    Expr b = this->Mutate(op->b);
    CHECK(a.defined());
    const auto v = static_cast<const Variable *>(a.get());
    if (v && rmif_.count(v)) {
      for (auto i : rmcond_) {
        if (i.same_as(e)) return Expr();
      }
      rmcond_.push_back(e);
      return Expr();
    }
    CHECK(b.defined());
    return (a.same_as(op->a) && b.same_as(op->b)) ? e : EQ::make(a, b);
  }

  std::set<const Variable *> rmif_;
  Array<Expr> rmcond_;
};

class RmCond : public IRMutator {
 public:
  explicit RmCond(std::set<const Variable *> &rmif) : rmif_(rmif) {}
  ~RmCond() override = default;

  Stmt Mutate_(const IfThenElse *op, const Stmt &s) final {
    Expr condition = this->Mutate(op->condition);
    // judge rm if for atomic add
    condition = Simplify(condition);
    auto rmcond = RmCondwithVar(rmif_);
    condition = rmcond.Mutate(condition);
    if (!condition.defined()) condition = Expr(1);
    condition = Simplify(condition);

    Stmt then_case = this->Mutate(op->then_case);
    Stmt else_case;
    if (op->else_case.defined()) else_case = this->Mutate(op->else_case);
    if (condition.same_as(op->condition) && then_case.same_as(op->then_case) && else_case.same_as(op->else_case))
      return s;
    auto res = IfThenElse::make(condition, then_case, else_case);
    if (!rmcond.rmcond_.empty()) res = AttrStmt::make(rmcond.rmcond_, ATOMIC_COND_CLEAN, Expr(1), res);
    return res;
  }

  std::set<const Variable *> rmif_;
};

Stmt NPUIslEmitter::RemoveCond(const Stmt &stmt) { return RmCond(rmif_).Mutate(stmt); }

/*
 * Sink realize inside multi-core "For" statements.
 *
 * Example before mutate:
 *
 * // attr [placeholder(input_B_grad_local_BUF, 0x1a00220)] realize_scope = "local.BUF"
 * realize input_B_grad_local_BUF<float16>([0, 5087]) {
 *   // attr [0] pragma_multi_core_depth = 1
 *   for (cc0, 0, 6) {
 *     for (cc1, 0, 640) { ... }
 *   }
 * }
 *
 * After mutate:
 *
 * // attr [0] pragma_multi_core_depth = 1
 * for (cc0, 0, 6) {
 *   // attr [placeholder(input_B_grad_local_BUF, 0x1a00220)] realize_scope = "local.BUF"
 *   realize input_B_grad_local_BUF<float16>([0, 5087]) {
 *     for (cc1, 0, 640) { ... }
 *   }
 * }
 */
class SinkRealizeInsideMulticore : public IRMutator {
 private:
  static Stmt InsertRealize(const Realize *op, const Stmt &body) {
    return Realize::make(op->func, op->value_index, op->type, op->bounds, op->condition, body);
  }

  static Stmt InsertAttrStmt(const AttrStmt *op, const Stmt &body) {
    return AttrStmt::make(op->node, op->attr_key, op->value, body);
  }

  void RealizeTensorHere(const std::string &tensor_name, Stmt &stmt) {
    stmt = InsertRealize(realized_map_[tensor_name], stmt);
    if (realize_attr_map_.count(tensor_name) > 0) {
      stmt = InsertAttrStmt(realize_attr_map_[tensor_name], stmt);
    }
  }

  void RealizeAllOuterTensorsHere(Stmt &stmt) {
    for (const auto &tensor : realized_map_) {
      RealizeTensorHere(tensor.first, stmt);
    }
    is_realized_ = true;
  }

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) override {
    if (auto realize_op = op->body.as<Realize>()) {
      std::string tensor_name = realize_op->func->func_name();
      CHECK_EQ(realize_attr_map_.count(tensor_name), 0);
      realize_attr_map_.emplace(tensor_name, op);

      Stmt stmt = Mutate(op->body);

      realize_attr_map_.erase(tensor_name);
      return stmt;
    } else if (auto for_op = op->body.as<For>()) {
      if (op->attr_key == "pragma_multi_core_depth") {
        Stmt for_body = IRMutator::Mutate(for_op->body);
        Stmt for_stmt =
          For::make(for_op->loop_var, for_op->min, for_op->extent, for_op->for_type, for_op->device_api, for_body);
        return AttrStmt::make(op->node, op->attr_key, op->value, for_stmt);
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Realize *op, const Stmt &s) final {
    std::string tensor_name = op->func->func_name();
    CHECK_EQ(realized_map_.count(tensor_name), 0);
    realized_map_.emplace(tensor_name, op);

    Stmt stmt = Mutate(op->body);

    if (!is_realized_) RealizeTensorHere(tensor_name, stmt);

    realized_map_.erase(tensor_name);
    return stmt;
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    Stmt original = For::make(op->loop_var, op->min, op->extent, op->for_type, op->device_api, op->body);
    RealizeAllOuterTensorsHere(original);
    return original;
  }

  Stmt Mutate_(const Block *op, const Stmt &s) final {
    Stmt original = Block::make(op->first, op->rest);
    RealizeAllOuterTensorsHere(original);
    return original;
  }

  bool is_realized_{false};
  std::unordered_map<std::string, const Realize *> realized_map_;
  std::unordered_map<std::string, const AttrStmt *> realize_attr_map_;
};

std::unordered_set<FunctionRef, NodeHash, NodeEqual> GatherTensors(const NodeRef &stmt) {
  std::unordered_set<FunctionRef, NodeHash, NodeEqual> tensors;
  PostOrderVisit(stmt, [&](const NodeRef &node) -> void {
    if (auto call = node.as<Call>()) {
      if (call->call_type == Call::CallType::Halide) {
        tensors.insert(call->func);
      }
    }
  });
  return tensors;
}

/*
 * Check the expr is constant, or has the form K * var + C, where K and C are constants
 */
bool IsLinearExprOfOneVar(const Expr &arg) {
  std::unordered_set<Var, NodeHash, NodeEqual> vars;
  GatherVars(arg, &vars);
  if (vars.empty()) {
    return isImm(arg);
  } else if (vars.size() >= 2) {
    return false;
  } else {
    Array<Var> array_vars;
    for (const auto &var : vars) array_vars.push_back(var);
    auto coefs = air::arith::DetectLinearEquation(arg, array_vars);
    bool is_linear_equation = (coefs.size() == 2);
    return is_linear_equation;
  }
}

unsigned GetBoundedInnerLoops(const std::vector<const For *> &loops,
                              const std::unordered_set<const Variable *> &free_loop_vars) {
  for (unsigned num_loops = 0; num_loops < loops.size(); ++num_loops) {
    auto curr_loop = loops[loops.size() - num_loops - 1];
    if (free_loop_vars.count(curr_loop->loop_var.get())) return num_loops;
  }
  return loops.size();
}

/*
 * Returns the number of innermost loops that do not access overlapping tensor indexes (args).
 */
unsigned GetInnerLoopsWithoutSelfDependence(const Array<Expr> &args, const std::vector<const For *> &loops) {
  std::unordered_set<const Variable *> free_loop_vars;
  for (const auto loop : loops) {
    free_loop_vars.insert(loop->loop_var.get());
  }
  for (const auto &arg : args) {
    if (!IsLinearExprOfOneVar(arg)) return 0;
    std::unordered_set<Var, ObjectHash, ObjectEqual> vars_in_arg;
    GatherVars(arg, &vars_in_arg);
    CHECK_LE(vars_in_arg.size(), 1);
    if (!vars_in_arg.empty()) {
      free_loop_vars.erase(vars_in_arg.begin()->get());
    }
  }
  return GetBoundedInnerLoops(loops, free_loop_vars);
}

/*
 * Specialized loop distribution for the following code:
 * (It is generated by npu_isl_emitter, so reschedule cannot distribute the loops.)
 *
 *   for (i, 0, 20)
 *     for (j, 0, 10)
 *       if (cond) {
 *         compute_local_BUF(0, j) = ...
 *         compute(10 * i + j) = compute_local_BUF(0, j)
 *       }
 *
 * --->
 *
 *   for (i, 0, 20) {
 *     for (j, 0, 10)
 *       if (cond)
 *         compute_local_BUF(0, j) = ...
 *     for (j, 0, 10)
 *       if (cond)
 *         compute(10 * i + j) = compute_local_BUF(0, j)
 *   }
 *
 * Match the following code structure:
 * nested For -> If -> a Block of two Provides S_1, S_2
 * (AttrStmt and other types of stmts are not allowed.)
 *
 * S_1: write a tensor with loop vars or constants as indexes
 * S_2: read the tensor in S_1 with the same indexes as S_1
 *
 * Additional check:
 * Both S_1 and S_2 lhs tensor does not appear in the value of S_1 or the if condition.
 */
class SpecialLoopDistribution : public IRMutator {
 private:
  static bool AllCallsHaveSameIndexes(const NodeRef &stmt, const NodeRef &call_func, const Array<Expr> &args) {
    bool has_mismatch = false;
    PostOrderVisit(stmt, [&](const NodeRef &node) -> void {
      auto call = node.as<Call>();
      if (call && call->call_type == Call::CallType::Halide && call->func == call_func) {
        if (args.size() != call->args.size()) {
          has_mismatch = true;
          return;
        }
        for (unsigned i = 0; i < args.size(); ++i) {
          if (!Equal(call->args[i], args[i])) {
            has_mismatch = true;
            return;
          }
        }
      }
    });
    return !has_mismatch;
  }

  unsigned DetermineNumLoopsToDistribute(const Stmt &stmt) {
    auto op = stmt.as<IfThenElse>();
    CHECK(op != nullptr);
    unsigned ret = 0;
    if (op->else_case.defined()) return ret;

    if (!op->then_case.defined()) return ret;
    auto block = op->then_case.as<Block>();
    if (block == nullptr) return ret;

    auto first = block->first.as<Provide>();
    if (first == nullptr) return ret;

    auto second = block->rest.as<Provide>();
    if (second == nullptr) return ret;
    auto tensors_in_cond = GatherTensors(op->condition);
    auto tensors_in_first = GatherTensors(first->value);
    auto tensors_in_second = GatherTensors(second->value);

    bool lhs_tensor_not_equal = first->func != second->func;
    bool s1_lhs_tensor_accessed_in_s2_rhs = tensors_in_second.count(first->func);
    bool rw_tensor_have_same_indexes = AllCallsHaveSameIndexes(second->value, first->func, first->args);
    bool lhs_tensor_not_in_s1_value = !tensors_in_first.count(first->func) && !tensors_in_first.count(second->func);
    bool lhs_tensor_not_in_if_cond = !tensors_in_cond.count(first->func) && !tensors_in_cond.count(second->func);
    if (lhs_tensor_not_equal && s1_lhs_tensor_accessed_in_s2_rhs && rw_tensor_have_same_indexes &&
        lhs_tensor_not_in_s1_value && lhs_tensor_not_in_if_cond) {
      ret = GetInnerLoopsWithoutSelfDependence(first->args, to_distribued_loops_);
    }
    return ret;
  }

  Stmt IfStmtLoopDistribution(const Stmt &stmt) {
    unsigned num_inner_loops = DetermineNumLoopsToDistribute(stmt);
    if (num_inner_loops == 0) {
      auto outer_loops = to_distribued_loops_;
      to_distribued_loops_.clear();
      Stmt new_body = Mutate(stmt);
      to_distribued_loops_ = outer_loops;  // WrapOuterLoops use "loops" var
      return WrapOuterLoops(num_inner_loops, new_body);
    } else {
      auto op = stmt.as<IfThenElse>();
      CHECK(op != nullptr);
      auto block = op->then_case.as<Block>();
      CHECK(block != nullptr);
      Stmt outer_block = DistributeLoops(num_inner_loops, op->condition, block->first, block->rest);
      return WrapOuterLoops(num_inner_loops, outer_block);
    }
  }

  Stmt DistributeLoops(unsigned num_loops_to_distribute, const Expr &condition, const Stmt &first, const Stmt &second) {
    Stmt first_block = IfThenElse::make(condition, first, Stmt());
    for (unsigned i = 0; i < num_loops_to_distribute; ++i) {
      auto curr_for = to_distribued_loops_[to_distribued_loops_.size() - i - 1];
      first_block = MakeForStmt(curr_for, first_block);
    }
    Stmt second_block = IfThenElse::make(condition, second, Stmt());
    for (unsigned i = 0; i < num_loops_to_distribute; ++i) {
      auto curr_for = to_distribued_loops_[to_distribued_loops_.size() - i - 1];
      second_block = MakeForStmt(curr_for, second_block);
    }
    return Block::make(first_block, second_block);
  }

  Stmt WrapOuterLoops(unsigned num_loops_to_distribute, const Stmt &body) {
    Stmt outer_block = body;
    for (unsigned i = num_loops_to_distribute; i < to_distribued_loops_.size(); ++i) {
      auto curr_for = to_distribued_loops_[to_distribued_loops_.size() - i - 1];
      outer_block = MakeForStmt(curr_for, outer_block);
    }
    return outer_block;
  }

  static Stmt MakeForStmt(const For *curr_for, const Stmt &body) {
    return For::make(curr_for->loop_var, curr_for->min, curr_for->extent, curr_for->for_type, curr_for->device_api,
                     body);
  }

  /*
   * Distribute as many innermost loops as possible.
   * Return statement is wrapped by all loops in "loops".
   */
  Stmt Mutate_(const For *op, const Stmt &s) override {
    auto backup_loops = to_distribued_loops_;
    to_distribued_loops_.push_back(op);
    Stmt stmt;
    if (op->body.as<IfThenElse>()) {
      stmt = IfStmtLoopDistribution(op->body);
    } else if (op->body.as<For>()) {
      stmt = IRMutator::Mutate(op->body);
    } else {
      to_distribued_loops_.clear();
      stmt = IRMutator::Mutate_(op, s);
      to_distribued_loops_ = backup_loops;
      stmt = WrapOuterLoops(0, stmt);
    }
    to_distribued_loops_ = backup_loops;
    return stmt;
  }

  std::vector<const For *> to_distribued_loops_;
};

Stmt NPUIslEmitter::Emit(const isl::ast_node &node) {
  Stmt stmt = EmitAst(node);
  stmt = RemoveCond(stmt);

  /// emit global realize
  if (!info_.mmu_info_.IsSpecGemm()) {
    for (const auto &i : global_realize_out_) {
      Tensor t = info_.FindTensor(i);
      if (realized_.count(t)) continue;
      stmt = InsertRealize(stmt, i, false);
    }

    for (const auto &i : realize_out_) {
      Tensor t = info_.FindTensor(i);
      if (realized_.count(t)) continue;
      stmt = InsertRealize(stmt, i, false);
    }

    auto realize_from_input = info_.user_config_.GetRealizeFromInput();
    for (const auto &i : realize_from_input) {
      if (FindUsingTensor(stmt).found(i.get_name()) && !info_.IsInBinds(i.get_name())) {
        stmt = InsertRealize(stmt, i, false);
      }
    }

    auto not_realized_tensors = FindNotRealizedTensors().Find(stmt);
    for (const auto &not_realized_tensor : not_realized_tensors) {
      isl::id var = isl::id(info_.GetCtx(), not_realized_tensor);
      if (!FindRealizeScopeToString(var).empty()) {
        // The tensor needs to be realized somewhere, but it is not realized in the correct scope.
        // So, we realize it in the outermost scope to fix it.
        stmt = InsertRealize(stmt, var, false);
      } else {
        // The tensor is a global var, no need to realize.
      }
    }

    stmt = SinkRealizeInsideMulticore().Mutate(stmt);
    stmt = SpecialLoopDistribution().Mutate(stmt);
  }

  std::unordered_map<const Variable *, VarExpr> vmap;
  for (const auto &i : iters_new_name_) {
    vmap.emplace(i.first, VarExpr(i.second));
  }
  return stmt;
}

void GetNameWithoutLocal(isl::id &tensor_id, ScopInfo &info) {
  if (!info.mmu_info_.IsSpecGemm()) {
    size_t pos = tensor_id.get_name().find("_local_");
    std::string substr = tensor_id.get_name().substr(0, pos);
    if (pos != 0) tensor_id = isl::id(tensor_id.ctx(), substr);
  }
}

isl::multi_aff NPUIslEmitter::TensorAccessMultAff(isl::id &tensor_id, const Array<Expr> &tensor_index,
                                                  const isl::id &node_id) {
  GetNameWithoutLocal(tensor_id, info_);
  return IslEmitter::TensorAccessMultAff(tensor_id, tensor_index, node_id);
}

bool NPUIslEmitter::IsCopyinFromAnotherBand(isl::multi_aff &access) {
  if (!info_.mmu_info_.IsSpecGemm()) {
    return IslEmitter::IsCopyinFromAnotherBand(access);
  }
  return false;
}

bool NPUIslEmitter::IsTransferStmt() {
  if (!info_.mmu_info_.IsSpecGemm()) {
    return IslEmitter::IsTransferStmt();
  }
  return false;
}

Stmt NPUIslEmitter::EmitAccessNodeCall(const Node *node, const VarMap &var_map_tmp,
                                       BufferedFootPrintInfo &buffer_footprint_info) {
  const Call *call = static_cast<const Call *>(node);
  Array<Expr> args;
  for (auto iv : call->args) {
    args.push_back(ReplaceLoopVar(var_map_tmp).Mutate(iv));
  }
  // Not hoisted, emitting just the mapped subscript.
  if (!buffer_footprint_info.cluster_id) {
    std::string call_name = call->name;
    if (IsTransferStmt() && (std::string::npos == call_name.find(LOCAL_BUF))) {
      call_name = call_name + LOCAL_BUF;
      Tensor t = info_.FindTensor(call_name);
      if (t.defined()) {
        return Evaluate::make(Call::make(call->type, call_name, args, call->call_type, t->op, call->value_index));
      } else {
        LOG(WARNING) << "Call can not found tensor!!!  tensor name: " << call_name;
      }
    }
    return Evaluate::make(Call::make(call->type, call->name, args, call->call_type, call->func, call->value_index));
  }
  return Stmt();
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
