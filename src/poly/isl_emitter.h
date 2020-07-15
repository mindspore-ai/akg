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

#ifndef POLY_ISL_EMITTER_H_
#define POLY_ISL_EMITTER_H_

#include <tvm/expr.h>
#include <tvm/ir.h>
#include <tvm/ir_pass.h>

#include "ir_pass.h"
#include "poly/isl.h"
#include "poly/scop.h"

namespace akg {
namespace ir {
namespace poly {
constexpr int EIGHT_BYTES = 8;
constexpr int ONE_ARGS = 1;
constexpr int TWO_ARGS = 2;

bool AOutThanB(std::vector<const Node *> a, std::vector<const Node *> b);
using IterMap = std::unordered_map<isl::id, const Variable *, isl::IslIdIslHash>;
using VarMap = std::unordered_map<isl::id, Expr, isl::IslIdIslHash>;

/*!
 * \brief transfer ISL to Halide
 */
class IslEmitter {
 private:
  Expr InterpretOp(const isl::ast_expr_op &e);
  Expr InterpretMultiargsOp(const isl::ast_expr_op &e);
  Expr InterpretUnaryOp(const isl::ast_expr_op &e);
  Expr InterpretBinaryOp(const isl::ast_expr_op &e);

 public:
  explicit IslEmitter(Scop &s_, const NodeInfoRepo &n_, const isl::id_list &i_)
      : scop_(s_), node_info_map_(n_), iter_names_(i_) {}
  virtual ~IslEmitter() = default;

  /// Interpret isl::ast_expr to Halide Expr
  //@{
  Expr Interpret(const isl::ast_expr &e);
  //@}

  // helper functions, which may can be moved into a separated class
  isl::space GetDomainSpace(const isl::id &stmt_id);
  isl::space GetSpace(const isl::id &tensor_id, const Array<Expr> &tensor_index, const isl::id &stmt_id);
  isl::multi_aff TensorAccessMultAff(const isl::id &tensor_id, const Array<Expr> &subscripts, const isl::id &stmt_id);
  isl::set Domain() const {
    auto iterator_map = node_info_map_.at(node_id_).iterator_map;
    return isl::map::from(iterator_map).range();
  }
  Stmt EmitAccessNode(const std::string &name, const Node *node, const Array<Expr> &tensor_index,
                      const VarMap &var_map_tmp);
  Stmt EmitAccessNodeFromPromoteAcsProvide(Scop &scop, isl::id var, const Node *node, Array<Expr> &args);
  Stmt EmitAccessNodeFromPromoteAcsCall(Scop &scop, isl::id var, const Node *node, Array<Expr> &args);
  /// Virtual emitters for different type node
  //@{
  virtual Stmt Emit(const isl::ast_node &node);
  virtual Stmt EmitFor(const isl::ast_node_for &node);
  virtual Stmt EmitIf(const isl::ast_node_if &node);
  virtual Stmt EmitMark(const isl::ast_node_mark &node);
  virtual Stmt EmitBlock(const isl::ast_node_block &node);
  virtual Stmt EmitUserStmt(const isl::ast_node_user &node);
  virtual Stmt EmitStmt(const isl::ast_node_user &node);
  virtual Stmt EmitAst(const isl::ast_node &node);
  virtual Stmt EmitUserStmtContent(const Node *node);
  virtual Stmt EmitUserStmtContent(const Provide *provide_node);
  virtual Stmt EmitUserStmtContent(const Evaluate *eva_node);
  virtual Stmt EmitUserStmtContent(const IfThenElse *if_node);
  virtual Stmt EmitUserStmtContent(const For *for_node);
  virtual Stmt EmitUserStmtContent(const Block *block_node);
  //@}

  std::unordered_set<isl::id, isl::IslIdIslHash> realize_use_;
  std::unordered_set<isl::id, isl::IslIdIslHash> realize_use_with_may_def_;
  std::unordered_set<isl::id, isl::IslIdIslHash> realize_must_def_;
  std::unordered_set<isl::id, isl::IslIdIslHash> realize_may_def_;
  std::unordered_set<isl::id, isl::IslIdIslHash> realize_out_;
  std::unordered_set<isl::id, isl::IslIdIslHash> global_realize_out_;

  /// Scop
  Scop &scop_;

  /// Node information map including
  const NodeInfoRepo &node_info_map_;

  /// Loop isl iters info
  //@{
  /// Loop isl iters list
  isl::id_list iter_names_;
  /// Loop declared halide iters
  std::vector<const Variable *> iters_;

  virtual void PushIter(const Variable *iter);
  virtual void PopIter(const Variable *iter);
  bool FindIter(const Variable *iter) const;
  const Variable *GetIterByName(const std::string &id) const;
  //@}

  std::map<const Variable *, std::string> iters_old_name_;
  std::map<const Variable *, std::string> iters_new_name_;

  // current ast node id
  isl::id node_id_;
  // current stmt id
  isl::id stmt_id_;

  VarMap var_map_;

  // emit in if
  std::vector<const Node *> cur_if_list_;
  std::unordered_map<isl::id, std::vector<const Node *>, isl::IslIdIslHash> if_map_;
  std::unordered_map<isl::id, VarMap, isl::IslIdIslHash> stmt_var_map_;
};

class ExtractIterfromExpr : public air::ir::IRVisitor {
 public:
  ExtractIterfromExpr() = default;
  ~ExtractIterfromExpr() override = default;
  void Visit_(const Variable *op) final {
    vec_.push_back(op);
    IRVisitor::Visit_(op);
  }

  std::vector<const Variable *> Run(const Expr &expr) {
    IRVisitor::Visit(expr);
    return vec_;
  }

 private:
  std::vector<const Variable *> vec_;
};

void FindBufferFootprintById(Scop::BufferedFootPrintInfo &buffer_footprint_info,
                             std::vector<Scop::BufferedFootPrintInfo> active_buffer_fp, isl::id id);
void GetNameWithoutLocal(isl::id &tensor_id, Scop &scop);
bool IsTransferStmt(Scop &scop, isl::id &stmt_id);
bool IsCopyinFromAnotherBand(Scop &scop, isl::multi_aff &access);
void AffSubForAstToSchedule(isl::pw_multi_aff &ast_to_schedule, bool &is_transfer_stmt,
                            bool &is_copyin_from_another_band);
Stmt EmitAccessNodeProvide(const Node *node, const VarMap &var_map_tmp, Scop::BufferedFootPrintInfo &buffer_fp_info);
Stmt EmitAccessNodeCall(const Node *node, const VarMap &var_map_tmp, Scop::BufferedFootPrintInfo &buffer_fp_info,
                        bool &is_transfer_stmt, Scop &scop);
}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif  // POLY_ISL_EMITTER_H_
