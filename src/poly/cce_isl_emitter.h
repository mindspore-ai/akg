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
#ifndef POLY_CCE_ISL_EMITTER_H_
#define POLY_CCE_ISL_EMITTER_H_

#include "ir_pass.h"
#include "isl_emitter.h"

namespace akg {
namespace ir {
namespace poly {
using IslIdSet = std::unordered_set<isl::id, isl::IslIdIslHash>;
class Liveness {
 public:
  std::vector<IslIdSet> must_def_;
  std::vector<IslIdSet> may_def_;
  std::vector<IslIdSet> use_;
  std::vector<IslIdSet> use_with_may_def_;
  std::vector<IslIdSet> out_;
  std::vector<IslIdSet> read_;
  std::vector<IslIdSet> write_;
};
enum AtomicType { Equ = 0, Add };
/*!
 * IslEmitter for CCE
 */
class CCEIslEmitter : public IslEmitter {
 public:
  CCEIslEmitter(ScopInfo &info, const NodeInfoRepo &n, const isl::id_list &i) : IslEmitter(info, n, i) {
    ProcBypassL1(info);
  }
  ~CCEIslEmitter() override = default;

  Stmt Emit(const isl::ast_node &node) final;

 private:
  // override emitters for CCE
  Stmt EmitFor(const isl::ast_node_for &node) final;
  Stmt EmitMark(const isl::ast_node_mark &node_id) override;
  Stmt EmitBlock(const isl::ast_node_block &node) final;
  Stmt EmitStmt(const isl::ast_node_user &node) final;
  Stmt EmitUserStmt(const isl::ast_node_user &node) override;

  // DMA emitters for CCE
  Expr EmitLoad(const isl::ast_expr &lhs, Type type);
  Stmt EmitRead(const isl::ast_node_user &node);
  Stmt EmitWrite(const isl::ast_node_user &node, AtomicType atomic);
  Stmt EmitSpecGemL1write(const isl::ast_node_mark &node, const Stmt &stmt);

  // emit mark node
  Stmt EmitMarkMulticore(const isl::ast_node_mark &node);
  Stmt EmitMarkFuseVector(const isl::ast_node_mark &node);
  Stmt EmitMarkAllocRealizeOut(const isl::ast_node_mark &node);
  Stmt EmitMarkAllocC(const isl::ast_node_mark &node);
  Stmt EmitMarkSpecGemm(const isl::ast_node_mark &node);

  // emit attrs
  void EmitAttrStmt(const isl::ast_node_block &block_node, const Liveness &liveness, bool is_L1, bool is_L0,
                    std::vector<Stmt> &stmts);
  void EmitReadAttrAtL0(std::vector<Stmt> &stmts, int i, Tensor &t);
  void EmitReadAttrAtL1(std::vector<Stmt> &stmts, int i, Tensor &t);
  void EmitReadAttr(const std::vector<IslIdSet> &read, std::vector<Stmt> &stmts, int i, bool is_L1, bool is_L0);
  void EmitWriteAttr(const std::vector<IslIdSet> &write, std::vector<Stmt> &stmts, int i, bool is_L1);
  void EmitAttrStmtAfterRealize(bool is_L1, bool is_L0, std::vector<Stmt> &stmts);
  Stmt EmitGemmRangeInfoBackPropFilter(const Stmt &stmt);
  Stmt EmitGemmRangeInfo(Stmt stmt);

  // emit realize
  void EmitRealize(const isl::ast_node_block &block_node, const Liveness &liveness_info, bool is_L1, bool is_L0,
                   std::vector<Stmt> &stmts);

  // emit access
  Stmt EmitAccessNodeCall(const Node *node, const VarMap &var_map_tmp, BufferedFootPrintInfo &buffer_fp_info) override;

  // tool func
  bool InjectMulticore(const std::string &iter);
  void CollectLiveness(const Liveness &liveness_info, bool is_L1, std::vector<IslIdSet> &real,
                       std::unordered_map<isl::id, std::set<int>, isl::IslIdIslHash> &liveness,
                       std::function<bool(const std::string &id)> const &CheckGoOut);
  void CollectGemmRangeInfoNewAxis(std::vector<Range> &range, std::vector<std::string> &prefix,
                                   std::unordered_map<std::string, bool> &outerAxis, Range &axisMRange,
                                   Map<std::string, Range> &range_map, Map<std::string, VarExpr> &axis_map);

  void CollectGemmMWSize(Range &axis_m_range, Map<std::string, Range> &range_map);
  void CollectGemmMWSizeDynamic(Map<std::string, Range> &range_map);
  Expr FindRealizeScope(const isl::id &var);
  std::string FindRealizeScopeToString(const isl::id &var);
  Stmt InsertRealize(Stmt stmt, const isl::id &var, bool is_L0);
  void RealizeOut();

  Stmt RemoveCond(const Stmt &stmt);
  void ProcBypassL1(const ScopInfo &info);
  void SetCube(const isl::id &stmt_id);
  std::string ReplaceAxis(const std::string &oldAxis);
  static std::vector<std::string> ConstructPrefix();
  void GemmTranspose(std::vector<Stmt> &stmts);
  void ConvBackPropFilterFixMadInit(const isl::ast_node_mark &node, Expr &mad_init_cond);

  isl::multi_aff TensorAccessMultAff(isl::id &tensor_id, const Array<Expr> &subscripts,
                                     const isl::id &stmt_id) override;
  bool IsTransferStmt() override;
  bool IsCopyinFromAnotherBand(isl::multi_aff &access) override;

  std::map<const Variable *, std::string> iters_old_name_;
  std::map<const Variable *, std::string> iters_new_name_;
  std::unordered_map<isl::id, VarMap, isl::IslIdIslHash> stmt_var_map_;

  std::set<Tensor> realized_;
  IslIdSet hoisted_read_;
  IslIdSet hoisted_write_;

  int bypassL1_{0};
  bool is_cube_{false};
  StmtOpInfo opinfo_;

  // range info index
  int range_idx_{0};
  int tile_idx_{0};
  int isolate_idx_{-1};

  bool is_old_gemm_l1write_{false};
  std::vector<Stmt> cube_l0write_;
  int gemm_transpose_index_{0};
  std::set<const Variable *> rmif_;

  void *args_{nullptr};

  struct {
    /* whether current band has multicore for loops */
    bool enabled{false};
    /* nesting depth of multicore For loops, starting from 0 */
    int multicore_depth{0};
    /* coincidence array of current band, indicating whether the loop of depth N can use multicore */
    std::vector<int> coincidence;
  } multicore_info;
};
}  // namespace poly
}  // namespace ir
}  // namespace akg
#endif  // POLY_CCE_ISL_EMITTER_H_
