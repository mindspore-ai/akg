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

#include <tvm/expr.h>
#include <tvm/ir.h>
#include <tvm/ir_pass.h>

#include "ir_pass.h"
#include "isl.h"
#include "scop.h"
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
/*!
 * IslEmitter for CCE
 */
class CCEIslEmitter : public IslEmitter {
 public:
  CCEIslEmitter(Scop &s, const NodeInfoRepo &n, const isl::id_list &i) : IslEmitter(s, n, i) { ProcBypassL1(s); }
  ~CCEIslEmitter() override = default;

  Stmt Emit(const isl::ast_node &node) final;

 private:
  // override emitters for CCE
  Stmt EmitFor(const isl::ast_node_for &node) final;
  Stmt EmitIf(const isl::ast_node_if &node) final;
  Stmt EmitMark(const isl::ast_node_mark &node_id) override;
  Stmt EmitBlock(const isl::ast_node_block &node) final;
  Stmt EmitStmt(const isl::ast_node_user &node) final;
  Stmt EmitUserStmt(const isl::ast_node_user &node) override;

  // DMA emitters for CCE
  Expr EmitLoad(const isl::ast_expr &lhs, Type type);
  Stmt EmitL1Read(const isl::ast_node_user &node);
  Stmt EmitL1Write(const isl::ast_node_user &node, Scop::AtomicType atomic);
  Stmt EmitSpecGemL1write(const isl::ast_node_mark &node, const Stmt &stmt);

  // RangeInfo emitters for CCE
  Stmt EmitGemmRangeInfoBackPropFilter(const Stmt &stmt);
  Stmt EmitGemmRangeInfo(Stmt stmt);

  // multicore emitters for CCE
  Stmt EmitMarkMulticore(const isl::ast_node_mark &node);
  bool InjectMulticore(const std::string &iter);

  Stmt EmitMarkFuseVector(const isl::ast_node_mark &node);
  Stmt EmitMarkAllocRealizeOut(const isl::ast_node_mark &node);
  Stmt EmitMarkAllocC(const isl::ast_node_mark &node);
  Stmt EmitMarkSpecGemm(const isl::ast_node_mark &node);

  void EmitAttrStmt(const isl::ast_node_block &block_node, const Liveness &liveness, bool is_L1, bool is_L0,
                    std::vector<Stmt> &stmts);

  void EmitAttrStmtL0(Tensor &t, bool &is_im2col, bool &is_filter_l0, bool &is_gemm_data_trans,
                      bool &is_gemm_weight_trans);

  void EmitAttrStmtL1(Tensor &t, bool &is_fractal, bool &is_filter_l1);

  void EmitAttrStmtLiveness(const Liveness &liveness, std::vector<Stmt> &stmts, int i, bool is_L1);

  void EmitAttrStmtAfterRealize(bool is_L1, bool is_L0, std::vector<Stmt> &stmts);
  void EmitRealize(const isl::ast_node_block &block_node, const Liveness &liveness_info, bool is_L1, bool is_L0,
                   std::vector<Stmt> &stmts);

  void EmitRealizeLivenessInfo(std::vector<IslIdSet> &real, const Liveness &liveness_info,
                               std::unordered_map<isl::id, std::set<int>, isl::IslIdIslHash> &liveness,
                               std::function<bool(const std::string &id)> const &CheckGoOut);
  void EmitGemmRangeInfoNewAxis(std::vector<Range> &range, std::vector<std::string> &prefix,
                                std::unordered_map<std::string, bool> &outerAxis, Range &axisMRange,
                                Map<std::string, Range> &range_map, Map<std::string, VarExpr> &axis_map);

  void EmitGemmRangeInfoDynamic(Range &axisMRange, Map<std::string, Range> &range_map);
  void EmitGemmRangeInfoStatic(Map<std::string, Range> &range_map);
  // realize info for CCE
  Expr FindRealizeScope(const isl::id &var);
  std::string FindRealizeScopeToString(const isl::id &var);
  Stmt InsertRealize(Stmt stmt, const isl::id &var, bool is_L0);
  void RealizeOut();

  Stmt RemoveCond(const Stmt &stmt);
  void ProcBypassL1(const Scop &scop);
  void SetCube(const isl::id &stmt_id);
  std::string ReplaceAxis(const std::string &oldAxis);
  static std::vector<std::string> ConstructPrefix();
  void GemmTranspose(std::vector<Stmt> &stmts);
  void ConvBackPropFilterFixMadInit(const isl::ast_node_mark &node, Expr &mad_init_cond);

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
