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
#ifndef POLY_GPU_ISL_EMITTER_H_
#define POLY_GPU_ISL_EMITTER_H_

#include "poly/isl_emitter.h"
#include "pass/utils.h"

namespace akg {
namespace ir {
namespace poly {

// add for mind tricks swizzle
constexpr auto MIND_TRICKS_PRESERVE_DIMENSION_MARKER = "mind_trick_preserve_dimension_marker";
constexpr auto MIND_TRICKS_SWIZZLE_PRAGMA = "pragma_swizzle_kernel";

// add for one dimension mapping
constexpr auto ORIGIN_THREAD_DIM_X = "bind_thread_x";

// example:
// atomic_SumOp
constexpr auto REDUCE_ATOMIC_FLAG_SIZE = 2;
constexpr auto REDUCE_ATOMIC_FLAG = "atomic";
constexpr auto REDUCE_ATOMIC_FLAG_POS = 0;
constexpr auto REDUCE_ATOMIC_FLAG_TYPE_POS = 1;

constexpr auto REDUCE_LIB_TYPE_FLAG = "reduceLibType";
constexpr auto REDUCE_INIT_FLAG = "InitStmt";

class GpuIslEmitter : public IslEmitter {
 public:
  GpuIslEmitter(ScopInfo &info, const NodeInfoRepo &n, const isl::id_list &i) : IslEmitter(info, n, i) {}
  ~GpuIslEmitter() override = default;

  bool NoNeedToEmitForTempTensor(const isl::id &id);
  Expr Interpret(const isl::ast_expr &e);
  Stmt EmitStmt(const isl::ast_node_user &node) override;
  Stmt EmitMark(const isl::ast_node_mark &node_id) override;
  void EmitterPreProcess() override;
  Stmt EmitterPostProcess(Stmt &s) override;
  virtual Expr AdaptPolyNewVar(std::string name);
  Expr AdaptThreadNewVar(const std::string &name, MappingCfg *mapping_cfg);
  Expr AdaptOneConfigForMulAxis(MappingCfg *mapping_cfg, const std::string &orig_name, const bool is_thread);
  int GetThreadExtent(const std::string &name);
  virtual Expr IterNameAdaptor(std::string name);
  Stmt EmitAccessNodeFromPromoteAcsCall(isl::id var, const Node *node, Array<Expr> &args) override;
  Stmt EmitIf(const isl::ast_node_if &node) override;
  Stmt EmitFor(const isl::ast_node_for &node) override;
  virtual Stmt SubstituteTensorStmt(const Stmt &s, Tensor origin, Tensor replaced);
  virtual Stmt EmitTensorOfTensorStmt(const Stmt &s);
  void UpdateGpuIndexDtype();
  std::map<std::string, VarExpr> iter_name_map_{
    {B0, VarExpr(BLOCK_IDX_X, Int(32))},  {B1, VarExpr(BLOCK_IDX_Y, Int(32))},  {B2, VarExpr(BLOCK_IDX_Z, Int(32))},
    {T0, VarExpr(THREAD_IDX_X, Int(32))}, {T1, VarExpr(THREAD_IDX_Y, Int(32))}, {T2, VarExpr(THREAD_IDX_Z, Int(32))}};

 private:
  // override emitters for GPU
  Stmt EmitBlock(const isl::ast_node_block &node) final;
  Expr EmitLoad(const isl::ast_expr &lhs, Type type) final;

  Stmt EmitAccessNodeFromPromoteAcsProvide(isl::id var, const Node *node, Array<Expr> &args);

  Stmt EmitSync();
  Stmt EmitAttr();  // thread_extent, virtual_thread

  Expr FindRealizeScope(const isl::id &var);
  std::string FindRealizeScopeToString(const isl::id &var);
  Stmt InsertRealize(Stmt stmt, const isl::id &var);

  Expr SingleConfigToMultiBand(std::string name);

  Expr ModifyTheInitExpr(const Expr &e);
  Expr ModifyTheCondExpr(const Expr &e, int inc);
  Expr ModifyTheIterExpr(const VarExpr &iter, int inc, const Expr &init);

  Stmt EmitRealizeForGlobalTensor(Stmt stmt);

  std::unordered_map<const Variable *, Expr> stride_modify_iter_map_;
};

class AtomicReturnStmtEmit : public IRMutator {
 public:
  explicit AtomicReturnStmtEmit(ScopInfo &scop_info) : scop_info_(scop_info) {}

  Stmt Mutate_(const AttrStmt *op, const Stmt &s);

  Stmt Mutate_(const Provide *op, const Stmt &s);

 private:
  ScopInfo &scop_info_;
  AtomicReturnData atomic_data_;
  bool in_atomic_area_{false};
};

}  // namespace poly
}  // namespace ir
}  // namespace akg
#endif  // POLY_GPU_ISL_EMITTER_H_
