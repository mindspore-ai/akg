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

#ifndef EMIT_INSN_INSN_INFO_H_
#define EMIT_INSN_INSN_INFO_H_

#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <limits.h>

#include <string>
#include <vector>
#include <set>
#include <unordered_set>

#include "tvm.h"
#include "ir_pass.h"
#include "common/array_api.h"

namespace akg {
using ir::GetInt32Const;
using ir::GetIntConst;
using ir::GetUIntConst;
using ir::IsConstExpr;
using ir::IsFlexVarInIf;

using ktvm::ir::substitute;

enum ArgType {
  ARG_VECTOR_ELEWISE = 1,
  ARG_VECTOR_REDUCTION,
  ARG_VECTOR_BROADCAST,
  ARG_VECTOR_REDUCTION_LAST_AXIS,
  ARG_VECTOR_REDUCTION_BISECTION,
  ARG_VECTOR_BROADCAST_LAST_AXIS,
  ARG_NOT_DEFINE
};

enum PatternType {
  PATTERN_3D = 1,
  PATTERN_PARTIAL_3D,
  PATTERN_2D,
  PATTERN_2D_BLOCK,
  PATTERN_1D
};

class StmtStoreInfoNode : public Node {
 public:
  Array<Expr> strides_;
  Array<Expr> shape_;
  Array<Var> var_;
  Array<Var> flex_var_;
  std::string scope_;
  std::string name_;
  Expr index_;
  Expr elem_offset_;
  Expr insn_offset_;
  Type dtype_;
  int data_alignment_{0};
  Var data_;
  Buffer buffer_;

  static constexpr const char *_type_key = "StmtStoreInfo";
  TVM_DECLARE_NODE_TYPE_INFO(StmtStoreInfoNode, Node);

  void VisitAttrs(ktvm::AttrVisitor *v) {
    v->Visit("strides", &strides_);
    v->Visit("shape", &shape_);
    v->Visit("var", &var_);
    v->Visit("flexVar", &flex_var_);
    v->Visit("scope", &scope_);
    v->Visit("name", &name_);
    v->Visit("index", &index_);
    v->Visit("elemOffset", &elem_offset_);
    v->Visit("insnOffset", &insn_offset_);
    v->Visit("dtype", &dtype_);
    v->Visit("dataAlignment", &data_alignment_);
    v->Visit("data", &data_);
  }
};

class StmtStoreInfo : public NodeRef {
 public:
  StmtStoreInfo() = default;
  explicit StmtStoreInfo(const ObjectPtr<Object> &n) : NodeRef(n), node_(n) {}
  ~StmtStoreInfo() = default;

  inline StmtStoreInfoNode *GetNode() const {
    return static_cast<StmtStoreInfoNode *>(node_.get());
  }

  inline const StmtStoreInfoNode *operator->() const {
    return static_cast<const StmtStoreInfoNode *>(node_.get());
  }

  void CleanFlexVar();

  StmtStoreInfo Copy() const;

  void Print() const {
    LOG(DEBUG) << "[ name: " << GetNode()->name_ << ", shape: " << GetNode()->shape_ << ", var: " << GetNode()->var_
               << ", flex_var: " << GetNode()->flex_var_ << ", stride: " << GetNode()->strides_
               << ", scope: " << GetNode()->scope_ << ", index: " << GetNode()->index_
               << ", elem_offset: " << GetNode()->elem_offset_ << ", insn_offset: " << GetNode()->insn_offset_
               << ", buffer: " << GetNode()->buffer_ << ", dtype: " << GetNode()->dtype_
               << ", data_alignment: " << GetNode()->data_alignment_ << ", data: " << GetNode()->data_ << " ]";
  }

 private:
  ObjectPtr<Object> node_;
};

using StmtInfoList = Array<StmtStoreInfo>;

class VectorArgInfoNode : public Node {
 public:
  VectorArgInfoNode() = default;
  ~VectorArgInfoNode() = default;

  struct LastAxisBroadcastInfo {
    int src_index_{-1};
    Expr src_op_;
    std::string intrin_name_;
  };

  int body_num_{0};
  int body_offset_{0};
  Expr dst_head_;
  Expr dst_stride_m0_;
  // when reduction last axis / VA mode, this param used as dst_stride
  Expr dst_stride_m1_;
  Array<Expr> src_head_list_;
  Array<Expr> src_stride_m0_list_;
  // when VA mode, this param used as dst_stride
  Array<Expr> src_stride_m1_list_;
  Expr repeat_;
  Expr scalar_;
  Expr insn_offset_scale_factor_;
  Expr block_offset_{make_zero(Int(32))};
  Array<Expr> vec_mask_;
  LastAxisBroadcastInfo last_axis_info_;
  // only needed in VA mode
  bool is_vaarg_{false};
  Array<Expr> dst_vasrc_extent_list_;
  Array<Expr> src0_vasrc_extent_list_;
  Array<Expr> src1_vasrc_extent_list_;

  static constexpr const char *_type_key = "VectorArgInfo";
  TVM_DECLARE_NODE_TYPE_INFO(VectorArgInfoNode, Node);

  void VisitAttrs(ktvm::AttrVisitor *v) {
    v->Visit("bodyNum", &body_num_);
    v->Visit("bodyOffset", &body_offset_);
    v->Visit("dstHead", &dst_head_);
    v->Visit("dstStrideM0", &dst_stride_m0_);
    v->Visit("dstStrideM1", &dst_stride_m1_);
    v->Visit("srcHeadList", &src_head_list_);
    v->Visit("srcStrideM0List", &src_stride_m0_list_);
    v->Visit("srcStrideM1List", &src_stride_m1_list_);
    v->Visit("repeat", &repeat_);
    v->Visit("scalar", &scalar_);
    v->Visit("insnOffsetScaleFactor", &insn_offset_scale_factor_);
    v->Visit("blockOffset", &block_offset_);
    v->Visit("vecMask", &vec_mask_);
    v->Visit("isVAArg", &is_vaarg_);
    v->Visit("dstVASrcExtentList", &dst_vasrc_extent_list_);
    v->Visit("src0VASrcExtentList", &src0_vasrc_extent_list_);
    v->Visit("src1VASrcExtentList", &src1_vasrc_extent_list_);
  }
};

class VectorArgInfo : public NodeRef {
 public:
  VectorArgInfo() = default;
  explicit VectorArgInfo(const ObjectPtr<Object> &n) : NodeRef(n), node_(n) {}
  ~VectorArgInfo() = default;

  inline VectorArgInfoNode *GetNode() const {
    return static_cast<VectorArgInfoNode *>(node_.get());
  }

  inline const VectorArgInfoNode *operator->() const {
    return static_cast<const VectorArgInfoNode *>(node_.get());
  }

  void Print() const {
    LOG(DEBUG) << "[ body_num: " << GetNode()->body_num_ << ", body_offset: " << GetNode()->body_offset_
               << ", dst_head: " << GetNode()->dst_head_ << ", dst_stride_m0: " << GetNode()->dst_stride_m0_
               << ", dst_stride_m1: " << GetNode()->dst_stride_m1_ << ", src_head_list: " << GetNode()->src_head_list_
               << ", src_stride_m0_list: " << GetNode()->src_stride_m0_list_
               << ", src_stride_m1_list: " << GetNode()->src_stride_m1_list_ << ", repeat: " << GetNode()->repeat_
               << ", scalar: " << GetNode()->scalar_
               << ", insn_offset_scale_factor: " << GetNode()->insn_offset_scale_factor_
               << ", vec_mask: " << GetNode()->vec_mask_ << " ]";
  }

 private:
  ObjectPtr<Object> node_;
};

class ArgInfoNode : public Node {
 public:
  VectorArgInfo body_arg_info_;
  VectorArgInfo tail_arg_info_;
  Array<VectorArgInfo> reduction_tail_args_;
  PatternType pattern_ = PATTERN_1D;
  ArgType arg_type_ = ARG_NOT_DEFINE;

  static constexpr const char *_type_key = "ArgInfo";
  TVM_DECLARE_NODE_TYPE_INFO(ArgInfoNode, Node);

  void VisitAttrs(ktvm::AttrVisitor *v) {
    v->Visit("body", &body_arg_info_);
    v->Visit("tail", &tail_arg_info_);
    v->Visit("reductionTailArgs", &reduction_tail_args_);
  }
};

class ArgInfo : public NodeRef {
 public:
  ArgInfo() = default;
  explicit ArgInfo(const ObjectPtr<Object> &n) : NodeRef(n), node_(n) {}
  ~ArgInfo() = default;

  inline ArgInfoNode *GetNode() const {
    return static_cast<ArgInfoNode *>(node_.get());
  }

  inline const ArgInfoNode *operator->() const {
    return static_cast<const ArgInfoNode *>(node_.get());
  }

  inline std::string GetPattern() const {
    switch (GetNode()->pattern_) {
      case PATTERN_3D:
        return "3d_pattern";
      case PATTERN_PARTIAL_3D:
        return "partial_3d_pattern";
      case PATTERN_2D:
        return "2d_pattern";
      case PATTERN_2D_BLOCK:
        return "2d_block_pattern";
      case PATTERN_1D:
        return "1d_pattern";
      default:
        return "1d_pattern";
    }
  }

 private:
  ObjectPtr<Object> node_;
};

class StmtInfo {
 public:
  StmtInfo() = default;
  StmtInfo(Array<VarExpr> var, Array<Stmt> op) : vars_(std::move(var)), ops_(std::move(op)) {}
  ~StmtInfo() = default;

  void RemoveItem(int index) {
    vars_ = RemoveItemAtIndex(vars_, index);
    ops_ = RemoveItemAtIndex(ops_, index);
  }

  void RemoveItem(size_t index) {
    vars_ = RemoveItemAtIndex(vars_, index);
    ops_ = RemoveItemAtIndex(ops_, index);
  }

  StmtInfo Copy() const;

  Array<VarExpr> vars_;
  Array<Stmt> ops_;
};

struct BisectionInfoWrapper {
  std::vector<StmtInfoList> bisec_info_list_;
  std::vector<StmtInfo> for_info_list_;
  std::vector<ArgInfo> arg_info_list_;
  Array<Expr> original_shape_;
  Map<std::string, Expr> dma_arg_info_map_;
};

struct InsnAxis {
  int min{0};
  int extent{0};
  Var var;
  int dst_stride{0};
  int src_stride{0};
  std::list<int> src_stride_list;
  std::list<int> stride_list;
};

IterVar GetCceAxis();

int CeilTo(int value, int target);

int FloorTo(int value, int target);

Expr EliminateVarInExpr(Expr e, const Array<Var> &vars);

void SortVarShapeAndStride(Array<Var> &vars, Array<Expr> &shapes, Array<Expr> &strides, bool reverse = false);

std::string GetBufScope(const std::string &name);

std::unordered_set<const Variable *> GetVariablesInExpr(const Expr &expr);

Array<VarExpr> GetVarsInExpr(const Expr &expr, bool exclude_upper_case_vars = false);

int GetScopeBlockSize(const StmtStoreInfo &dst_info, const StmtStoreInfo &src_info);

Expr GetInsnOffset(const StmtStoreInfo &com_info, const Array<Var> &elim_var);

int GetUbBlkSize(const Type &type);

StmtInfo GetForInfo(const Stmt &s);

void GetIfForInfo(const Stmt &s, StmtInfo &if_info, StmtInfo &for_info);

Array<Expr> GetBinaryOpExprChildren(const Expr &e);

Array<Stmt> GetStores(const Stmt &s);

void GetStoreAndLoads(const Stmt &s, Array<NodeRef> &stores, Array<NodeRef> &loads);

void CleanNonLinearVar(const StmtInfoList &dst_info_list, const StmtInfoList &src_info_list, const StmtInfo &if_info);

StmtInfoList GetComputationInfo(const Array<NodeRef> &stores, const StmtInfo &for_info);

void GetCompactComputationInfo(const Stmt &stmt, Array<StmtStoreInfo> &dst_info_list,
                               Array<StmtStoreInfo> &src_info_list, StmtInfo &if_info, StmtInfo &for_info,
                               bool same_dtype = true, bool clean_non_linear = true);

void CompactComputationInfoList(Array<StmtStoreInfo> &dst_info_list, Array<StmtStoreInfo> &src_info_list,
                                const StmtInfo &if_info, StmtInfo &for_info);

void CompactComputationInfoList(StmtStoreInfo &dst_info, Array<StmtStoreInfo> &src_info_list, const StmtInfo &if_info,
                                StmtInfo &for_info);

void CleanForInfoVars(StmtInfo &for_info, const Array<Var> &elim_var);

int GetVecMaxLen(const Type &dtype);

Array<Expr> GenMaskVec(const Type &d_type, unsigned int start, unsigned int end = UINT_MAX, unsigned int stride = 1);

bool IsElementwise(const StmtInfoList &dst_info_list, const StmtInfoList &src_info_list);

bool IsBroadcast(const StmtInfoList &dst_info_list, const StmtInfoList &src_info_list);

bool IsLastAxisBroadcast(const StmtInfoList &dst_info_list, const StmtInfoList &src_info_list);

int GetLastAxisReductionIdx(const StmtInfoList &dst_info_list, const StmtInfoList &src_info_list);

bool IsLastAxisReduction(const StmtInfoList &dst_info_list, const StmtInfoList &src_info_list);

int GetBisectionReductionIdx(const StmtInfoList &dst_info_list, const StmtInfoList &src_info_list, int &compare_idx);

bool IsBisectionReduction(const StmtInfoList &dst_info_list, const StmtInfoList &src_info_list);

bool HasVars(const Expr &index, const Var &vec_var);

int GetVectorizedVarPosition(const Expr &index, Array<Var> &loop_vars);
}  // namespace akg

namespace ktvm {
bool Equal(const akg::StmtStoreInfo &lhs, const akg::StmtStoreInfo &rhs);
}
#endif  // EMIT_INSN_INSN_INFO_H_
