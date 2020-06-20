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

#include "emit_insn/insn_emitter.h"

#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm/packed_func_ext.h>
#include <tvm/runtime/registry.h>

#include <unordered_set>
#include <map>
#include <numeric>
#include <set>
#include <algorithm>

#include "pass/ir_util.h"
#include "ir_pass.h"
#include "common/array_api.h"
#include "cce_params.h"
#include "insn_builder.h"
#include "insn_info.h"
#include "insn_pattern.h"
#include "insn_emitter_multimask.h"

namespace akg {
namespace ir {
/// Sort indexes
/// \param v
/// \return
std::vector<size_t> SortIndexes(const std::vector<int> &v) {
  std::vector<size_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) { return v[i1] > v[i2]; });
  return idx;
}

/// Function for emit single vector intrin
/// \param op                - The input stmt to be emitted as intrin
/// \param intrin_name        - The CCE intrin name
/// \param broadcast_last_axis - Tag of broadcast_last_axis mode
/// \return Stmt of emitted CCE intrin
Stmt SingleVecEmitter(const Stmt &op, std::string intrin_name, bool broadcast_last_axis = false) {
  CHECK(op);
  Stmt result;
  // optimization of copy_ubuf_to_ubuf
  bool is_dma_opt = false;
  if (intrin_name == INTRIN_NAME_COPY_UB_TO_UB) {
    CommentManager::GetInstance().AddComment("Insn_type", "dma_copy");
    CommentManager::GetInstance().AddComment("Insn_name", INTRIN_NAME_COPY_UB_TO_UB);
    CommentManager::GetInstance().AddComment("Vadds_replace_copy", "enable");
    intrin_name = "vadds";
    is_dma_opt = true;
  } else {
    CommentManager::GetInstance().AddComment("Insn_type", "single_vector");
    CommentManager::GetInstance().AddComment("Insn_name", intrin_name);
  }

  StmtInfoList dst_info_list;
  StmtInfoList src_info_list;
  StmtStoreInfo scalar_info;
  StmtInfo for_info;
  StmtInfo if_info;
  std::string mode = GetSingleVecComputationInfo(op, intrin_name, dst_info_list, src_info_list, if_info, for_info);
  CHECK(!dst_info_list.empty());

  if (broadcast_last_axis) {
    mode = "broadcast_last_axis";
    // In this case, must come from binary vec, so must have two src
    CHECK(src_info_list.size() >= 2) << "Broadcast last axis mode must have at least two srcs.";
    if (!IsTwoItemEqual(src_info_list[0]->var_, dst_info_list[0]->var_, -1)) {
      scalar_info = src_info_list[0];
      src_info_list.Set(0, src_info_list[1]);
    } else if (!IsTwoItemEqual(src_info_list[1]->var_, dst_info_list[0]->var_, -1)) {
      scalar_info = src_info_list[1];
    }
  } else {
    if (mode == "broadcast" && !src_info_list.empty() && dst_info_list.size() == 1) {
      if (!IsTwoItemEqual(src_info_list[0]->var_, dst_info_list[0]->var_, -1)) {
        mode = "broadcast_last_axis";
      }
      if (src_info_list.size() > 1) {
        if (!IsTwoItemEqual(src_info_list[1]->var_, dst_info_list[0]->var_, -1)) {
          mode = "broadcast_last_axis";
        } else {
          scalar_info = src_info_list[0];
          src_info_list.Set(0, src_info_list[1]);
        }
      }
    }
  }

  if (broadcast_last_axis) {
    mode = "broadcast_last_axis";
  }

  if (intrin_name == INTRIN_NAME_VECTOR_DUP) {
    auto dst_info = dst_info_list[0];
    if (dst_info->var_.size() > 1 &&
        GetIntConst(GetItem(dst_info->strides_, -1)) == GetIntConst(GetItem(dst_info->shape_, -1)) + 1) {
      // diagnoal broadcast case
      return op;
    }
    dst_info.CleanFlexVar();
  }

  // check is single vector broadcast reduce mode exist
  SingleVecPatternGenerator generator = SingleVecPatternGenerator(dst_info_list, src_info_list, for_info, mode);
  auto params = generator.GetInsnArgs();
  dst_info_list = params.dst_info_list;
  src_info_list = params.src_info_list;
  for_info = params.for_info;
  ArgInfo arg_info = params.arg_info;

  CommentManager::GetInstance().AddComment("Compute_type", mode);
  CommentManager::GetInstance().AddComment("Pattern", arg_info.GetPattern());

  if (intrin_name == "vadds" || intrin_name == "vmuls" || intrin_name == INTRIN_NAME_VECTOR_DUP) {
    auto stores = GetStores(op);
    auto store = stores[0].as<Store>();
    auto scalar = Expr(0);
    if (intrin_name == "vadds" || intrin_name == "vmuls") {
      if (!dst_info_list.empty()) {
        scalar = FloatImm::make(dst_info_list[0]->dtype_, 0.000000);
      }
      if (!dst_info_list[0]->dtype_.is_float()) {
        return op;
      }
      if (!is_dma_opt) {
        if (!scalar_info.defined()) {
          auto children = GetBinaryOpExprChildren(store->value);
          if (children.empty()) {
            LOG(FATAL) << store->value << " is not binary op.";
          }
          scalar = children[1];
        } else {
          scalar = Load::make(scalar_info->dtype_, scalar_info->data_, scalar_info->index_, Expr(1));
        }
      }
    } else if (intrin_name == INTRIN_NAME_VECTOR_DUP) {
      if (store->value->IsInstance<Load>()) {
        // scale is load
        scalar =
          Load::make(src_info_list[0]->dtype_, store->value.as<Load>()->buffer_var, src_info_list[0]->index_, Expr(1));
      } else {
        // scale is imm
        scalar = store->value;
      }
    }

    if (arg_info->body_arg_info_.defined()) {
      arg_info->body_arg_info_.GetNode()->scalar_ = scalar;
    }
    if (arg_info->tail_arg_info_.defined()) {
      arg_info->tail_arg_info_.GetNode()->scalar_ = scalar;
    }
  }

  if (intrin_name == "vconv_deq") {
    result = InsertBody(
      result, Evaluate::make(Call::make(Float(16), "set_deqscale", {FloatImm::make(Float(16), 1.0)}, Call::Extern)));
  }

  SingleVecInsnBuilder single_vec_builder =
    SingleVecInsnBuilder(dst_info_list[0], src_info_list[0], arg_info, intrin_name);
  auto insn_list = single_vec_builder.EmitIntrin();

  if (intrin_name == INTRIN_NAME_VECTOR_DUP && dst_info_list[0]->var_.empty()) {
    Stmt store;
    auto ScanStore = [&store](const NodeRef &op) {
      const auto e = op.as<Store>();
      if (e != nullptr) {
        store = Store::make(e->buffer_var, e->value, e->index, e->predicate);
      }
    };
    ktvm::ir::PostOrderVisit(op, ScanStore);
    store = EmitSetVecMaskIntrin(store, dst_info_list[0]->dtype_);
    insn_list = {store};
  }

  return FoldInsnWithForInfo(insn_list, if_info, for_info, result);
}

/// Function to emit binary vector intrin
/// \param op           - The input stmt to be emitted as intrin
/// \param intrin_name   - The CCE insn name
/// \param enable_bisect - Tag of enable bisect-reduction mode
/// \param postfix      - postfix
/// \return Stmt of emitted CCE intrin
Stmt BinaryVecEmitter(const Stmt &op, std::string intrin_name, bool enable_bisect = true, int postfix = 0) {
  CHECK(op);
  StmtInfoList dst_info_list;
  StmtInfoList src_info_list;
  StmtInfo for_info;
  StmtInfo if_info;
  auto arg_info = GetBinaryVecInsnArgs(op, intrin_name, dst_info_list, src_info_list, if_info, for_info, enable_bisect);
  CommentManager::GetInstance().AddComment("Insn_type", "binary_vector");
  CommentManager::GetInstance().AddComment("Insn_name", intrin_name);

  switch (arg_info->arg_type_) {
    case ARG_VECTOR_BROADCAST_LAST_AXIS: {
      CommentManager::GetInstance().CleanComments();
      intrin_name += "s";
      return SingleVecEmitter(op, intrin_name, true);
    }
    case ARG_VECTOR_REDUCTION_LAST_AXIS: {
      CommentManager::GetInstance().AddComment("Compute_type", "reduce_last_axis");
      auto dst_info = dst_info_list[0];
      auto src_info = src_info_list[1];
      if (src_info_list[0]->var_.size() > src_info_list[1]->var_.size()) {
        src_info = src_info_list[0];
      }
      const int vec_max_len = GetVecMaxLen(dst_info->dtype_);
      if (enable_bisect && GetIntConst(GetItem(src_info->shape_, -1)) > vec_max_len) {
        CommentManager::GetInstance().AddComment("Bisect_optimize", "enabled");
        auto wrapper =
          SeparateComInfoToBisectionInfoList(dst_info_list, src_info_list, for_info, if_info, true, postfix);
        return EmitCceBinaryVectorToBisectionReduction(wrapper, if_info, intrin_name);
      } else {
        CommentManager::GetInstance().AddComment("Pattern", arg_info.GetPattern());
        ReduceLastAxisPatternGenerator generator =
          ReduceLastAxisPatternGenerator(dst_info, src_info, for_info, intrin_name);
        auto result = generator.GetInsnArgs();
        arg_info = result.arg_info;
        dst_info = result.dst_info_list[0];
        src_info = result.src_info_list[0];
        for_info = result.for_info;
        return EmitCceBinaryVectorToReduceLastAxis(dst_info, src_info, if_info, for_info, arg_info, intrin_name);
      }
    }
    case ARG_VECTOR_REDUCTION_BISECTION: {
      CommentManager::GetInstance().AddComment("Compute_type", "reduction");
      CommentManager::GetInstance().AddComment("Bisect_optimize", "enabled");
      auto wrapper =
        SeparateComInfoToBisectionInfoList(dst_info_list, src_info_list, for_info, if_info, false, postfix);
      return EmitCceBinaryVectorToBisectionReduction(wrapper, if_info, intrin_name);
    }
    default: {
      CommentManager::GetInstance().AddComment("Pattern", arg_info.GetPattern());
      std::string mode;
      switch (arg_info->arg_type_) {
        case ARG_VECTOR_ELEWISE:
          mode = "elewise";
          break;
        case ARG_VECTOR_REDUCTION:
          mode = "reduction";
          break;
        case ARG_VECTOR_BROADCAST:
          mode = "broadcast";
          break;
        default:
          mode = "unknown";
          break;
      }
      CommentManager::GetInstance().AddComment("Compute_type", mode);

      auto dst_info = dst_info_list[0];
      MultiVecInsnBuilder builder = MultiVecInsnBuilder(dst_info, src_info_list, arg_info, intrin_name);
      auto insn_list = builder.EmitIntrin();
      Stmt stmt;
      return FoldInsnWithForInfo(insn_list, if_info, for_info, stmt);
    }
  }
}

/// Function to emit scalar intrin
/// \param op         - The input stmt to be emitted as intrin
/// \param intrin_name - The CCE insn name
/// \return Stmt of CCE scalar intrin
Stmt ReturnOpEmitter(const Stmt &op) {
  CHECK(op);
  // do not change index for scatter op
  CommentManager::GetInstance().AddComment("Insn_name", "scalar");
  return op;
}

/// Generate CCE vconv cmd name
/// \param src_type - Src data type
/// \param dst_type - Dst data type
/// \param bak_fix  - Tag of trunc mode
/// \return CCE vconv intrin name
std::string GetConvCmd(const Type &src_type, const Type &dst_type, const std::string &bak_fix) {
  auto GetTypeMark = [](const Type &type) -> std::string {
    if (type == Int(8)) {
      return "s8";
    } else if (type == UInt(8)) {
      return "u8";
    } else if (type == Int(16)) {
      return "s16";
    } else if (type == Int(32)) {
      return "s32";
    } else if (type == Float(16)) {
      return "f16";
    } else if (type == Float(32)) {
      return "f32";
    }
    LOG(FATAL) << "Error: cannot cast the unsupported type";
    return "-1";
  };
  std::string cast_type = GetTypeMark(src_type) + "2" + GetTypeMark(dst_type);
  return "vconv_" + cast_type + bak_fix;
}

/// Function to emit vconv intrin
/// \param op     - The input stmt to be emitted as intrin
/// \param bak_fix - Tag of trunc mode
/// \return Stmt of emitted CCE intrin
Stmt SingleVconvEmitter(const Stmt &op, const std::string &bak_fix) {
  CHECK(op.defined());
  Array<NodeRef> dst_stmt, src_stmt;
  GetStoreAndLoads(op, dst_stmt, src_stmt);
  CHECK(!src_stmt.empty());
  CHECK(!dst_stmt.empty());
  CHECK(src_stmt[0].as<Load>());
  CHECK(dst_stmt[0].as<Store>());
  auto intrin_name = GetConvCmd(src_stmt[0].as<Load>()->type, dst_stmt[0].as<Store>()->value.type(), bak_fix);
  return SingleVecEmitter(op, intrin_name);
}

/// Function to emit argmax cast intrin
/// \param op - The input stmt to be emitted as intrin
/// \return Stmt of emitted CCE intrin
Stmt SingleFargmaxCastEmitter(const Stmt &op) {
  CHECK(op);
  StmtInfoList dst_info_list, src_info_list;
  StmtInfo if_info, for_info;
  GetCompactComputationInfo(op, dst_info_list, src_info_list, if_info, for_info, false);
  Array<Buffer> srcs;
  std::transform(src_info_list.begin(), src_info_list.end(), std::back_inserter(srcs.CopyOnWrite()->data), GenBufferId);

  auto base_stmt = EmitFargmaxCast(srcs, dst_info_list[0]);

  CommentManager::GetInstance().AddComment("Insn_name", "vconv_s162s32");
  CommentManager::GetInstance().AddComment("Insn_type", "single_vector");

  return GenIfAndFor(base_stmt, if_info, for_info);
}

/// Function to emit COR intrin
/// \param op - The input stmt to be emitted as intrin
/// \return Stmt of emitted CCE intrin
Stmt BinaryNmsEmitter(const Stmt &op) {
  CHECK(op.defined());
  Array<Buffer> dst_list, src_list;
  GetBufferIdFromStmt(op, dst_list, src_list);
  CHECK(!dst_list.empty());
  CHECK_GE(src_list.size(), 2);
  Buffer dst = dst_list[0];
  Buffer src = src_list[1];

  auto stores = GetStores(op);
  CHECK(!stores.empty());
  Stmt store = GetStores(op)[0];
  CHECK(store.as<Store>());
  CHECK(store.as<Store>()->value.as<Call>());
  Expr thre = store.as<Store>()->value.as<Call>()->args[2];
  const int BOX_PER_INSN = 16;
  CHECK(!src->shape.empty());
  int boxnum = GetInt32Const(src->shape[0]);
  size_t buffer_num = 3;
  Array<Var> var_list;
  Array<Buffer> buffer_list;
  for (size_t i = 0; i < buffer_num; ++i) {
    std::string buffer_name = "buf_" + std::to_string(i + 1);
    Var buf_var = Var(buffer_name, Float(16));
    Buffer buffer = BufferNode::make(buf_var, Float(16), {BOX_PER_INSN * boxnum}, {}, Expr(), buffer_name, SCOPE_UBUF,
                                     0, 0, BufferType::kDefault);
    var_list.push_back(buf_var);
    buffer_list.push_back(buffer);
  }
  Stmt result;

  if (boxnum / BOX_PER_INSN > 1) {
    VarExpr loop_var = VarExpr("i");
    result = EmitIou(loop_var, true, boxnum, src, src, buffer_list[2], buffer_list[0], buffer_list[1]);
    result = InsertBody(result, EmitCor(loop_var, thre, dst, buffer_list[0], buffer_list[2]));
    result = For::make(loop_var, 0, boxnum / BOX_PER_INSN, ForType::Serial, ktvm::ir::DeviceAPI::None, result);
  } else {
    result = EmitIou(Expr(0), true, boxnum, src, src, buffer_list[2], buffer_list[0], buffer_list[1]);
    result = InsertBody(result, EmitCor(Expr(0), thre, dst, buffer_list[0], buffer_list[2]));
  }

  for (size_t i = 0; i < buffer_num; ++i) {
    size_t reverse_idx = buffer_num - 1 - i;
    result = Allocate::make(var_list[reverse_idx], Float(16), {BOX_PER_INSN * boxnum}, const_true(), result);
    result = AttrStmt::make(var_list[reverse_idx], STORAGE_SCOPE, StringImm::make(SCOPE_UBUF), result);
  }

  CommentManager::GetInstance().AddComment("Insn_name", "nms");
  CommentManager::GetInstance().AddComment("Insn_type", "rpn");

  return result;
}

/// Function to emit IOU intrin
/// \param op - The input stmt to be emitted as intrin
/// \return Stmt of emitted CCE intrin
Stmt BinaryIouEmitter(const Stmt &op) {
  CHECK(op);
  Array<Buffer> dst_list;
  Array<Buffer> src_list;
  GetBufferIdFromStmt(op, dst_list, src_list);
  CHECK(!dst_list.empty());
  CHECK_GE(src_list.size(), 2);
  Buffer dst = dst_list[0];
  Buffer src0 = src_list[0];
  Buffer src1 = src_list[1];
  const int BOX_PER_INSN = 16;
  CHECK(!src0->shape.empty());
  CHECK(!src1->shape.empty());
  int boxnum0 = GetInt32Const(src0->shape[0]);
  int boxnum1 = GetInt32Const(src1->shape[0]);
  Var BufAVar = Var("buf_A", Float(16));
  Var BufBVar = Var("buf_B", Float(16));
  Stmt result;
  Stmt stmt;
  Buffer BufferA = BufferNode::make(BufAVar, Float(16), {BOX_PER_INSN * boxnum1}, {}, Expr(), "buf_A", SCOPE_UBUF, 0, 0,
                                    BufferType::kDefault);
  Buffer BufferB = BufferNode::make(BufBVar, Float(16), {BOX_PER_INSN * boxnum1}, {}, Expr(), "buf_B", SCOPE_UBUF, 0, 0,
                                    BufferType::kDefault);

  if (boxnum0 / BOX_PER_INSN > 1) {
    VarExpr loop_var = VarExpr("i");
    stmt = EmitIou(loop_var, false, boxnum1, src0, src1, dst, BufferA, BufferB);
    stmt = For::make(loop_var, Expr(0), boxnum0 / BOX_PER_INSN, ForType::Serial, ktvm::ir::DeviceAPI::None, stmt);
  } else {
    stmt = EmitIou(Expr(0), false, boxnum1, src0, src1, dst, BufferA, BufferB);
  }
  result = InsertBody(result, stmt);

  result = Allocate::make(BufBVar, Float(16), {BOX_PER_INSN * boxnum1}, const_true(), result);
  result = AttrStmt::make(BufBVar, STORAGE_SCOPE, StringImm::make(SCOPE_UBUF), result);
  result = Allocate::make(BufAVar, Float(16), {BOX_PER_INSN * boxnum1}, const_true(), result);
  result = AttrStmt::make(BufAVar, STORAGE_SCOPE, StringImm::make(SCOPE_UBUF), result);

  CommentManager::GetInstance().AddComment("Insn_name", "iou");
  CommentManager::GetInstance().AddComment("Insn_type", "rpn");

  return result;
}

/// Function to package proposal sort emitter
/// \param op
/// \param top_k
/// \return
Stmt BinarySortEmitter(const Stmt &op, bool top_k) {
  Array<Buffer> dst_list;
  Array<Buffer> src_list;
  GetBufferIdFromStmt(op, dst_list, src_list);
  auto stores = GetStores(op);
  CHECK(!stores.empty());
  CHECK(!dst_list.empty());
  CHECK_GE(src_list.size(), 2);
  auto store = stores[0];
  auto dst = dst_list[0];
  auto src = src_list[1];
  return EmitProposalSort(store, src, dst, top_k);
}

/// Function to emit proposal sort intrin
/// \param op - The input stmt to be emitted as intrin
/// \return Stmt of emitted CCE intrin
Stmt BinaryProposalSortEmitter(const Stmt &op) {
  CHECK(op);
  CommentManager::GetInstance().AddComment("Insn_name", "proposal_sort");
  CommentManager::GetInstance().AddComment("Insn_type", "rpn");
  return BinarySortEmitter(op, false);
}

/// Function to emit topk sort intrin
/// \param op - The input stmt to be emitted as intrin
/// \return Stmt of emitted CCE intrin
Stmt BinaryTopkSortEmitter(const Stmt &op) {
  CHECK(op);
  CommentManager::GetInstance().AddComment("Insn_name", "topk");
  CommentManager::GetInstance().AddComment("Insn_type", "rpn");
  return BinarySortEmitter(op, true);
}

/// Function to emit vconv intrin
/// \param op - The input stmt to be emitted as intrin
/// \return Stmt of emitted CCE intrin
Stmt SingleCastEmitter(const Stmt &op) {
  CHECK(op);
  Array<NodeRef> stores;
  Array<NodeRef> loads;
  GetStoreAndLoads(op, stores, loads);
  CHECK(!stores.empty());
  CHECK(!loads.empty());
  auto store = stores[0].as<Store>();
  auto load = loads[0].as<Load>();
  CHECK(store);
  CHECK(load);
  auto src_type = load->type;
  auto dst_type = store->value.type();
  std::string intrin_name = GetConvCmd(src_type, dst_type, "");
  if (intrin_name == "vconv_s322f16") {
    intrin_name = "vconv_deq";
  }
  if (intrin_name == "vconv_f162s32") {
    intrin_name = "vconv_f162s32f";
  }
  if (intrin_name == "vconv_f322s32") {
    intrin_name = "vconv_f322s32r";
  }
  return SingleVecEmitter(op, intrin_name);
}

/// Function to handle select parameters
/// \param dst_info_list
/// \param src_info_list
/// \param for_info
/// \param if_info
/// \param select_name
/// \return Stmt of emitted CCE intrin
Stmt SelectParamHandle(StmtInfoList &dst_info_list, StmtInfoList &src_info_list, StmtInfo &for_info, StmtInfo &if_info,
                       const std::string &select_name) {
  CHECK(!dst_info_list.empty());
  CHECK(!src_info_list.empty());

  Stmt result;
  ArgInfo arg_info = GetMultiVecInsnArgs(dst_info_list, src_info_list, for_info);
  MultiVecInsnBuilder builder = MultiVecInsnBuilder(dst_info_list[0], src_info_list, arg_info, select_name);
  auto insn_list = builder.EmitIntrin();

  return FoldInsnWithForInfo(insn_list, if_info, for_info, result);
}

/// Function to emit mutable mask intrin
/// \param op - The input stmt to be emitted as intrin
/// \return Stmt of emitted CCE intrin
Stmt MutableMaskEmitter(const Stmt &op) {
  CHECK(op.defined());
  auto stores = GetStores(op);
  CHECK(!stores.empty());
  auto store = stores[0].as<Store>();
  CHECK(store);
  CHECK(store->value.as<Select>());
  auto condition = store->value.as<Select>()->condition;
  CHECK(condition->IsInstance<LT>());
  Expr const_value = store->value.as<Select>()->true_value;

  StmtInfoList dst_info_list;
  StmtInfoList src_info_list;
  StmtInfo for_info;
  StmtInfo if_info;
  GetCompactComputationInfo(op, dst_info_list, src_info_list, if_info, for_info, true, false);
  auto dst_info = GetItem(dst_info_list, 0);
  auto src_info0 = GetItem(src_info_list, 0);

  CHECK(!for_info.vars_.empty());
  CHECK_LE(for_info.vars_.size(), 2);
  auto elim_var = for_info.vars_[0];
  Expr loop_extent = for_info.ops_[0].as<For>()->extent;
  if (for_info.vars_.size() == 2) {
    elim_var = for_info.vars_[1];
    loop_extent = for_info.ops_[1].as<For>()->extent;
  }

  // select((cc3 < cc0), 0.000000h, tri_matrix_0_local__ub[cc3]) is upper matrix
  // select((cc0 < cc3), 0.000000h, tri_matrix_0_local__ub[cc3]) is lower matrix
  bool lower = ktvm::ir::Equal(condition.as<LT>()->b, elim_var);
  CleanForInfoVars(for_info, {elim_var});
  auto loop_var = lower ? condition.as<LT>()->a : condition.as<LT>()->b;

  Type dtype = dst_info->dtype_;
  bool is_fp32 = dtype.bits() == 32;

  // Broadcast true value
  Stmt broadcast;
  Var true_buffer_var = Var("true_value_local_UB");
  const int vec_max_len = GetVecMaxLen(dtype);
  Buffer const_buffer = BufferNode::make(true_buffer_var, dtype, {vec_max_len}, {}, Expr(), "true_value_local_UB",
                                         SCOPE_UBUF, 0, 0, BufferType::kDefault);

  if (IsConstExpr(const_value)) {
    broadcast = EmitCceIntrinTemplate(
      Stmt(), dtype, {GetAccessPtr(const_buffer, "w", 0), const_value, Expr(1), Expr(1), Expr(1), Expr(0), Expr(0)},
      INTRIN_NAME_VECTOR_DUP);
  }

  // Copy data
  Expr copy_src = GetAccessPtr(GenBufferId(src_info0), "r", Expr(0));
  Expr dst = GetAccessPtr(GenBufferId(dst_info), "w", Expr(0));
  if (src_info_list.size() == 2) {
    copy_src = GetAccessPtr(GenBufferId(src_info_list[1]), "r", Expr(0));
  }

  int block_size = GetUbBlkSize(dtype);
  CHECK_NE(block_size, 0);
  Stmt insn = EmitCceIntrinTemplate(
    Stmt(), dtype, {dst, copy_src, Expr(0), Expr(1), truncdiv(loop_extent + block_size, block_size), Expr(0), Expr(0)},
    INTRIN_NAME_COPY_UB_TO_UB);

  // Gen mask
  auto mask_var = Var("masks", UInt(64));
  Buffer mask_buffer =
    BufferNode::make(mask_var, UInt(64), {4}, {}, Expr(), "masks", SCOPE_REG, 0, 0, BufferType::kDefault);

  MutableMaskParams params;
  params.loop_var_ = loop_var;
  params.mask_var_ = mask_var;
  params.loop_extent_ = loop_extent;
  params.lower_ = lower;
  params.is_fp32_ = is_fp32;
  params.broadcast_ = broadcast;
  params.const_buffer_ = const_buffer;

  insn = EmitMutableMaskGen(insn, params);
  insn = EmitMutableMaskVec(insn, dst_info_list, src_info_list, params);

  Stmt result;
  result = FoldInsnWithForInfo({insn}, if_info, for_info, result);

  if (broadcast.defined()) {
    result = InsertBody(broadcast, result);
    result =
      AttrStmt::make(const_buffer->data, STORAGE_SCOPE, Expr(SCOPE_UBUF),
                     Allocate::make(const_buffer->data, const_buffer->dtype, {vec_max_len}, const_true(), result));
  }

  int reg_size = lower ? 2 : 4;
  result = AttrStmt::make(mask_buffer->data, STORAGE_SCOPE, Expr(SCOPE_REG),
                          Allocate::make(mask_buffer->data, mask_buffer->dtype, {reg_size}, const_true(), result));

  CommentManager::GetInstance().AddComment("Insn_name", "triangle");
  CommentManager::GetInstance().AddComment("Insn_type", "multi_vector");

  return result;
}

/// Function to emit select intrin
/// \param op - The input stmt to be emitted as intrin
/// \return Stmt of emitted CCE intrin
Stmt SelectWithScalarEmitter(const Stmt &op) {
  CHECK(op);
  StmtInfo if_info;
  StmtInfo for_info;
  StmtInfoList src_info_list;
  StmtInfoList dst_info_list;
  Stmt tmp_op = op;
  std::string select_name;

  GetCompactComputationInfo(op, dst_info_list, src_info_list, if_info, for_info, true);
  CommentManager::GetInstance().AddComment("Insn_name", "select");

  CHECK(!src_info_list.empty());
  CHECK(!dst_info_list.empty());
  auto dst_info = dst_info_list[0];
  int block_size = GetUbBlkSize(dst_info->dtype_);
  int val_len = block_size * 8;

  while (tmp_op->IsInstance<For>() || tmp_op->IsInstance<IfThenElse>()) {
    if (tmp_op.as<For>()) {
      tmp_op = tmp_op.as<For>()->body;
    } else if (tmp_op.as<IfThenElse>()) {
      tmp_op = tmp_op.as<IfThenElse>()->then_case;
    }
  }

  CHECK(tmp_op.as<Store>());
  auto sel = tmp_op.as<Store>()->value.as<Select>();
  CHECK(sel);
  auto sel_type = sel->condition;
  auto true_value = sel->true_value;
  auto false_value = sel->false_value;

  Expr condition_a;
  Expr condition_b;
  Expr tmp_cond;
  if (sel_type->IsInstance<LT>()) {
    select_name = "vselect_LT";
    CHECK(sel->condition.as<LT>());
    condition_a = sel->condition.as<LT>()->a;
    condition_b = sel->condition.as<LT>()->b;
  } else if (sel_type->IsInstance<EQ>()) {
    select_name = "vselect_EQ";
    CHECK(sel->condition.as<EQ>());
    condition_a = sel->condition.as<EQ>()->a;
    condition_b = sel->condition.as<EQ>()->b;
  } else if (sel_type->IsInstance<GT>()) {
    select_name = "vselect_GT";
    CHECK(sel->condition.as<GT>());
    condition_a = sel->condition.as<GT>()->a;
    condition_b = sel->condition.as<GT>()->b;
  } else if (sel_type.as<Cast>() || sel_type->IsInstance<And>() || sel_type->IsInstance<Or>() ||
             sel_type->IsInstance<Not>()) {
    // Special case: A = vselect(bool, 0.0f, 1.0f)
    CommentManager::GetInstance().AddComment("Insn_type", "scalar");
    return op;
  }

  if (condition_a->IsInstance<Variable>()) {
    if (condition_b->IsInstance<Variable>()) {
      return MutableMaskEmitter(op);
    }
    CommentManager::GetInstance().AddComment("Insn_type", "scalar");
    return op;
  }

  if ((!condition_a->IsInstance<Load>() && !GetVarsInExpr(condition_a).empty()) ||
      (!condition_b->IsInstance<Load>() && !GetVarsInExpr(condition_b).empty())) {
    CommentManager::GetInstance().AddComment("Insn_type", "scalar");
    return op;
  }

  Stmt result;
  std::string cond_buffer_name = "condition_local_UB";
  Var cond_buffer_var = Var(cond_buffer_name, Handle());
  Buffer cond_buffer = BufferNode::make(cond_buffer_var, condition_a.type(), {val_len}, Array<Expr>(), Expr(),
                                        cond_buffer_name, SCOPE_UBUF, 0, 0, BufferType::kDefault);

  std::string true_buffer_name = "true_local_UB";
  Var true_buffer_var = Var(true_buffer_name, Handle());
  Buffer true_buffer = BufferNode::make(true_buffer_var, true_value.type(), {val_len}, Array<Expr>(), Expr(),
                                        true_buffer_name, SCOPE_UBUF, 0, 0, BufferType::kDefault);

  std::string false_buffer_name = "false_local_UB";
  Var false_buffer_var = Var(false_buffer_name, Handle());
  Buffer false_buffer = BufferNode::make(false_buffer_var, false_value.type(), {val_len}, Array<Expr>(), Expr(),
                                         false_buffer_name, SCOPE_UBUF, 0, 0, BufferType::kDefault);

  auto CopyAndResetComInfo = [&src_info_list](const Buffer &buffer, const Type &type) -> StmtStoreInfo {
    StmtStoreInfo new_info = src_info_list[0].Copy();
    new_info.GetNode()->insn_offset_ = Expr(0);
    new_info.GetNode()->buffer_ = buffer;
    new_info.GetNode()->dtype_ = type;
    return new_info;
  };

  StmtStoreInfo condition_info = CopyAndResetComInfo(cond_buffer, condition_a.type());
  StmtStoreInfo true_info = CopyAndResetComInfo(true_buffer, true_value.type());
  StmtStoreInfo false_info = CopyAndResetComInfo(false_buffer, false_value.type());

  if (IsConstExpr(condition_a)) {
    tmp_cond = condition_a;
  } else if (IsConstExpr(condition_b)) {
    tmp_cond = condition_b;
  }

  // Case 1: E = vselect(A < B, C, D)
  // Case 2: D = vselect(1.0f < A, B, C)
  if (!IsConstExpr(true_value) && !IsConstExpr(false_value)) {
    // E = vselect(A < B, C, D)
    if (!IsConstExpr(condition_a) && !IsConstExpr(condition_b)) {
      ArgInfo arg_info = GetMultiVecInsnArgs(dst_info_list, src_info_list, for_info);

      MultiVecInsnBuilder builder = MultiVecInsnBuilder(dst_info_list[0], src_info_list, arg_info, select_name);
      auto insn_list = builder.EmitIntrin();

      result = FoldInsnWithForInfo(insn_list, if_info, for_info, Stmt());
    } else if ((IsConstExpr(condition_a) && !IsConstExpr(condition_b)) ||
               (!IsConstExpr(condition_a) && IsConstExpr(condition_b))) {
      // D = vselect(1.0f < A, B, C)
      if (IsConstExpr(condition_a)) {
        src_info_list = {condition_info, src_info_list[0], src_info_list[1], src_info_list[2]};
      } else if (IsConstExpr(condition_b)) {
        src_info_list = {src_info_list[0], condition_info, src_info_list[1], src_info_list[2]};
      }

      result = EmitSetVecMaskIntrin(Stmt(), condition_a.type());
      result = InsertBody(result, Evaluate::make(Call::make(condition_a.type(), INTRIN_NAME_VECTOR_DUP,
                                                            {GetAccessPtr(cond_buffer, "w", Expr(0)), tmp_cond, Expr(1),
                                                             Expr(1), Expr(1), Expr(8), Expr(8)},
                                                            Call::Extern)));

      result = InsertBody(result, SelectParamHandle(dst_info_list, src_info_list, for_info, if_info, select_name));

      result =
        Allocate::make(cond_buffer_var, condition_a.type(), {make_const(Int(32), val_len)}, const_true(), result);
      result = AttrStmt::make(cond_buffer_var, STORAGE_SCOPE, Expr("local.UB"), result);
    }
  } else if ((IsConstExpr(true_value) && !IsConstExpr(false_value)) ||
             (!IsConstExpr(true_value) && IsConstExpr(false_value))) {
    // Case 3: D = vselect(A < B, C, 0.0f) or D = vselect(A < B, 0.0f, C)
    // Case 4: C = vselect(1.0f < A, B, 0.0f) or C = vselect(1.0f < A, 0.0f, B)
    Expr tmp_value = true_value;
    auto tmp_buffer = true_buffer;
    auto buffer_var = true_buffer_var;
    StmtInfoList new_src_info_list;
    if (IsConstExpr(false_value)) {
      tmp_value = false_value;
      tmp_buffer = false_buffer;
      buffer_var = false_buffer_var;
    }

    bool case_c = (IsConstExpr(condition_a) && !IsConstExpr(condition_b)) ||
                  (!IsConstExpr(condition_a) && IsConstExpr(condition_b));
    result = EmitSetVecMaskIntrin(Stmt(), tmp_value.type());
    result = InsertBody(result, Evaluate::make(Call::make(tmp_value.type(), INTRIN_NAME_VECTOR_DUP,
                                                          {GetAccessPtr(tmp_buffer, "w", Expr(0)), tmp_value, Expr(1),
                                                           Expr(1), Expr(1), Expr(8), Expr(8)},
                                                          Call::Extern)));
    if (case_c) {
      // C = vselect(1.0f < A, B, 0.0f) or C = vselect(1.0f < A, 0.0f, B)
      result = InsertBody(result, EmitSetVecMaskIntrin(Stmt(), condition_a.type()));
      result = InsertBody(result, Evaluate::make(Call::make(condition_a.type(), INTRIN_NAME_VECTOR_DUP,
                                                            {GetAccessPtr(cond_buffer, "w", Expr(0)), tmp_cond, Expr(1),
                                                             Expr(1), Expr(1), Expr(8), Expr(8)},
                                                            Call::Extern)));

      if (IsConstExpr(condition_a) && IsConstExpr(true_value)) {
        new_src_info_list = {condition_info, src_info_list[0], true_info, src_info_list[1]};
      } else if (IsConstExpr(condition_a) && IsConstExpr(false_value)) {
        new_src_info_list = {condition_info, src_info_list[0], src_info_list[1], false_info};
      } else if (IsConstExpr(condition_b) && IsConstExpr(true_value)) {
        new_src_info_list = {src_info_list[0], condition_info, true_info, src_info_list[1]};
      } else if (IsConstExpr(condition_b) && IsConstExpr(false_value)) {
        new_src_info_list = {src_info_list[0], condition_info, src_info_list[1], false_info};
      }
    } else {
      if (IsConstExpr(true_value)) {
        new_src_info_list = {src_info_list[0], src_info_list[1], true_info, src_info_list[2]};
      } else if (IsConstExpr(false_value)) {
        new_src_info_list = {src_info_list[0], src_info_list[1], src_info_list[2], false_info};
      }
    }

    result = InsertBody(result, SelectParamHandle(dst_info_list, new_src_info_list, for_info, if_info, select_name));

    if (case_c) {
      result =
        Allocate::make(cond_buffer_var, condition_a.type(), {make_const(Int(32), val_len)}, const_true(), result);
      result = AttrStmt::make(cond_buffer_var, STORAGE_SCOPE, Expr("local.UB"), result);
    }

    result = Allocate::make(buffer_var, true_value.type(), {make_const(Int(32), val_len)}, const_true(), result);
    result = AttrStmt::make(buffer_var, STORAGE_SCOPE, Expr("local.UB"), result);
  } else if (IsConstExpr(true_value) && IsConstExpr(false_value)) {
    // Case 5: C = vselect(A < B, 0.0f, 1.0f)
    // Case 6: B = vselect(1.0f < A, 0.0f, 2.0f) or B = vselect(A < 1.0f, 0.0f, 2.0f)
    // C = vselect(A < B, 0.0f, 1.0f)
    bool case_b = (IsConstExpr(condition_a) && !IsConstExpr(condition_b)) ||
                  (!IsConstExpr(condition_a) && IsConstExpr(condition_b));
    if (case_b) {
      // B = vselect(1.0f < A, 0.0f, 2.0f) or B = vselect(A < 1.0f, 0.0f, 2.0f)
      if (IsConstExpr(condition_a)) {
        src_info_list = {condition_info, src_info_list[0]};
      } else if (IsConstExpr(condition_b)) {
        src_info_list = {src_info_list[0], condition_info};
      }

      result = EmitSetVecMaskIntrin(Stmt(), condition_a.type());
      result = InsertBody(result, Evaluate::make(Call::make(condition_a.type(), INTRIN_NAME_VECTOR_DUP,
                                                            {GetAccessPtr(cond_buffer, "w", Expr(0)), tmp_cond, Expr(1),
                                                             Expr(1), Expr(1), Expr(8), Expr(8)},
                                                            Call::Extern)));
    }

    result = InsertBody(result, EmitSetVecMaskIntrin(Stmt(), true_value.type()));
    result = InsertBody(result, Evaluate::make(Call::make(true_value.type(), INTRIN_NAME_VECTOR_DUP,
                                                          {GetAccessPtr(true_buffer, "w", Expr(0)), true_value, Expr(1),
                                                           Expr(1), Expr(1), Expr(8), Expr(8)},
                                                          Call::Extern)));

    result = InsertBody(result, EmitSetVecMaskIntrin(Stmt(), false_value.type()));
    result = InsertBody(result, Evaluate::make(Call::make(false_value.type(), INTRIN_NAME_VECTOR_DUP,
                                                          {GetAccessPtr(false_buffer, "w", Expr(0)), false_value,
                                                           Expr(1), Expr(1), Expr(1), Expr(8), Expr(8)},
                                                          Call::Extern)));

    src_info_list.push_back(true_info);
    src_info_list.push_back(false_info);

    result = InsertBody(result, SelectParamHandle(dst_info_list, src_info_list, for_info, if_info, select_name));
    if (case_b) {
      result =
        Allocate::make(cond_buffer_var, condition_a.type(), {make_const(Int(32), val_len)}, const_true(), result);
      result = AttrStmt::make(cond_buffer_var, STORAGE_SCOPE, Expr("local.UB"), result);
    }
    result = Allocate::make(true_buffer_var, true_value.type(), {make_const(Int(32), val_len)}, const_true(), result);
    result = AttrStmt::make(true_buffer_var, STORAGE_SCOPE, Expr("local.UB"), result);
    result = Allocate::make(false_buffer_var, false_value.type(), {make_const(Int(32), val_len)}, const_true(), result);
    result = AttrStmt::make(false_buffer_var, STORAGE_SCOPE, Expr("local.UB"), result);
  }

  CHECK(result.defined()) << "Error: Can not support this kind of operation.";
  CommentManager::GetInstance().AddComment("Insn_type", "multi_vector");

  return result;
}

/// Template of CCE scalar DMA intrin emitter
/// \param op - The input stmt to be emitted as intrin
/// \param src_info
/// \param dst_info
/// \return Stmt of emitted CCE intrin
Stmt EmitScalarDmaIntrinTemplate(const Stmt &op, const StmtStoreInfo &src_info, const StmtStoreInfo &dst_info) {
  CHECK(op);
  Array<NodeRef> stores;
  Array<NodeRef> loads;
  GetStoreAndLoads(op, stores, loads);
  CHECK(!stores.empty());
  CHECK(!loads.empty());
  auto load = loads[0].as<Load>();
  auto store = stores[0].as<Store>();
  CHECK(load);
  CHECK(store);

  auto load_var = Load::make(src_info->dtype_, load->buffer_var, src_info->index_, const_true());
  return Store::make(store->buffer_var, load_var, dst_info->index_, const_true());
}

/// Function to emit dma copy intrin
/// \param op - The input stmt to be emitted as intrin
/// \param enable_cover_protect - Enable cover protection optimization
/// \return Stmt of emitted CCE intrin
Stmt DmaMovEmitter(const Stmt &op, bool enable_cover_protect) {
  CHECK(op);
  std::string dma_mode;
  std::string intrin_name;
  StmtInfoList dst_info_list;
  StmtInfoList src_info_list;
  StmtInfo if_info;
  StmtInfo for_info;
  GetDmaComputationInfo(op, dst_info_list, src_info_list, if_info, for_info, dma_mode, intrin_name);

  auto check_alignment = [](const Expr &align, const Array<Expr> &shape) {
    if (GetIntConst(align) == 1 || shape.size() == 1u) {
      return true;
    }

    if (shape.empty()) {
      return false;
    }
    Expr sz = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
      sz = sz * shape[i];
      if (GetIntConst(align) == GetIntConst(sz)) {
        return true;
      }
    }
    return false;
  };

  const auto &dst_info = dst_info_list[0];
  const auto &src_info = src_info_list[0];
  int block_size = GetUbBlkSize(dst_info->dtype_);

  // check scalar to scalar
  // check if dst is considered as scalar
  // check if src is considered as scalar
  bool is_broadcast =
    (dst_info->var_.empty() || (!dst_info->strides_.empty() && GetIntConst(GetItem(dst_info->strides_, -1)) != 1)) &&
    (src_info->var_.empty() || (!src_info->strides_.empty() && GetIntConst(GetItem(src_info->strides_, -1)) != 1));
  // check vector to vector, but in scalar dma mode
  bool last_dim_equal = !dst_info->var_.empty() && !src_info->var_.empty() && !dst_info->strides_.empty() &&
                        !src_info->strides_.empty() &&
                        GetItem(dst_info->var_, -1).get() == GetItem(src_info->var_, -1).get() &&
                        GetIntConst(GetItem(dst_info->strides_, -1)) != GetIntConst(GetItem(src_info->strides_, -1));
  bool broadcast_scalar = intrin_name == "broadcast" && is_broadcast;
  bool ubuf_scalar = intrin_name == INTRIN_NAME_COPY_UB_TO_UB && (is_broadcast || last_dim_equal);

  if (broadcast_scalar || ubuf_scalar) {
    int shape1 = GetInt32Const(GetItem(dst_info->shape_, -1));
    int stride1 = GetInt32Const(GetItem(dst_info->strides_, -1));
    if (ubuf_scalar && shape1 < block_size && stride1 == block_size &&
        IsTwoItemEqual(dst_info->strides_, src_info->strides_, -1, true) && src_info->dtype_.bits() != 64) {
      // if last dim small than blocksize, then use vadds
      return SingleVecEmitter(op, intrin_name);
    }
    CommentManager::GetInstance().AddComment("Insn_type", "dma_copy");
    CommentManager::GetInstance().AddComment("Insn_name", "scalar");
    if (src_info->var_.empty() && dst_info->var_.empty()) {
      return op;
    } else {
      // check align
      if (!check_alignment(dst_info->data_alignment_, dst_info->shape_)) {
        return op;
      }
      Stmt base_stmt = EmitScalarDmaIntrinTemplate(op, src_info, dst_info);
      return GenIfAndFor(base_stmt, if_info, for_info, false);
    }
  }

  if (intrin_name == "broadcast") {
    return SingleVecEmitter(op, INTRIN_NAME_VECTOR_DUP);
  } else if (intrin_name == INTRIN_NAME_COPY_UB_TO_UB) {
    // Use vadds to optimize dma copy
    if (if_info.vars_.empty() && dst_info->dtype_.is_float() && src_info->dtype_.is_float()) {
      if ((dst_info->dtype_.bits() == 32 && src_info->dtype_.bits() == 32) ||
          (dst_info->dtype_.bits() == 16 && src_info->dtype_.bits() == 16)) {
        int repeat_len = block_size * FULL_BLOCK_NUM;
        CHECK_NE(block_size, 0);
        int shape1 = GetInt32Const(GetItem(dst_info->shape_, -1));
        if ((shape1 >= repeat_len / 2 && shape1 <= repeat_len) ||
            (dst_info->shape_.size() >= 3 && shape1 <= block_size) ||
            (dst_info->shape_.size() >= 2 && shape1 % block_size == 0)) {
          // if last dim shape is too small, there is no need to opt
          return SingleVecEmitter(op, intrin_name);
        }
      }
    }
  }

  CommentManager::GetInstance().AddComment("Insn_type", "dma_copy");

  Stmt base_stmt;
  if (dma_mode == "cce_copy") {
    Map<std::string, Expr> ub_copy_pre;
    Map<std::string, Expr> ub_copy_post;
    auto arg_info_map =
      GetDmaCopyInsnArgs(intrin_name, dst_info_list, src_info_list, for_info, ub_copy_pre, ub_copy_post);
    if (intrin_name == "vtranspose_scalar") {
      base_stmt = EmitScalarDmaIntrinTemplate(op, src_info, dst_info);
      CommentManager::GetInstance().AddComment("Insn_name", "scalar");
    } else if (intrin_name == "vtranspose") {
      Array<Expr> args = {arg_info_map["loop_width"], arg_info_map["loop_height"], arg_info_map["shape_width"]};
      Array<Expr> pre_ub_copy_args;
      if (!ub_copy_pre.empty()) {
        pre_ub_copy_args = Array<Expr>(
          {ub_copy_pre["nBurst"], ub_copy_pre["lenBurst"], ub_copy_pre["srcStride"], ub_copy_pre["dstStride"]});
      }
      Array<Expr> post_ub_copy_args;
      if (!ub_copy_post.empty()) {
        post_ub_copy_args = Array<Expr>(
          {ub_copy_post["nBurst"], ub_copy_post["lenBurst"], ub_copy_post["srcStride"], ub_copy_post["dstStride"]});
      }
      TransposeInsnBuilder builder =
        TransposeInsnBuilder(dst_info, src_info, args, pre_ub_copy_args, post_ub_copy_args);
      base_stmt = builder.EmitSingleIntrin();
      CommentManager::GetInstance().AddComment("Insn_name", intrin_name);
    } else {
      DmaInsnBuilder dma_builder =
        DmaInsnBuilder(dst_info, src_info, intrin_name, arg_info_map, false, false, enable_cover_protect);
      base_stmt = dma_builder.EmitSingleIntrin();
      CommentManager::GetInstance().AddComment("Insn_name", intrin_name);
    }
  } else if (dma_mode == "cce_load") {
    auto arg_info_map = GetDmaLoad2DInsnArgs(intrin_name, dst_info_list, src_info_list, for_info);
    DmaInsnBuilder builder = DmaInsnBuilder(dst_info, src_info, intrin_name, arg_info_map, true);
    base_stmt = builder.EmitSingleIntrin();
    CommentManager::GetInstance().AddComment("Insn_name", intrin_name);
  } else {
    LOG(FATAL) << "Unsupported dma mode " + dma_mode;
  }
  return GenIfAndFor(base_stmt, if_info, for_info, false);
}

/// Function to emit dma atomic add intrin
/// \param op - The input stmt to be emitted as intrin
/// \return Stmt of emitted CCE intrin
Stmt DmaAtomicAddEmitter(const Stmt &op) {
  CHECK(op);
  std::string intrin_name = "copy_ubuf_to_gm";
  StmtInfoList org_dst_info_list;
  StmtInfoList org_src_info_list;
  StmtInfoList dst_info_list;
  StmtInfoList src_info_list;
  StmtInfo if_info;
  StmtInfo for_info;
  GetCompactComputationInfo(op, org_dst_info_list, org_src_info_list, if_info, for_info);

  if (org_dst_info_list.size() == 1u && org_src_info_list.size() == 2u) {
    if (Equal(org_dst_info_list[0]->data_, org_src_info_list[0]->data_)) {
      dst_info_list.push_back(org_dst_info_list[0]);
      src_info_list.push_back(org_src_info_list[1]);
    } else if (Equal(org_dst_info_list[0]->data_, org_src_info_list[1]->data_)) {
      dst_info_list.push_back(org_dst_info_list[0]);
      src_info_list.push_back(org_src_info_list[0]);
    } else {
      LOG(FATAL) << "Error: The IR of DMA Atomic Add is wrong, not support A = B + C, please check.";
    }
  } else {
    LOG(FATAL) << "Error: The IR of DMA Atomic Add is wrong, please check.";
  }

  if (src_info_list[0]->scope_ == SCOPE_UBUF && dst_info_list[0]->scope_ == DMA_COPY_GLOBAL) {
    intrin_name = "copy_ubuf_to_gm";
  } else {
    LOG(FATAL) << "Error: The Buffer scopes of DMA Atomic Add is wrong, please check.";
  }

  CommentManager::GetInstance().AddComment("Insn_type", "dma_copy");
  CommentManager::GetInstance().AddComment("Insn_name", intrin_name);
  CommentManager::GetInstance().AddComment("Atomic_add", "enable");

  const auto &dst_info = dst_info_list[0];
  const auto &src_info = src_info_list[0];

  auto arg_info_map = GetDmaCopyInsnArgs(intrin_name, dst_info_list, src_info_list, for_info);

  const bool is_atomic_add = true;
  DmaInsnBuilder dma_builder = DmaInsnBuilder(dst_info, src_info, intrin_name, arg_info_map, false, is_atomic_add);
  Stmt base_stmt = dma_builder.EmitSingleIntrin();
  auto stmt = GenIfAndFor(base_stmt, if_info, for_info, false);
  auto config_atomic_open = Evaluate::make(Call::make(UInt(64), "set_atomic_add_open", {}, Call::Extern));
  auto config_atomic_close = Evaluate::make(Call::make(UInt(64), "set_atomic_add_close", {}, Call::Extern));
  stmt = InsertBody(config_atomic_open, stmt);
  stmt = InsertBody(stmt, config_atomic_close);
  return stmt;
}

/// Function to emit dropout intrin
/// \param op - The input stmt to be emitted as intrin
/// \return Stmt of emitted CCE intrin
Stmt BinaryDropoutEmitter(const Stmt &op) {
  CHECK(op);
  Array<NodeRef> stores;
  Array<NodeRef> loads;
  GetStoreAndLoads(op, stores, loads);
  CHECK_EQ(loads.size(), 2);

  const Load *mask = nullptr;
  for (size_t i = 0; i != loads.size();) {
    auto op_load = loads[i].as<Load>();
    CHECK(op_load != nullptr);
    if (op_load->type.code() == kDLUInt) {
      mask = op_load;
      loads = RemoveItemAtIndex(loads, i);
    } else {
      ++i;
    }
  }
  CHECK(mask != nullptr);
  CHECK_EQ(loads.size(), 1) << "There must be only one input.";

  StmtInfo for_info;
  StmtInfo if_info;
  GetIfForInfo(op, if_info, for_info);

  StmtInfoList dst_info_list = GetComputationInfo(stores, for_info);
  StmtInfoList src_info_list = GetComputationInfo(loads, for_info);

  CompactComputationInfoList(dst_info_list, src_info_list, if_info, for_info);
  auto src0 = src_info_list[0];
  CHECK_EQ(src0->var_.size(), 1);

  auto src1 = src0.Copy();
  CHECK(src1.GetNode());
  src1.GetNode()->shape_.Set(0, Simplify(truncdiv((src0->shape_[0] + BITS_PER_BYTE - 1), BITS_PER_BYTE)));
  src1.GetNode()->index_ = truncdiv(src0->index_, BITS_PER_BYTE);
  src1.GetNode()->dtype_ = mask->type;
  src1.GetNode()->scope_ = GetBufScope(mask->buffer_var->name_hint);
  src1.GetNode()->name_ = mask->buffer_var->name_hint;
  src1.GetNode()->data_ = mask->buffer_var;
  src1.GetNode()->data_alignment_ = GetInt32Const(mask->predicate);

  SingleVecPatternGenerator generator = SingleVecPatternGenerator(dst_info_list, src_info_list, for_info, "elewise");
  auto params = generator.GetInsnArgs();
  dst_info_list = params.dst_info_list;
  src_info_list = params.src_info_list;
  for_info = params.for_info;
  ArgInfo arg_info = params.arg_info;
  CHECK_EQ(arg_info->pattern_, PATTERN_1D);

  auto swap_func = [](VectorArgInfoNode *ptr) {
    CHECK(ptr->body_num_ <= 1) << "Value: " << ptr->body_num_;
    ptr->body_num_ = GetInt32Const(ptr->repeat_);
    CHECK_GT(ptr->body_num_, 0);
    ptr->repeat_ = Expr(1);
  };
  if (arg_info.GetNode()->body_arg_info_.defined()) {
    swap_func(arg_info.GetNode()->body_arg_info_.GetNode());
  }
  if (arg_info.GetNode()->tail_arg_info_.defined()) {
    swap_func(arg_info.GetNode()->tail_arg_info_.GetNode());
  }

  CommentManager::GetInstance().AddComment("Insn_name", "dropout");

  return EmitDropout(dst_info_list, src_info_list, src1, arg_info, if_info, for_info);
}

/// For variable in inner loop vars:
/// Replace "var == 0" to true.
/// Replace "0 == var" to true.
/// Report error on other cases.
class EliminateVarsInCondExprMutator : public IRMutator {
 public:
  explicit EliminateVarsInCondExprMutator(const std::unordered_set<const Variable *> &vars_) : vars(vars_) {}
  ~EliminateVarsInCondExprMutator() override = default;

  Expr Mutate_(const EQ *op, const Expr &e) override {
    if (op->a.as<Variable>() && vars.count(op->a.as<Variable>()) > 0 && op->b.as<IntImm>() &&
        op->b.as<IntImm>()->value == 0) {
      return const_true();
    }
    if (op->b.as<Variable>() && vars.count(op->b.as<Variable>()) > 0 && op->a.as<IntImm>() &&
        op->a.as<IntImm>()->value == 0) {
      return const_true();
    }
    return IRMutator::Mutate_(op, e);
  }

  Expr Mutate_(const Variable *op, const Expr &e) override {
    CHECK(!vars.count(op)) << "found unknown inner loop var " << op->name_hint << " in IF condition of MAD";
    return IRMutator::Mutate_(op, e);
  }

 private:
  const std::unordered_set<const Variable *> vars;
};

Expr EliminateVarsInCondExpr(const Expr &e, const std::unordered_set<const Variable *> &vars) {
  return EliminateVarsInCondExprMutator(vars).Mutate(e);
}

/// Function to emit mad intrin
/// \param op - The input stmt to be emitted as intrin
/// \return Stmt of emitted CCE intrin
Stmt MadEmitter(const Stmt &op) {
  CHECK(op);
  Array<Buffer> dst;
  Array<Buffer> src;
  GetBufferIdFromStmt(op, dst, src);
  Array<Expr> m = {Expr(0)};
  Array<Expr> k = {Expr(0)};
  Array<Expr> n = {Expr(0)};
  int init = {1};
  const std::string intrin_name = "mad";
  Expr condition = const_false();
  Type out_dtype = Float(16);
  std::unordered_set<const Variable *> loop_vars;

  PackedFunc _PostOrder =
    PackedFunc([&m, &k, &n, &init, &condition, &out_dtype, &loop_vars](const TVMArgs &args, TVMRetValue *ret) {
      Stmt st = args[0];
      if (st.as<AttrStmt>()) {
        auto ptr = st.as<AttrStmt>();
        if (ptr->attr_key == "pragma_mad_m") {
          m.Set(0, ptr->value);
        } else if (ptr->attr_key == "pragma_mad_k") {
          k.Set(0, ptr->value);
        } else if (ptr->attr_key == "pragma_mad_n") {
          n.Set(0, ptr->value);
        } else if (ptr->attr_key == "pragma_gemm_out_dtype") {
          std::string type_str = ptr->value.as<StringImm>()->value;
          if (type_str.find("float") == 0) {
            CHECK_GE(type_str.size(), 5);
            out_dtype = Float(std::strtol(type_str.substr(5).c_str(), nullptr, 0));
          } else if (type_str.find("int") == 0) {
            CHECK_GE(type_str.size(), 3);
            out_dtype = Int(std::strtol(type_str.substr(3).c_str(), nullptr, 0));
          } else if (type_str.find("uint") == 0) {
            CHECK_GE(type_str.size(), 4);
            out_dtype = UInt(std::strtol(type_str.substr(4).c_str(), nullptr, 0));
          } else {
            LOG(FATAL) << "Unsupported type string type " << type_str;
          }
        } else if (ptr->attr_key == "init") {
          init = static_cast<int>(ptr->value.as<IntImm>()->value);
        }
      }
      if (st.as<Store>()) {
        auto ptr = st.as<Store>();
        if (ptr->value->IsInstance<IntImm>() || ptr->value->IsInstance<UIntImm>()) {
          if (GetIntConst(ptr->value) == 0) {
            condition = const_true();
          }
        } else if (ptr->value.as<FloatImm>()) {
          if (ptr->value.as<FloatImm>()->value == 0.0f) {
            condition = const_true();
          }
        }
      }
      if (st.as<IfThenElse>()) {
        Expr cond = EliminateVarsInCondExpr(st.as<IfThenElse>()->condition, loop_vars);
        condition = And::make(condition, cond);
      }
      if (st.as<For>()) {
        loop_vars.erase(st.as<For>()->loop_var.get());
      }
    });
  ktvm::runtime::PackedFunc _PreOrder = ktvm::runtime::PackedFunc([&loop_vars](const TVMArgs &args, TVMRetValue *ret) {
    Stmt st = args[0];
    if (st.as<For>()) {
      loop_vars.insert(st.as<For>()->loop_var.get());
    }
  });
  Array<Expr> only_enable = {Expr("AttrStmt"), Expr("IfThenElse"), Expr("Store"), Expr("For")};
  static_cast<void>(ktvm::ir::IRTransform(op, _PreOrder, _PostOrder, only_enable));

  // wgt shape
  const int k_wgt_lanes = WGT_ELEM_BYTES * 8 / WGT_WIDTH;
  CHECK(k_wgt_lanes == BLOCK_OUT * BLOCK_REDUCE);
  Array<Expr> wgt_shape;
  if (Equal(Simplify(FloorDiv::make(n[0], BLOCK_OUT)), 0)) {
    wgt_shape = {truncdiv(k[0], BLOCK_REDUCE), 1, truncmod(n[0], BLOCK_OUT), BLOCK_REDUCE};
    CHECK(GetIntConst(wgt_shape[2] * wgt_shape[3]) < k_wgt_lanes);
  } else {
    wgt_shape = {truncdiv(k[0], BLOCK_REDUCE), truncdiv(n[0], BLOCK_OUT), BLOCK_OUT, BLOCK_REDUCE};
    CHECK(GetIntConst(wgt_shape[2] * wgt_shape[3]) == k_wgt_lanes);
  }

  // inp shape
  const int k_inp_lanes = INP_ELEM_BYTES * 8 / INP_WIDTH;
  CHECK(k_inp_lanes == BLOCK_IN * BLOCK_REDUCE);
  Array<Expr> inp_shape;
  if (Equal(Simplify(FloorDiv::make(m[0], BLOCK_OUT)), 0)) {
    inp_shape = {Expr(1), truncdiv(k[0], BLOCK_REDUCE), truncmod(m[0], BLOCK_IN), BLOCK_REDUCE};
    CHECK(GetIntConst(inp_shape[2] * inp_shape[3]) < k_inp_lanes);
  } else {
    inp_shape = {truncdiv(m[0], BLOCK_IN), truncdiv(k[0], BLOCK_REDUCE), BLOCK_IN, BLOCK_REDUCE};
    CHECK(GetIntConst(inp_shape[2] * inp_shape[3]) == k_inp_lanes);
  }
  CHECK(ktvm::ir::Equal(inp_shape[1], wgt_shape[0]));
  CHECK(ktvm::ir::Equal(inp_shape[3], wgt_shape[3]));

  // out shape
  const int k_out_lanes = OUT_ELEM_BYTES * 8 / OUT_WIDTH;
  CHECK(k_out_lanes == BLOCK_OUT * BLOCK_IN);
  Array<Expr> out_shape;
  if (Equal(Simplify(FloorDiv::make(n[0], BLOCK_OUT)), 0)) {
    out_shape = {Expr(1), truncdiv(m[0], BLOCK_IN), BLOCK_IN, truncmod(n[0], BLOCK_OUT)};
    CHECK(GetIntConst(out_shape[2] * out_shape[3]) < k_out_lanes);
  } else if (Equal(Simplify(FloorDiv::make(m[0], BLOCK_OUT)), 0)) {
    out_shape = {truncdiv(n[0], BLOCK_OUT), Expr(1), truncmod(m[0], BLOCK_IN), BLOCK_OUT};
    CHECK(GetIntConst(out_shape[2] * out_shape[3]) < k_out_lanes);
  } else {
    out_shape = {truncdiv(n[0], BLOCK_OUT), truncdiv(m[0], BLOCK_IN), BLOCK_IN, BLOCK_OUT};
    CHECK(GetIntConst(out_shape[2] * out_shape[3]) == k_out_lanes);
  }
  CHECK(ktvm::ir::Equal(out_shape[0], wgt_shape[1]));
  CHECK(ktvm::ir::Equal(out_shape[1], inp_shape[0]));
  Buffer dwgt = BufferNode::make(src[2]->data, Float(WGT_WIDTH), wgt_shape, {}, Expr(0), src[2]->name, SCOPE_CB,
                                 k_wgt_lanes, k_wgt_lanes, BufferType::kDefault);
  Buffer dinp = BufferNode::make(src[1]->data, Float(INP_WIDTH), inp_shape, {}, Expr(0), src[1]->name, SCOPE_CA,
                                 k_inp_lanes, k_inp_lanes, BufferType::kDefault);
  Buffer dout = BufferNode::make(src[0]->data, out_dtype, out_shape, {}, Expr(0), dst[0]->name, SCOPE_CC, k_out_lanes,
                                 k_out_lanes, BufferType::kDefault);
  Array<Expr> args = {GetAccessPtr(dout, "rw"),    GetAccessPtr(dinp, "r"),     GetAccessPtr(dwgt, "r"),
                      out_shape[1] * out_shape[2], inp_shape[1] * inp_shape[3], out_shape[0] * out_shape[3]};
  args.push_back(Expr(1));
  auto then_stmt = Evaluate::make(Call::make(dout->dtype, intrin_name, args, Call::Extern));
  args.Set(args.size() - 1, Expr(0));
  auto else_stmt = Evaluate::make(Call::make(dout->dtype, intrin_name, args, Call::Extern));

  CommentManager::GetInstance().AddComment("Insn_name", intrin_name);
  CommentManager::GetInstance().AddComment("Insn_type", "cube");

  if (init) {
    return IfThenElse::make(condition, then_stmt, else_stmt);
  } else {
    return else_stmt;
  }
}

/// Transform float32 to float16 in binary value.
/// \param fp32
/// \return
uint32_t Fp32ToFp16InBin(float fp32) {
  // Here not check type limitation, meanning the transformation
  // should be valid.
  uint32_t *bitv = reinterpret_cast<uint32_t *>(&fp32);
  if (!(*bitv & 0xFFFFFFFF)) {
    return 0x0;
  }
  uint32_t fraction = (*bitv >> 13) & 0x3FF;
  uint32_t exponent = (*bitv >> 23) & 0xFF;
  exponent = (exponent - 127 + 15) & 0x1F;
  uint32_t sign = (fp32 > 0) ? 0 : 1;
  return (sign << 15) | (exponent << 10) | fraction;
}

/// Calculate the binary value for padding.
/// \param pad_value
/// \return
uint32_t CalPadValueInBinary(Expr pad_value) {
  uint32_t bin_value = 0x0;
  // Padding only support fp16, int8, uint8.
  if (auto fpimm = pad_value.as<FloatImm>()) {
    if (fpimm->type.bits() == 16) {
      bin_value = Fp32ToFp16InBin(fpimm->value);
    } else {
      CHECK(false);
    }
  } else if (auto intimm = pad_value.as<IntImm>()) {
    if (intimm->type.bits() == 8) {
      auto value = static_cast<uint32_t>(intimm->value);
      uint32_t valid = (value & 0x7F) | (value > 0 ? 0 : 0x80);
      // For int8 padding data, high 8bit and low 8bit should be the same.
      bin_value = (valid << 8) | valid;
    } else {
      CHECK(false);
    }
  } else if (auto uintimm = pad_value.as<UIntImm>()) {
    if (uintimm->type.bits() == 8) {
      bin_value = (uintimm->value & 0xFF);
    } else {
      CHECK(false);
    }
  }
  return bin_value;
}

/// Function to emit Img2col intrin cbuf_to_ub
/// \param op
/// \param attrs
/// \param src
/// \return
Stmt Im2ColEmitterL1UB(const Stmt &op, const std::unordered_map<std::string, ObjectRef> &attrs, const Buffer &src,
                       bool is_dynamic) {
  CHECK(op);
  Array<Buffer> dst;
  Array<Buffer> src_list;
  GetBufferIdFromStmt(op, dst, src_list);

  // reg_fmatrix

  // reg_xt
  Expr sw = Downcast<Expr>(attrs.at("stride_w"));
  Expr sh = Downcast<Expr>(attrs.at("stride_h"));
  Expr kw = Downcast<Expr>(attrs.at("filter_w"));
  Expr kh = Downcast<Expr>(attrs.at("filter_h"));
  Expr dilation_w = Downcast<Expr>(attrs.at("dilation_w"));
  Expr dilation_h = Downcast<Expr>(attrs.at("dilation_h"));
  Expr dst_jmp_offset = Downcast<Expr>(attrs.at("jump_offset"));
  Expr en_repeat = Downcast<Expr>(attrs.at("repeat_mode"));

  const int left_pad = GetInt32Const(Downcast<Expr>(attrs.at("pad_left")));
  const int right_pad = GetInt32Const(Downcast<Expr>(attrs.at("pad_right")));
  Expr config;
  if (!is_dynamic) {
    const int wi = GetInt32Const(Downcast<Expr>(attrs.at("w")));
    const int hi = GetInt32Const(Downcast<Expr>(attrs.at("h")));
    const int top_pad = GetInt32Const(Downcast<Expr>(attrs.at("pad_top")));
    const int bottom_pad = GetInt32Const(Downcast<Expr>(attrs.at("pad_bottom")));
    config = make_const(
      UInt(64), static_cast<uint64_t>((uint32_t)wi) | static_cast<uint64_t>((uint32_t)hi) << 16u |
                  static_cast<uint64_t>((uint32_t)left_pad) << 32u | static_cast<uint64_t>((uint32_t)right_pad) << 40u |
                  static_cast<uint64_t>((uint32_t)top_pad) << 48u | static_cast<uint64_t>((uint32_t)bottom_pad) << 56u);
  } else {
    const Expr wi = Downcast<Expr>(attrs.at("w"));
    const Expr hi = Downcast<Expr>(attrs.at("h"));
    const Expr top_pad = Downcast<Expr>(attrs.at("pad_top"));
    const Expr bottom_pad = Downcast<Expr>(attrs.at("pad_bottom"));
    const auto dim_w = Cast::make(UInt(64), wi);
    const auto dim_h = Cast::make(UInt(64), hi) << 16u;
    const auto padding_left = Cast::make(UInt(64), left_pad) << 32u;
    const auto padding_right = Cast::make(UInt(64), right_pad) << 40u;
    const auto padding_top = Cast::make(UInt(64), top_pad) << 48u;
    const auto padding_bottom = Cast::make(UInt(64), bottom_pad) << 56u;
    config = Simplify_cce(dim_w | dim_h | padding_left | padding_right | padding_top | padding_bottom);
  }

  // reg_xm
  Expr dst_offset = Expr(0);
  Expr pos_wk = Downcast<Expr>(attrs.at("pos_w"));
  Expr pos_hk = Downcast<Expr>(attrs.at("pos_h"));
  Expr first_wi = Downcast<Expr>(attrs.at("firstWi"));
  Expr first_hi = Downcast<Expr>(attrs.at("firstHi"));
  Expr idx_c = make_const(Int(32), 0);
  Expr n_repeat = Downcast<Expr>(attrs.at("repeat_time"));
  Expr csize = make_const(Int(32), 0);

  Expr first_arg;
  if (!is_dynamic) {
    first_arg = GetAccessPtr(dst[0], "w", dst_offset);
  } else {
    std::unordered_map<const Variable *, Expr> var_map;
    PostOrderVisit(op, [&var_map](const NodeRef &node) {
      if (auto for_node = node.as<For>()) {
        if (Equal(for_node->min, Expr(0))) {
          var_map[for_node->loop_var.get()] = for_node->extent;
        }
      }
    });
    first_arg = Substitute(GetAccessPtr(dst[0], "w", dst_offset), var_map);
  }

  Array<Expr> args = {first_arg,  GetAccessPtr(src, "r", Expr(0)),
                      pos_wk,     pos_hk,
                      first_wi,   first_hi,
                      idx_c,      sw,
                      sh,         kw,
                      kh,         dilation_w,
                      dilation_h, dst_jmp_offset,
                      en_repeat,  n_repeat,
                      csize};
  Stmt fmatrix = Evaluate::make(Call::make(dst[0]->dtype, "set_fmatrix", {config}, Call::Extern));
  Stmt res;

  if (dst[0]->scope == SCOPE_UBUF) {
    uint32_t pad_value_t = is_dynamic ? 0xFBFF : CalPadValueInBinary(Downcast<Expr>(attrs.at("pad_value")));
    Expr pad16_value = Cast::make(UInt(64), static_cast<int32_t>(pad_value_t));
    Stmt padding = Evaluate::make(Call::make(dst[0]->dtype, "set_padding", {pad16_value}, Call::Extern));
    auto im2col = Evaluate::make(Call::make(dst[0]->dtype, "img2col_cbuf_to_ub", args, Call::Extern));
    std::vector<Stmt> calls{padding, fmatrix, im2col};
    res = Block::make(calls);
  }

  CommentManager::GetInstance().AddComment("Insn_name", "img2col_cbuf_to_ub");
  CommentManager::GetInstance().AddComment("Insn_type", "dma");

  return res;
}

/// Function to emit Img2col intrin
/// \param op
/// \param attrs
/// \param src
/// \return
Stmt Im2ColEmitter(const Stmt &op, const std::unordered_map<std::string, ObjectRef> &attrs, const Buffer &src,
                   bool is_dynamic) {
  CHECK(op);
  Array<Buffer> dst;
  Array<Buffer> src_list;
  GetBufferIdFromStmt(op, dst, src_list);

  // reg_xt
  Expr sw = Downcast<Expr>(attrs.at("stride_w"));
  Expr sh = Downcast<Expr>(attrs.at("stride_h"));
  Expr kw = Downcast<Expr>(attrs.at("filter_w"));
  Expr kh = Downcast<Expr>(attrs.at("filter_h"));
  Expr dilation_w = Downcast<Expr>(attrs.at("dilation_w"));
  Expr dilation_h = Downcast<Expr>(attrs.at("dilation_h"));
  Expr dst_jmp_offset = Downcast<Expr>(attrs.at("jump_offset"));
  Expr en_repeat = Downcast<Expr>(attrs.at("repeat_mode"));
  Expr n_repeat = Downcast<Expr>(attrs.at("repeat_time"));

  VarExpr idx3d_loop("idx_3d_loop");

  // reg_xm
  Expr m_idx = Downcast<Expr>(attrs.at("idx_m"));
  Expr k_idx = Downcast<Expr>(attrs.at("idx_k"));
  Expr win_h = Downcast<Expr>(attrs.at("win_h"));
  Expr win_w = Downcast<Expr>(attrs.at("win_w"));

  Expr dst_offset;
  if (is_zero(en_repeat)) {
    m_idx += idx3d_loop * 16;
    dst_offset = n_repeat * idx3d_loop * 16 * 16;
  } else {
    k_idx += idx3d_loop * 16;
    dst_offset = idx3d_loop * 16 * 16;
  }

  Expr idx_w = Downcast<Expr>(attrs.at("idx_w"));
  Expr idx_h = Downcast<Expr>(attrs.at("idx_h"));
  Expr kw_l0 = Downcast<Expr>(attrs.at("kw_l0"));
  Expr kh_l0 = Downcast<Expr>(attrs.at("kh_l0"));
  Expr win_in_h =
    Cast::make(UInt(64), Simplify(truncdiv(truncdiv(truncmod(k_idx, (kh_l0 * kw_l0 * 16)), 16), kw_l0) + idx_h));
  Expr win_in_w =
    Cast::make(UInt(64), Simplify(truncmod(truncdiv(truncmod(k_idx, (kh_l0 * kw_l0 * 16)), 16), kw_l0) + idx_w));

  Expr idx_c = Simplify(truncdiv(k_idx, (kh_l0 * kw_l0 * 16)));
  Expr csize = make_const(Int(32), 0);

  // reg_fmatrix
  Expr config;
  Expr win_out_h;
  Expr win_out_w;
  if (!is_dynamic) {
    const int wi = GetInt32Const(Downcast<Expr>(attrs.at("w")));
    const int hi = GetInt32Const(Downcast<Expr>(attrs.at("h")));
    const int left_pad = GetInt32Const(Downcast<Expr>(attrs.at("pad_left")));
    const int right_pad = GetInt32Const(Downcast<Expr>(attrs.at("pad_right")));
    const int top_pad = GetInt32Const(Downcast<Expr>(attrs.at("pad_top")));
    const int bottom_pad = GetInt32Const(Downcast<Expr>(attrs.at("pad_bottom")));
    win_out_h = Cast::make(Int(16), Simplify(truncdiv(truncmod(m_idx, (win_h * win_w)), win_w) * sh - top_pad));
    win_out_w =
      Cast::make(Int(16), Simplify(Simplify_cce(truncmod(truncmod(m_idx, (win_h * win_w)), win_w)) * sw - left_pad));
    if (wi < 0 || hi < 0 || left_pad < 0 || right_pad < 0 || top_pad < 0 || bottom_pad < 0) {
      CHECK(false) << "wrong imm";
    }
    const auto dim_w = static_cast<uint64_t>(wi);
    const auto dim_h = static_cast<uint64_t>(hi) << 16u;
    const auto padding_left = static_cast<uint64_t>(left_pad) << 32u;
    const auto padding_right = static_cast<uint64_t>(right_pad) << 40u;
    const auto padding_top = static_cast<uint64_t>(top_pad) << 48u;
    const auto padding_bottom = static_cast<uint64_t>(bottom_pad) << 56u;
    config = make_const(UInt(64), dim_w | dim_h | padding_left | padding_right | padding_top | padding_bottom);
  } else {
    const Expr wi = Downcast<Expr>(attrs.at("w"));
    const Expr hi = Downcast<Expr>(attrs.at("h"));
    const Expr left_pad = Downcast<Expr>(attrs.at("pad_left"));
    const Expr right_pad = Downcast<Expr>(attrs.at("pad_right"));
    const Expr top_pad = Downcast<Expr>(attrs.at("pad_top"));
    const Expr bottom_pad = Downcast<Expr>(attrs.at("pad_bottom"));
    win_out_h = Cast::make(Int(16), Simplify(truncdiv(truncmod(m_idx, (win_h * win_w)), win_w) * sh - top_pad));
    win_out_w =
      Cast::make(Int(16), Simplify(Simplify_cce(truncmod(truncmod(m_idx, (win_h * win_w)), win_w)) * sw - left_pad));

    const auto dim_w = Cast::make(UInt(64), wi);
    const auto dim_h = Cast::make(UInt(64), hi) << 16u;
    const auto padding_left = Cast::make(UInt(64), left_pad) << 32u;
    const auto padding_right = Cast::make(UInt(64), right_pad) << 40u;
    const auto padding_top = Cast::make(UInt(64), top_pad) << 48u;
    const auto padding_bottom = Cast::make(UInt(64), bottom_pad) << 56u;
    config = Simplify_cce(dim_w | dim_h | padding_left | padding_right | padding_top | padding_bottom);
  }

  Array<Expr> args = {GetAccessPtr(dst[0], "w", dst_offset),
                      GetAccessPtr(src, "r", Expr(0)),
                      win_in_w,
                      win_in_h,
                      win_out_w,
                      win_out_h,
                      idx_c,
                      sw,
                      sh,
                      kw,
                      kh,
                      dilation_w,
                      dilation_h,
                      dst_jmp_offset,
                      en_repeat,
                      n_repeat,
                      csize};
  Stmt call1 = Evaluate::make(Call::make(dst[0]->dtype, "set_fmatrix", {config}, Call::Extern));
  Stmt call2;
  std::string intrin_name;
  if (dst[0]->scope == SCOPE_CA) {
    intrin_name = "img2col_cbuf_to_ca";
  } else {
    CHECK(dst[0]->scope == SCOPE_CB);
    intrin_name = "img2col_cbuf_to_cb";
  }
  call2 = Evaluate::make(Call::make(dst[0]->dtype, intrin_name, args, Call::Extern));
  Stmt body = Block::make(call1, call2);

  CommentManager::GetInstance().AddComment("Insn_name", intrin_name);
  CommentManager::GetInstance().AddComment("Insn_type", "dma");

  return For::make(idx3d_loop, Expr(0), dst_jmp_offset, ForType::Serial, ktvm::ir::DeviceAPI::None, body);
}

/// Function to emit argmax/argmin intrin
/// \param op
/// \param intrin_name
/// \return
Stmt BinaryArgOpEmitter(const Stmt &op, const std::string &intrin_name) {
  CHECK(op);
  StmtInfoList dst_info_list;
  StmtInfoList src_info_list;
  StmtInfo if_info;
  StmtInfo for_info;
  ArgInfo arg_info;

  static_cast<void>(GetBinaryVecInsnArgs(op, intrin_name, dst_info_list, src_info_list, if_info, for_info));
  auto dst_info = dst_info_list[0];
  auto src_info = src_info_list[1];

  if (src_info_list[0]->var_.size() > src_info_list[1]->var_.size()) {
    src_info = src_info_list[0];
  }
  ReduceLastAxisPatternGenerator generator = ReduceLastAxisPatternGenerator(dst_info, src_info, for_info, intrin_name);
  auto result = generator.GetInsnArgs();
  arg_info = result.arg_info;
  dst_info = result.dst_info_list[0];
  src_info = result.src_info_list[0];
  for_info = result.for_info;

  CHECK(intrin_name == "argmax" || intrin_name == "argmin") << "Invalid argop type " << intrin_name;
  const auto f16_max = Expr(0x7bff);
  const auto f16_min = Expr(0xfbff);
  Expr init = intrin_name == "argmax" ? f16_min : f16_max;

  CommentManager::GetInstance().AddComment("Insn_name", intrin_name);
  CommentManager::GetInstance().AddComment("Insn_type", "single_vector");
  return EmitCceArgmaxIntrinHub(if_info, for_info, arg_info, dst_info, src_info, intrin_name, init);
}

/// Function to emit argmax intrin
/// \param op
/// \return
Stmt BinaryArgmaxEmitter(const Stmt &op) { return BinaryArgOpEmitter(op, "argmax"); }

/// Function to emit argmin intrin
/// \param op
/// \return
Stmt BinaryArgminEmitter(const Stmt &op) { return BinaryArgOpEmitter(op, "argmin"); }

/// Function to emit vaxpy intrin
/// \param op - The input stmt to be emitted as intrin
/// \return Stmt of emitted CCE intrin
Stmt VaxpyEmitter(const Stmt &op) {
  CHECK(op);
  Stmt stmt = BinaryVecEmitter(op, "vaxpy");
  Array<Expr> scale_factor;
  // get scale factor
  PostOrderVisit(op, [&scale_factor](const NodeRef &n) {
    if (n.as<Call>() && n.as<Call>()->name == "vaxpy") {
      CHECK(n.as<Call>()->args.size() >= 3);
      scale_factor.push_back(n.as<Call>()->args[2]);
    }
  });
  // because vaxpy only needs src0, so we remove extra src1 from argument list
  PackedFunc ReplaceIns = PackedFunc([&scale_factor](TVMArgs args, TVMRetValue *ret) {
    Expr n = args[0];
    if (n->IsInstance<Call>()) {
      auto call = n.as<Call>();
      if (call->name == "vaxpy") {
        auto args_vaxpy = call->args;
        Array<Expr> new_args = {args_vaxpy[0],    // dst
                                args_vaxpy[1],    // src0
                                scale_factor[0],  // src1
                                args_vaxpy[3],    // repeat
                                args_vaxpy[4],    // dst_sm0
                                args_vaxpy[5],    // src0_sm0
                                args_vaxpy[7],    // dst_sm1
                                args_vaxpy[8]};   // src0_sm1
        *ret = Call::make(call->type, call->name, new_args, call->call_type, call->func, call->value_index);
      }
    }
  });

  CommentManager::GetInstance().AddComment("Insn_name", "vaxpy");
  CommentManager::GetInstance().AddComment("Insn_type", "single_vector");

  return ktvm::ir::IRTransform(stmt, ktvm::runtime::PackedFunc{nullptr}, ReplaceIns, {StringImm::make("Call")});
}

/// Function to emit vnchwconv intrin
/// \param op - The input stmt to be emitted as intrin
/// \return Stmt of emitted CCE intrin
Stmt VnchwconvEmitter(const Stmt &op) {
  CHECK(op);
  StmtInfoList dst_info_list;
  StmtInfoList src_info_list;
  StmtInfo for_info;
  StmtInfo if_info;
  GetCompactComputationInfo(op, dst_info_list, src_info_list, if_info, for_info, true, false);

  auto dst_info = dst_info_list[0];
  auto src_info = src_info_list[0];

  CHECK(for_info.ops_.size() >= 2) << "There should be at least 2 for loops but has " << for_info.ops_.size()
                                   << " loops.";

  auto last_dim_var = GetItem(dst_info->var_, -1);
  size_t target_idx = 0;
  bool suc = GetIndexOfElement(src_info->var_, last_dim_var, target_idx);
  CHECK(suc);

  Expr c1_loop_extent = 1;
  for (size_t i = 0; i < target_idx; ++i) {
    CHECK(i < for_info.ops_.size() && for_info.ops_[i].as<For>());
    c1_loop_extent *= for_info.ops_[i].as<For>()->extent;
  }

  Stmt result;
  Expr hx_wextent = 1;
  CHECK_GE(for_info.ops_.size(), target_idx + 1);
  for (int i = 0; i < static_cast<int>(for_info.ops_.size() - target_idx) - 1; ++i) {
    // last dim is c0, so start from -2
    CHECK(GetItem(for_info.ops_, -2 - i).as<For>());
    hx_wextent *= GetItem(for_info.ops_, -2 - i).as<For>()->extent;
  }

  int c0 = 16;
  int c1 = GetInt32Const(c1_loop_extent);
  int block_size = GetUbBlkSize(dst_info->dtype_);

  Expr repeat = floordiv(hx_wextent, c0);
  Expr dst_stride = GetIntConst(repeat) > 1 ? c0 : Expr(0);
  Expr src_stride = GetIntConst(repeat) > 1 ? Expr(1) : Expr(0);

  Type array_type = UInt(64);
  Expr buffer_size = Expr(8);
  Buffer dst_buffer_id = GenBufferId(dst_info);
  Buffer src_buffer_id = GenBufferId(src_info);
  VarExpr c_idx = VarExpr("c_idx");
  VarExpr buffer_idx = VarExpr("buffer_idx");

  Expr dst = GetAccessPtr(dst_buffer_id, "r", buffer_idx * block_size + c_idx * hx_wextent * block_size);
  Expr src = GetAccessPtr(src_buffer_id, "r", buffer_idx * hx_wextent + c_idx * hx_wextent * block_size);
  Expr dst1 =
    GetAccessPtr(dst_buffer_id, "r", buffer_idx * block_size + c_idx * hx_wextent * block_size + block_size * 8);
  Expr src1 =
    GetAccessPtr(src_buffer_id, "r", buffer_idx * hx_wextent + c_idx * hx_wextent * block_size + hx_wextent * 8);

  const size_t va_reg_num = 4;
  Array<Buffer> addr_buffer_list;
  Array<Expr> index_list = {src, src1, dst, dst1};
  Stmt assign_value_stmt;
  // create array and assign address
  for (size_t i = 0; i < va_reg_num; ++i) {
    std::string var_name = "va" + std::to_string(i) + "AddrArray";
    std::string buf_name = "address_array" + std::to_string(i);
    Buffer addr_buffer = BufferNode::make(Var(var_name, Handle()), array_type, {buffer_size}, {}, Expr(), buf_name,
                                          SCOPE_REG, 0, 0, BufferType::kDefault);
    addr_buffer_list.push_back(addr_buffer);
    Expr addr = Load::make(array_type, addr_buffer->data, Expr(buffer_idx), const_true());
    assign_value_stmt = InsertBody(
      assign_value_stmt,
      Evaluate::make(Call::make(UInt(64), "printer_cast",
                                {Call::make(UInt(64), "reg", {addr}, Call::Extern), index_list[i]}, Call::Extern)));
  }

  result = For::make(buffer_idx, Expr(0), buffer_size, ForType::Serial, ktvm::ir::DeviceAPI::None, assign_value_stmt);

  Array<Expr> args;
  for (size_t i = 0; i < addr_buffer_list.size(); ++i) {
    // set va register and calculate
    args = {StringImm::make("VA" + std::to_string(i)), GetAccessPtr(addr_buffer_list[i], "r", Expr(0))};
    result = EmitCceIntrinTemplate(result, dst_info->dtype_, args, "set_va_reg_sb");
  }

  // call scatter intrin
  args = {StringImm::make("VA2"), StringImm::make("VA0"), repeat, dst_stride, src_stride};
  result = EmitCceIntrinTemplate(result, dst_info->dtype_, args, "scatter_vnchwconv_b16");

  // allocate storage scope
  for (int i = static_cast<int>(addr_buffer_list.size()) - 1; i >= 0; --i) {
    result = AttrStmt::make(
      addr_buffer_list[i]->data, STORAGE_SCOPE, Expr(SCOPE_REG),
      Allocate::make(addr_buffer_list[i]->data, addr_buffer_list[i]->dtype, {buffer_size}, const_true(), result));
  }
  result = For::make(c_idx, Expr(0), Expr(c1), ForType::Serial, ktvm::ir::DeviceAPI::None, result);

  CommentManager::GetInstance().AddComment("Insn_name", "vnchwconv");
  CommentManager::GetInstance().AddComment("Insn_type", "scatter");

  return result;
}

/// Function to emit reduce sum block
/// \param attr_stmt
/// \param enable_bisect
/// \param count
/// \return Stmt of emitted CCE intrin
Stmt InsnFromVbaddAttr(const AttrStmt *attr_stmt, bool enable_bisect, int count) {
  auto reduce = attr_stmt->body.as<For>();
  if (reduce) {
    return BinaryVecEmitter(GetRef<Stmt>(attr_stmt), "vadd", enable_bisect, count);
  } else {
    return Stmt();
  }
}

/// Function to combine two indepent reduce block
/// \param a
/// \param b
/// \return Stmt of emitted CCE intrin
Stmt ReduceCombine(Stmt a, Stmt b) {
  Stmt res;
  auto a_attr = a.as<AttrStmt>();
  auto b_attr = b.as<AttrStmt>();
  if (a_attr && b_attr) {
    auto a_alloc = a_attr->body.as<Allocate>();
    auto b_alloc = b_attr->body.as<Allocate>();
    if (a_alloc && b_alloc) {
      auto a_block = a_alloc->body.as<Block>();
      auto b_block = b_alloc->body.as<Block>();
      CHECK(a_block);
      CHECK(b_block);
      if (Equal(a_block->first, b_block->first)) {
        res = InsertBody(res, a_block->first, true);
      } else {
        res = InsertBody(res, Block::make(a_block->first, b_block->first), true);
      }
      while (a_block->rest.as<Block>() && b_block->rest.as<Block>()) {
        a_block = a_block->rest.as<Block>();
        b_block = b_block->rest.as<Block>();
        if (Equal(a_block->first, b_block->first)) {
          res = InsertBody(res, a_block->first, true);
        } else {
          res = InsertBody(res, Block::make(a_block->first, b_block->first), true);
        }
      }
      if (Equal(a_block->rest, b_block->rest)) {
        res = InsertBody(res, a_block->rest, true);
      } else {
        res = InsertBody(res, Block::make(a_block->rest, b_block->rest), true);
      }
      res = Allocate::make(b_alloc->buffer_var, b_alloc->type, b_alloc->extents, b_alloc->condition, res,
                           b_alloc->new_expr, b_alloc->free_function);
      res = AttrStmt::make(b_attr->node, b_attr->attr_key, b_attr->value, res);
      res = Allocate::make(a_alloc->buffer_var, a_alloc->type, a_alloc->extents, a_alloc->condition, res,
                           a_alloc->new_expr, a_alloc->free_function);
      res = AttrStmt::make(a_attr->node, a_attr->attr_key, a_attr->value, res);
    }
  }
  return res;
}

/// Function to emit combined reduce
/// \param op
/// \param enable_bisect
/// \return Stmt of emitted CCE intrin
Stmt ReduceCombineEmitter(const Stmt &op, bool enable_bisect) {
  auto block_it = op.as<Block>();
  CHECK(block_it);
  auto first_rd = block_it->first.as<AttrStmt>();
  CHECK(first_rd);
  int count = 0;
  Stmt result = InsnFromVbaddAttr(first_rd, enable_bisect, count);
  count++;
  Stmt res_it;
  while (block_it->rest.as<Block>()) {
    block_it = block_it->rest.as<Block>();
    res_it = InsnFromVbaddAttr(block_it->first.as<AttrStmt>(), enable_bisect, count);
    count++;
    result = ReduceCombine(result, res_it);
  }
  res_it = InsnFromVbaddAttr(block_it->rest.as<AttrStmt>(), enable_bisect, count);
  result = ReduceCombine(result, res_it);
  return result;
}

/// Call diffsrent emitter with given insn_name
/// \param insn_name
/// \param op
/// \param enable_bisect - Enable bisection optimization
/// \param enable_cover_protect - Enable cover protection optimization
/// \return
Stmt InsnEmit(std::string insn_name, const Stmt &op, bool enable_bisect, bool enable_cover_protect, int comment_level) {
  CHECK(op.defined());

  static const std::map<std::string, std::string> ReplaceAttrPragmaMap = {
    // vector binary
    {"binary_vcadd", "vec_binary_add"},
    {"vaxpy", "vec_binary_axpy"},
    // vector single
    {"vec_single_fabs", "vec_single_abs"},
    {"broadcast", "vec_broadcast"},
    // cube
    {"mad", "cube_mad"},
    {"ub2gm", "cube_ub2gm"},
    {"im2col", "cube_img2col"},
    // special attrs
    {"vec_binary_proposal_sort", "vec_proposal_sort"},
    {"vec_binary_topk_sort", "vec_topk_sort"},
    {"vec_binary_dropout", "vec_dropout"},
    {"vec_binary_fargmax", "vec_argmax"},
    {"vec_binary_fargmin", "vec_argmin"},
    {"vec_binary_iou", "vec_iou"},
    {"vec_binary_nms", "vec_nms"},
    {"mask_broadcast", "vec_broadcast"},
  };

  static const std::map<std::string, std::string> BinaryVecInsnMap = {
    // vadd.f16 support target:mini_v100 tiny_v100 lite_v100 cloud_v100
    // vadd.s32 support target:mini_v100 tiny_v100 lite_v100 cloud_v100
    // vadd.f32 support target:mini_v100 cloud_v100
    // vadd contains two situations:
    // 1. normal elewise vector add
    // - all src[i].shape = dst.shape
    // 2. reductive vector add
    // - exist src[i].shape != dst.shape
    {"vec_binary_add", "vadd"},
    // vsub.f16 support target:mini_v100 tiny_v100 lite_v100 cloud_v100
    // vsub.s32 support target:mini_v100 tiny_v100 lite_v100 cloud_v100
    // vsub.f32 support target:mini_v100 cloud_v100
    {"vec_binary_sub", "vsub"},
    // vmul.f16 support target:mini_v100 tiny_v100 lite_v100 cloud_v100
    // vmul.s32 support target:mini_v100 tiny_v100 lite_v100 cloud_v100
    // vmul.f32 support target:mini_v100 cloud_v100
    {"vec_binary_mul", "vmul"},
    // vmin.f16 support target:mini_v100 tiny_v100 lite_v100 cloud_v100
    // vmin.s32 support target:mini_v100 tiny_v100 lite_v100 cloud_v100
    // vmin.f32 support target:mini_v100 cloud_v100
    {"vec_binary_min", "vmin"},
    // vmax.f16 support target:mini_v100 tiny_v100 lite_v100 cloud_v100
    // vmax.s32 support target:mini_v100 tiny_v100 lite_v100 cloud_v100
    // vmax.f32 support target:mini_v100 cloud_v100
    {"vec_binary_max", "vmax"},
    {"vec_binary_div", "vdiv"},
    {"vec_binary_and", "vand"},
    {"vec_binary_bitwise_and", "vand"},
    {"vec_binary_or", "vor"},
    {"vec_binary_bitwise_or", "vor"},
    {"vec_binary_vmadd", "vmadd"},
    {"vec_binary_vmaddrelu", "vmaddrelu"},
    {"vec_binary_vmla", "vmla"}};

  static const std::map<std::string, std::string> SingleVecInsnMap = {
    // vmuls.f16 support target:mini_v100 tiny_v100 lite_v100 cloud_v100
    // vmuls.f32 supporttarget:mini_v100 cloud_v100
    {"vec_single_muls", "vmuls"},
    // vadds.f16 support target:mini_v100 tiny_v100 lite_v100 cloud_v100
    // vadds.f32 support target:mini_v100 cloud_v100
    {"vec_single_adds", "vadds"},
    // vrelu.f16 support target:mini_v100 tiny_v100 lite_v100 cloud_v100
    {"vec_single_relu", "vrelu"},
    // vabs.f16 support target:mini_v100 tiny_v100 lite_v100 cloud_v100
    // vabs.f32 support target:mini_v100 cloud_v100
    {"vec_single_abs", "vabs"},
    // vln.f16 support target:mini_v100 tiny_v100 lite_v100 cloud_v100
    // vln.f32 support target:cloud_v100
    {"vec_single_log", "vln"},
    // vexp.f16 support target:mini_v100 tiny_v100 lite_v100 cloud_v100
    // vexp.f32 support target:cloud_v100
    {"vec_single_exp", "vexp"},
    // vrec.f16 support target:mini_v100 tiny_v100 lite_v100 cloud_v100
    // vrec.f32 support target:mini_v100 cloud_v100
    {"vec_single_rec", "vrec"},
    // vnot support target:mini_v100 tiny_v100 lite_v100 cloud_v100
    {"vec_single_not", "vnot"},
    {"vec_single_bitwise_not", "vnot"},
    // vsqrt support target:cloud_v100
    {"vec_single_sqrt", "vsqrt"},
    {"vec_single_rsqrt", "vrsqrt"},
    {"vec_broadcast", "vector_dup"}};

  static const std::map<std::string, std::string> SingleCastInsnMap = {
    {"vec_single_floor", "f"}, {"vec_single_round", "r"}, {"vec_single_ceil", "c"}, {"vec_single_trunc", "z"}};

  static const std::set<std::string> ReturnOpInsnSet = {"scalar_dma", "scatter", "vec_binary_select_loop_var"};

  static const std::map<std::string, std::function<Stmt(const Stmt &)>> InsnFunctorMap = {
    {"dma_atomic_add", DmaAtomicAddEmitter},
    {"vec_single_cast", SingleCastEmitter},
    {"vec_argmax_cast", SingleFargmaxCastEmitter},
    {"vec_proposal_sort", BinaryProposalSortEmitter},
    {"vec_topk_sort", BinaryTopkSortEmitter},
    {"vec_iou", BinaryIouEmitter},
    {"vec_nms", BinaryNmsEmitter},
    {"vec_argmax", BinaryArgmaxEmitter},
    {"vec_argmin", BinaryArgminEmitter},
    {"vec_dropout", BinaryDropoutEmitter},
    {"cube_mad", MadEmitter},
    {"vec_select_scalar", SelectWithScalarEmitter},
    {"vec_binary_axpy", VaxpyEmitter},
    {"opt_broadcast", MultiMaskEmitter},
    {"vec_single_four2five_nchw", VnchwconvEmitter}};

  if (ReplaceAttrPragmaMap.count(insn_name) != 0) {
    insn_name = ReplaceAttrPragmaMap.find(insn_name)->second;
  }

  Stmt result;
  CommentManager::GetInstance().CleanComments();
  // Get alignment
  auto stores = GetStores(op);
  CHECK(!stores.empty() && stores[0].as<Store>());
  auto predicate = GetInt32Const(stores[0].as<Store>()->predicate);
  CommentManager::GetInstance().AddComment("Alignment", std::to_string(predicate));
  CommentManager::GetInstance().AddComment("Pragma", insn_name);

  if (insn_name == "dma_copy") {
    result = DmaMovEmitter(op, enable_cover_protect);
  } else if (InsnFunctorMap.count(insn_name) != 0) {
    result = InsnFunctorMap.find(insn_name)->second(op);
  } else if (BinaryVecInsnMap.count(insn_name) != 0) {
    result = BinaryVecEmitter(op, BinaryVecInsnMap.find(insn_name)->second, enable_bisect);
  } else if (SingleVecInsnMap.count(insn_name) != 0) {
    result = SingleVecEmitter(op, SingleVecInsnMap.find(insn_name)->second);
  } else if (SingleCastInsnMap.count(insn_name) != 0) {
    result = SingleVconvEmitter(op, SingleCastInsnMap.find(insn_name)->second);
  } else if (ReturnOpInsnSet.count(insn_name) != 0) {
    result = ReturnOpEmitter(op);
  } else if (insn_name == "reduce_reorder") {
    result = ReduceCombineEmitter(op, enable_bisect);
  } else {
    LOG(FATAL) << "No such intrinsic rule: " << insn_name;
  }

  CHECK(result.defined()) << "result stmt is undefined!";

  std::string comment = CommentManager::GetInstance().GenComment(comment_level);
  if (!comment.empty()) {
    result = AttrStmt::make(make_zero(Int(32)), "pragma_insn_comment", Expr(comment), result);
  }

  return result;
}
}  // namespace ir
}  // namespace akg
