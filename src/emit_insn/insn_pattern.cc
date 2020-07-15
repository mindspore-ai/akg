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

#include "insn_pattern.h"

#include <tvm/runtime/packed_func.h>
#include <tvm/base.h>
#include <tvm/ir_pass.h>
#include <tvm/api_registry.h>

#include <bitset>
#include <set>

#include "common/array_api.h"

namespace akg {
using air::runtime::TVMArgs;
using air::runtime::TVMRetValue;

/// Check is scalar ir
/// \param info_list
/// \return
bool IsScalarMode(const StmtInfoList &info_list) {
  auto IsScalar_judge = [](const StmtStoreInfo &info) {
    return info->strides_.empty() || GetInt32Const(GetItem(info->strides_, -1)) == 1;
  };
  return !std::any_of(info_list.begin(), info_list.end(), IsScalar_judge);
}

/// SplitAxis when data alignment not illigal
/// \param InfoList
/// \param for_info
/// \param which axis to split
/// \param inner size of loop split
/// \return
void SplitAxis(StmtInfoList &com_info_list, StmtInfo &for_info, const Var &axis_var, int new_size) {
  // check axis_var in for_info
  size_t var_idx = 0;
  bool suc = GetIndexOfElement(for_info.vars_, VarExpr(axis_var), var_idx);
  CHECK(suc) << "Error: The split axis var is not in for_info.";

  // check axis shape and new shape is legal
  auto ori_for_op = for_info.ops_[var_idx].as<For>();
  CHECK(ori_for_op != nullptr);
  int axis_shape = GetInt32Const(ori_for_op->extent);
  CHECK_NE(new_size, 0);
  CHECK(axis_shape % new_size == 0) << "Error: New size is not equal to multiple of original axis size.";

  // split for loop
  auto ori_for_var = for_info.vars_[var_idx];

  VarExpr for_var_outer = VarExpr(ori_for_var->name_hint + "0");
  Expr for_extent_outer = Expr(axis_shape / new_size);
  VarExpr for_var_inner = VarExpr(ori_for_var->name_hint + "1");
  Expr for_extent_inner = Expr(new_size);

  auto for_op_inner = For::make(for_var_inner, ori_for_op->min, for_extent_inner, ori_for_op->for_type,
                                ori_for_op->device_api, ori_for_op->body);
  auto for_op_outer =
    For::make(for_var_outer, Expr(0), for_extent_outer, ori_for_op->for_type, ori_for_op->device_api, for_op_inner);

  for_info.RemoveItem(var_idx);

  Insert(for_info.ops_, var_idx, for_op_inner);
  Insert(for_info.ops_, var_idx, for_op_outer);
  Insert(for_info.vars_, var_idx, for_var_inner);
  Insert(for_info.vars_, var_idx, for_var_outer);

  // split com_info var and update index
  for (size_t i = 0; i < com_info_list.size(); ++i) {
    auto info = com_info_list[i];
    size_t cur_idx = 0;
    if (GetIndexOfElement(info->var_, axis_var, cur_idx)) {
      info.GetNode()->var_.Set(cur_idx, Var(for_var_inner));
      Insert(info.GetNode()->var_, cur_idx, Var(for_var_outer));
      info.GetNode()->shape_.Set(cur_idx, for_extent_inner);
      Insert(info.GetNode()->shape_, cur_idx, for_extent_outer);
      Insert(info.GetNode()->strides_, cur_idx, info->shape_[cur_idx + 1] * info->strides_[cur_idx]);
      // Update index
      auto new_index = info->index_;
      new_index = EliminateVarInExpr(new_index, {axis_var});
      CHECK_LT(cur_idx, info->var_.size() - 1);
      new_index = new_index + info->var_[cur_idx] * info->strides_[cur_idx] +
                  info->var_[cur_idx + 1] * info->strides_[cur_idx + 1];
      new_index = CanonicalSimplify(new_index);
      info.GetNode()->index_ = new_index;
      com_info_list.Set(i, info);
    }
  }
}

/// Get CCE Multiple Vector Elemwise mode instructions args.
/// \param dst_info_list
/// \param src_info_list
/// \param for_info
/// \param intrin_name
/// \return
ArgInfo GetMultiVecInsnArgs(StmtInfoList &dst_info_list, StmtInfoList &src_info_list, StmtInfo &for_info) {
  ArgInfo arg_info = ArgInfo(make_node<ArgInfoNode>());

  CHECK(!dst_info_list.empty());
  auto dst_info = dst_info_list[0];
  int block_size = GetUbBlkSize(dst_info->dtype_);

  StmtInfoList info_list = src_info_list;
  Insert(info_list, 0, dst_info);

  for (auto info : info_list) {
    if (info->shape_.empty()) {
      LOG(FATAL) << "CCE Vector Insn Error: dst_buffer and src_buffer can not be scalar, should keep len(shape) > 0.";
    }
  }

  CHECK_NE(block_size, 0);
  int last_dim_shape = (GetInt32Const(GetItem(dst_info->shape_, -1)) + block_size - 1) / block_size * block_size;

  auto dst_var = dst_info->var_;
  auto dst_shape = dst_info->shape_;
  auto dst_strides = dst_info->strides_;
  Array<Array<Var>> src_var_list;
  Array<Array<Expr>> src_shape_list;
  Array<Array<Expr>> src_stride_list;
  for (auto info : src_info_list) {
    src_var_list.push_back(info->var_);
    src_shape_list.push_back(info->shape_);
    src_stride_list.push_back(info->strides_);
  }

  const int vec_max_len = GetVecMaxLen(dst_info->dtype_);
  CHECK_NE(vec_max_len, 0);

  auto Get1DPattern = [last_dim_shape, vec_max_len, src_stride_list, dst_info, dst_var](
                        VectorArgInfo &body_args, VectorArgInfo &tail_args, Array<Var> &elim_var) {
    int body_len = last_dim_shape / vec_max_len * vec_max_len;
    int tail_len = last_dim_shape % vec_max_len;
    if (body_len > 0) {
      body_args = VectorArgInfo(make_node<VectorArgInfoNode>());
      body_args.GetNode()->body_num_ = 1;
      body_args.GetNode()->repeat_ = body_len / vec_max_len;
      body_args.GetNode()->dst_stride_m0_ = Expr(1);
      body_args.GetNode()->dst_stride_m1_ = Expr(FULL_BLOCK_NUM);
      for (auto stride : src_stride_list) {
        body_args.GetNode()->src_stride_m0_list_.push_back(Expr(1));
        body_args.GetNode()->src_stride_m1_list_.push_back(Expr(FULL_BLOCK_NUM));
      }
      body_args.GetNode()->vec_mask_ = GetVecMask(vec_max_len, 1, dst_info->dtype_);
    }
    if (tail_len > 0) {
      tail_args = VectorArgInfo(make_node<VectorArgInfoNode>());
      tail_args.GetNode()->dst_head_ = Expr(body_len);
      tail_args.GetNode()->repeat_ = Expr(1);
      tail_args.GetNode()->dst_stride_m0_ = Expr(1);
      tail_args.GetNode()->dst_stride_m1_ = Expr(0);
      for (auto stride : src_stride_list) {
        tail_args.GetNode()->src_head_list_.push_back(Expr(body_len));
        tail_args.GetNode()->src_stride_m0_list_.push_back(Expr(1));
        tail_args.GetNode()->src_stride_m1_list_.push_back(Expr(0));
      }
      tail_args.GetNode()->vec_mask_ = GetVecMask(tail_len, 1, dst_info->dtype_);
    }

    if (!dst_var.empty()) {
      elim_var = GetRange(dst_var, -1, 1);
    }
  };

  VectorArgInfo body_args = VectorArgInfo();
  VectorArgInfo tail_args = VectorArgInfo();
  Array<Var> elim_var = {};

  // vcmp + vsel only support 1d pattern
  Get1DPattern(body_args, tail_args, elim_var);
  arg_info.GetNode()->pattern_ = PATTERN_1D;
  arg_info.GetNode()->body_arg_info_ = body_args;
  arg_info.GetNode()->tail_arg_info_ = tail_args;
  dst_info.GetNode()->insn_offset_ = GetInsnOffset(dst_info, elim_var);
  dst_info_list.Set(0, dst_info);

  for (size_t i = 0; i < src_info_list.size(); ++i) {
    auto src_info = src_info_list[i];
    src_info.GetNode()->insn_offset_ = GetInsnOffset(src_info, elim_var);
    src_info_list.Set(i, src_info);
  }

  CleanForInfoVars(for_info, elim_var);

  return arg_info;
}

/// Get first non zero shape from input shapes
/// \param dst_shape
/// \param src0_shape
/// \param src1_shape
/// \return
int PatternGenerator::GetNonZeroShape(const Expr &dst_shape, const Expr &src0_shape, const Expr &src1_shape) {
  int shape = 0;
  for (int val :
       {GetInt32Const(dst_shape), GetInt32Const(src0_shape), src1_shape.defined() ? GetInt32Const(src1_shape) : 0}) {
    if (val == 0) {
      continue;
    }
    if (shape != 0 && val != shape) {
      LOG(FATAL) << "Error: same var has different shapes. " << GetIntConst(dst_shape) << " "
                 << GetIntConst(src0_shape);
    }
    shape = val;
  }
  CHECK(shape != 0) << "Error: all shapes are equal to 0.";
  return shape;
}

/// In case
/// for (cc3) {
///   A[(cc3*16)] = (B[(cc3*16)] - C[(cc3*16)])
/// }
/// and
/// for (cc3) {
///   A[(cc3*16)] = OP(B[(cc3*16)])
/// }
/// We can append a dummy axis so the ir can be emitted with more efficient pattern
/// \param dst_info_list
/// \param src_info_list
/// \param for_info
void FillLastDim(StmtInfoList &dst_info_list, StmtInfoList &src_info_list, StmtInfo &for_info) {
  CHECK(!dst_info_list.empty());
  CHECK(!src_info_list.empty());
  auto dst_info = dst_info_list[0];
  CHECK(dst_info.GetNode());
  int block_size = GetUbBlkSize(dst_info->dtype_);
  if (dst_info->strides_.empty() || GetInt32Const(GetItem(dst_info->strides_, -1)) != block_size) {
    return;
  }
  for (auto info : src_info_list) {
    CHECK(info.GetNode());
    if (info->strides_.empty() || GetInt32Const(GetItem(info->strides_, -1)) != block_size) {
      return;
    }
  }
  if (GetItem(for_info.ops_, -1).as<For>() && GetInt32Const(GetItem(for_info.ops_, -1).as<For>()->extent) == 1) {
    // for (cc1, 0, 1) {
    //     A[0] = max(A[0], B[cc1*16])
    // }
    // in this case don't append axis
    return;
  }

  Var tmp_var(DummyLastVar);
  Stmt body = Evaluate::make(Expr(0));
  if (!for_info.ops_.empty()) {
    CHECK(GetItem(for_info.ops_, -1).as<For>());
    body = GetItem(for_info.ops_, -1).as<For>()->body;
  }
  auto new_for = For::make(tmp_var, Expr(0), Expr(block_size), ForType::Serial, DeviceAPI::None, body);
  for_info.vars_.push_back(tmp_var);
  for_info.ops_.push_back(new_for);

  for (size_t i = 0; i < for_info.ops_.size() - 1; ++i) {
    auto idx = for_info.ops_.size() - 2 - i;
    auto current_for = for_info.ops_[idx].as<For>();
    CHECK(current_for != nullptr);
    SetItem(for_info.ops_, idx,
            For::make(current_for->loop_var, current_for->min, current_for->extent, current_for->for_type,
                      current_for->device_api, for_info.ops_[idx + 1]));
  }

  dst_info.GetNode()->strides_.push_back(1);
  dst_info.GetNode()->index_ += tmp_var;
  dst_info.GetNode()->shape_.push_back(block_size);
  dst_info.GetNode()->var_.push_back(tmp_var);
  dst_info_list.Set(0, dst_info);

  for (auto &info : src_info_list) {
    if (!Equal(dst_info, info)) {
      info.GetNode()->strides_.push_back(1);
      info.GetNode()->index_ += tmp_var;
      info.GetNode()->shape_.push_back(block_size);
      info.GetNode()->var_.push_back(tmp_var);
    }
  }
}

/// Fill empty variable with default value
/// \param info_list
void FillEmptyVar(Array<StmtStoreInfo> &info_list) {
  auto _Fill2ComInfo = [](const StmtStoreInfo com1_info, const StmtStoreInfo com2_info) {
    auto info = com2_info.GetNode();
    auto &com1_var = com1_info->var_;
    auto &com2_var = info->var_;
    auto &com2_shape = info->shape_;
    auto &com2_strides = info->strides_;
    size_t index = 0;
    for (auto &var1 : com1_var) {
      if (!IsInArray(com2_var, var1)) {
        Insert(com2_var, index, var1);
        Insert(com2_shape, index, Expr(0));
        Insert(com2_strides, index, Expr(0));
      }
      index += 1;
    }
    return com2_info;
  };

  auto _Clear = [](const StmtStoreInfo &info) {
    if (info->var_.empty()) {
      info.GetNode()->strides_ = {};
      info.GetNode()->shape_ = {};
    }
  };

  for (auto &info : info_list) {
    _Clear(info);
  }

  for (size_t i = 0; i < info_list.size(); ++i) {
    const auto &com1_info = info_list[i];
    for (size_t j = 0; j < info_list.size(); ++j) {
      if (i == j) {
        continue;
      }
      const auto &com2_info = info_list[j];
      info_list.Set(j, _Fill2ComInfo(com1_info, com2_info));
    }
  }
}

/// Get vector mask
/// \param data_len
/// \param data_num
/// \param data_type
/// \param begin
/// \return
Array<Expr> GetVecMask(int data_len, int data_num, const Type data_type, int begin) {
  auto vec_max_len_long = static_cast<uint64_t>(static_cast<uint32_t>(GetVecMaxLen(data_type)));
  auto block_size_long = static_cast<uint64_t>(static_cast<uint32_t>(GetUbBlkSize(data_type)));
  auto data_len_long = static_cast<uint64_t>(static_cast<uint32_t>(data_len));
  auto data_num_long = static_cast<uint64_t>(static_cast<uint32_t>(data_num));
  auto begin_long = static_cast<uint64_t>(static_cast<uint32_t>(begin));

  constexpr uint64_t quarter_vec_len = 32;
  constexpr uint64_t half_vec_len = 64;
  constexpr uint64_t full_vec_len = 128;

  if (data_type.bits() == BITS_PER_BYTE) {
    // this function only supports 16 and 32 bit operations. In case of bool, storage type is 8 bit, but
    // bitwise vector and/or/not intrinsics require 16 bit data
    vec_max_len_long >>= 1u;
    block_size_long >>= 1u;
    data_len_long = (data_len_long >> 1u) + (data_len_long & 1u);
  }
  if (data_len_long * data_num_long > vec_max_len_long || data_num_long < 1) {
    LOG(FATAL) << "Get vector mask error.";
  }
  // only 32, 64 and 128 vec_max_len are allowed
  if (vec_max_len_long != quarter_vec_len && vec_max_len_long != half_vec_len && vec_max_len_long != full_vec_len) {
    LOG(FATAL) << "Error: mask length is error.";
  }
  std::bitset<full_vec_len> submask, mask;
  for (size_t i = 0; i < half_vec_len; ++i) {
    submask = submask.set(i);
  }
  // get block mask
  if (data_len_long <= block_size_long && data_num_long > 1) {
    for (size_t j = 0; j < data_num_long; ++j) {
      for (size_t i = begin_long; i < data_len_long; ++i) {
        mask = mask.set(j * block_size_long + i);
      }
    }
  } else if (data_num_long == 1) {
    for (size_t i = begin_long; i < data_len_long; ++i) {
      mask = mask.set(i);
    }
  }

  Array<Expr> ret;
  ret.push_back(Expr(static_cast<uint64_t>(((mask >> half_vec_len) & submask).to_ullong())));
  ret.push_back(Expr(static_cast<uint64_t>((mask & submask).to_ullong())));
  return ret;
}

/// Remove zero strides in com_info
/// \param info_list
void CleanZeroStrides(StmtStoreInfo &info) {
  StmtInfoList info_list = {info};
  CleanZeroStrides(info_list);
  info = info_list[0];
}

/// Remove zero strides in info_list
/// \param info_list
void CleanZeroStrides(Array<StmtStoreInfo> &info_list) {
  auto _CleanComInfo = [](const StmtStoreInfo com_info) {
    auto info = com_info.GetNode();
    auto &com_var = info->var_;
    auto &com_shape = info->shape_;
    auto &com_strides = info->strides_;
    size_t i = 0;
    while (i < com_strides.size()) {
      if (GetIntConst(com_strides[i]) == 0) {
        com_var = RemoveItemAtIndex(com_var, i);
        com_shape = RemoveItemAtIndex(com_shape, i);
        com_strides = RemoveItemAtIndex(com_strides, i);
      } else {
        i += 1;
      }
    }
    if (com_var.empty()) {
      if (com_shape.empty()) {
        com_shape.push_back(Expr(int32_t(1)));
      }
      if (com_strides.empty()) {
        com_strides.push_back(Expr(int32_t(1)));
      }
    }
    return com_info;
  };

  for (size_t i = 0; i < info_list.size(); ++i) {
    info_list.Set(i, _CleanComInfo(info_list[i]));
  }
}

/// Swap axis in Array
/// \param var
/// \param shape
/// \param strides
/// \param idx1
/// \param idx2
void PatternGenerator::GetShapeInfoAndSwap(Array<Var> &var, Array<Expr> &shape, Array<Expr> &strides, int idx1,
                                           int idx2) {
  auto tmp_var = GetItem(var, idx1);
  SetItem(var, idx1, GetItem(var, idx2));
  SetItem(var, idx2, tmp_var);
  auto tmp_shape = GetItem(shape, idx1);
  SetItem(shape, idx1, GetItem(shape, idx2));
  SetItem(shape, idx2, tmp_shape);
  auto tmp_stride = GetItem(strides, idx1);
  SetItem(strides, idx1, GetItem(strides, idx2));
  SetItem(strides, idx2, tmp_stride);
}

/// Get insn args of load 2D intrin
/// \param intrin_name
/// \param dst_info_list
/// \param src_info_list
/// \param for_info
/// \return arg_info_map
Map<std::string, Expr> GetDmaLoad2DInsnArgs(const std::string &intrin_name, const StmtInfoList &dst_info_list,
                                            const StmtInfoList &src_info_list, StmtInfo &for_info) {
  if (dst_info_list.size() != 1u || src_info_list.size() != 1u) {
    LOG(FATAL) << "CCE DMA Load2D Insn only supports ONE dst and ONE src.";
  }

  Map<std::string, Expr> arg_info_map;
  auto &dst_info = dst_info_list[0];
  auto &src_info = src_info_list[0];
  bool transpose_call = false;

  CHECK(!dst_info->shape_.empty() && !src_info->shape_.empty())
    << "CCE Copy Insn Error: dst_buffer and src_buffer can not be scalar, should keep len(shape) > 0.";
  // get basic_args [n_burst, len_burst, src_stride, dst_stride]
  const auto &dst_var = dst_info->var_;
  const auto &dst_shape = dst_info->shape_;
  const auto &dst_strides = dst_info->strides_;
  const auto &src_shape = src_info->shape_;
  const auto &src_var = src_info->var_;
  const auto &src_strides = src_info->strides_;
  const int block_size = GetScopeBlockSize(dst_info, src_info);
  CHECK_NE(block_size, 0);

  Array<Var> elim_var;
  Expr repeat;
  Expr src_stride;
  // check the same loop-var
  if (GetItem(dst_var, -1).get() == GetItem(src_var, -1).get()) {
    // check if data continuous as row
    if (GetIntConst(GetItem(dst_strides, -1)) != 1 || GetIntConst(GetItem(src_strides, -1)) != 1) {
      LOG(FATAL) << "Error: Load2D move data is not continuous in last axis.";
    }
    // check if src and dst shapes are available
    if (GetIntConst(GetItem(dst_shape, -1)) % block_size != 0 ||
        GetIntConst(GetItem(src_shape, -1)) % block_size != 0) {
      LOG(FATAL) << "Error: Load2D move buffer shape is not multiple of block_size.";
    }
    if (GetIntConst(GetItem(dst_shape, -1)) != GetIntConst(GetItem(src_shape, -1))) {
      LOG(FATAL) << "Shape of dst and src MUST be the same.";
    }
    if (dst_var.size() > 1u && src_var.size() > 1u && GetItem(dst_var, -2).get() == GetItem(src_var, -2).get() &&
        GetIntConst(GetItem(dst_shape, -1)) == block_size && GetIntConst(GetItem(src_shape, -1)) == block_size &&
        GetIntConst(GetItem(dst_strides, -2)) == block_size &&
        GetIntConst(GetItem(src_strides, -2)) % block_size == 0) {
      repeat = GetItem(dst_shape, -2);
      src_stride = truncdiv(GetItem(src_strides, -2), block_size);
      elim_var = Array<Var>({GetItem(dst_var, -1), GetItem(dst_var, -2)});
    } else if (dst_var.size() > 2u && src_var.size() > 2u && GetItem(dst_var, -2).get() == GetItem(src_var, -3).get() &&
               GetItem(dst_var, -3).get() == GetItem(src_var, -2).get() &&
               GetIntConst(GetItem(dst_shape, -1)) == block_size && GetIntConst(GetItem(src_shape, -1)) == block_size &&
               GetIntConst(GetItem(dst_strides, -2)) == block_size &&
               GetIntConst(GetItem(src_strides, -3)) % block_size == 0) {
      repeat = GetItem(dst_shape, -2);
      src_stride = truncdiv(GetItem(src_strides, -3), block_size);
      elim_var = Array<Var>({GetItem(dst_var, -1), GetItem(dst_var, -2)});
    } else {
      repeat = truncdiv(GetItem(dst_shape, -1), block_size);
      src_stride = Expr(1);
      elim_var = Array<Var>({GetItem(dst_var, -1)});
    }
  } else {
    if (dst_var.size() < 2u && src_var.size() < 2u) {
      LOG(FATAL) << "Error: load_gm_to_cb does not support 1D data";
    }
    if (GetItem(dst_var, -1).get() != GetItem(src_var, -2).get() ||
        GetItem(dst_var, -2).get() != GetItem(src_var, -1).get()) {
      LOG(FATAL) << "Error: Load2D last axis of dst and src are different.";
    }
    if (intrin_name == "load_gm_to_cb") {
      LOG(FATAL) << "Error: load_gm_to_cb does not support transpose";
    }
    if (GetIntConst(GetItem(src_strides, -1)) != 1 || GetIntConst(GetItem(dst_strides, -1)) != 1) {
      LOG(FATAL) << "Error: Load2d move data is not continuous in last axis";
    }
    if (GetIntConst(GetItem(src_shape, -1)) != GetIntConst(GetItem(src_strides, -2)) ||
        GetIntConst(GetItem(dst_shape, -1)) != GetIntConst(GetItem(dst_strides, -2))) {
      LOG(FATAL) << "Error: transpose last two dims are not continuous.";
    }
    if (GetIntConst(GetItem(dst_shape, -1) * GetItem(dst_shape, -2)) != block_size ||
        GetIntConst(GetItem(src_shape, -1) * GetItem(src_shape, -2)) != block_size) {
      LOG(FATAL) << "Error: transpose last two dims can not match blocksize.";
    }
    if (dst_shape.size() < 2u || src_shape.size() < 2u) {
      LOG(FATAL) << "Error: dst_shape or src_shape size is less than 2";
    }
    if (dst_shape.size() == 2u && src_shape.size() == 2u) {
      repeat = Expr(1);
      src_stride = Expr(1);
      elim_var = GetRange(dst_var, -2, 2);
      // } else if (GetInt32Const(GetItem(dst_shape, -3)) && GetInt32Const(GetItem(src_shape, -3)) &&
    } else if (GetInt32Const(GetItem(dst_shape, -3)) == GetInt32Const(GetItem(src_shape, -3)) &&
               GetItem(dst_var, -3).get() == GetItem(src_var, -3).get() &&
               GetInt32Const(GetItem(dst_strides, -3)) == block_size &&
               GetInt32Const(GetItem(src_strides, -3)) % block_size == 0) {
      repeat = GetItem(dst_shape, -3);
      src_stride = truncdiv(GetItem(src_strides, -3), block_size);
      elim_var = GetRange(dst_var, -3, 3);
    } else if (dst_var.size() > 3u && src_var.size() > 3u && GetItem(dst_var, -3).get() == GetItem(src_var, -4).get() &&
               GetItem(dst_var, -4).get() == GetItem(src_var, -3).get() &&
               GetInt32Const(GetItem(dst_strides, -3)) == block_size &&
               GetInt32Const(GetItem(src_strides, -4)) % block_size == 0) {
      repeat = GetItem(dst_shape, -3);
      src_stride = truncdiv(GetItem(src_strides, -4), block_size);
      elim_var = GetRange(dst_var, -3, 3);
    } else {
      repeat = Expr(1);
      src_stride = Expr(1);
      elim_var = GetRange(dst_var, -2, 2);
    }
    transpose_call = true;
  }
  // get arg_info_map
  arg_info_map.Set("repeat", repeat);
  arg_info_map.Set("srcStride", src_stride);
  arg_info_map.Set("sid", 0);
  arg_info_map.Set("baseIdx", 0);
  arg_info_map.Set("transposeCall", transpose_call);
  // compute offset for cce instructions
  src_info.GetNode()->insn_offset_ = GetInsnOffset(src_info, elim_var);
  dst_info.GetNode()->insn_offset_ = GetInsnOffset(dst_info, elim_var);
  // clean vars in for_var
  CleanForInfoVars(for_info, elim_var);

  return arg_info_map;
}

/// Get computation info of dma copy
/// \param stmt
/// \param dst_info_list
/// \param src_info_list
/// \param if_info
/// \param for_info
/// \param dma_mode
/// \param intrin_name
void GetDmaComputationInfo(const Stmt &stmt, StmtInfoList &dst_info_list, StmtInfoList &src_info_list,
                           StmtInfo &if_info, StmtInfo &for_info, std::string &dma_mode, std::string &intrin_name) {
  GetCompactComputationInfo(stmt, dst_info_list, src_info_list, if_info, for_info);

  for (auto e : dst_info_list) {
    e.CleanFlexVar();
  }
  for (auto e : src_info_list) {
    e.CleanFlexVar();
  }

  CHECK(dst_info_list.size() == 1u && src_info_list.size() == 1u) << "CCE Copy Insn only support ONE dst and ONE src.";
  // get intrin name
  intrin_name = "";
  dma_mode = "";
  std::string dst_scope = dst_info_list[0]->scope_;
  std::string src_scope = src_info_list[0]->scope_;

  if (dst_scope == SCOPE_UBUF && src_scope == SCOPE_UBUF) {
    const auto &dst_var = dst_info_list[0]->var_;
    const auto &src_var = src_info_list[0]->var_;
    CHECK(!dst_var.empty() || src_var.empty());
    if (dst_var.empty() && src_var.empty()) {
      intrin_name = "copy_ubuf_to_ubuf";
    } else if (!dst_var.empty() && src_var.empty()) {
      intrin_name = "broadcast";
    } else if (GetItem(dst_var, -1).get() == GetItem(src_var, -1).get()) {
      intrin_name = "copy_ubuf_to_ubuf";
    } else if (dst_var.size() > 1u && src_var.size() > 1u && IsSame(dst_var, src_var, false) &&
               !IsSame(dst_var, src_var)) {
      // dst_var and src_var has same vars with different order, use vtranspose intrin
      intrin_name = "vtranspose";
    } else {
      // default intrin is broadcast
      intrin_name = "broadcast";
    }
  } else {
    std::map<std::string, std::string> buffer_map = {{DMA_COPY_GLOBAL, "gm"}, {SCOPE_CBUF, "cbuf"},
                                                     {SCOPE_UBUF, "ubuf"},    {SCOPE_CC, "matrix_cc"},
                                                     {SCOPE_CA, "ca"},        {SCOPE_CB, "cb"}};
    std::string intrin_header = dst_scope == SCOPE_CA || dst_scope == SCOPE_CB ? "load_" : "copy_";
    if (buffer_map.count(src_scope) == 0 || buffer_map.count(dst_scope) == 0) {
      LOG(FATAL) << "Unsupported CCE_MOV scope strategy. " + dst_scope;
    }
    intrin_name = intrin_header + buffer_map[src_scope] + "_to_" + buffer_map[dst_scope];
  }

  // get dma mode
  if (!intrin_name.empty()) {
    if (intrin_name.substr(0, 4) == "copy" || intrin_name == "vtranspose") {
      dma_mode = "cce_copy";
    } else if (intrin_name.substr(0, 4) == "load") {
      dma_mode = "cce_load";
    }
  }
}

/// Get CCE Copy Insn args
/// \param intrin_name
/// \param dst_info_list
/// \param src_info_list
/// \param for_info
/// \return
Map<std::string, Expr> GetDmaCopyInsnArgs(std::string &intrin_name, const StmtInfoList &dst_info_list,
                                          const StmtInfoList &src_info_list, StmtInfo &for_info) {
  Map<std::string, Expr> ub_copy_pre;
  Map<std::string, Expr> ub_copy_post;
  return GetDmaCopyInsnArgs(intrin_name, dst_info_list, src_info_list, for_info, ub_copy_pre, ub_copy_post);
}

/// Get CCE Copy Insn args
/// \param dst_info_list
/// \param src_info_list
/// \param for_info
/// \param ub_copy_pre
/// \param ub_copy_post
/// \return
Map<std::string, Expr> GetDmaCopyInsnArgs(std::string &intrin_name, const StmtInfoList &dst_info_list,
                                          const StmtInfoList &src_info_list, StmtInfo &for_info,
                                          Map<std::string, Expr> &ub_copy_pre, Map<std::string, Expr> &ub_copy_post) {
  CHECK(!src_info_list.empty());
  CHECK(!dst_info_list.empty());
  Map<std::string, Expr> arg_info_map;
  auto dst_info = dst_info_list[0];
  auto src_info = src_info_list[0];

  // check shape len
  if (dst_info->shape_.empty() || src_info->shape_.empty()) {
    LOG(FATAL) << "CCE Copy Insn Error: dst_buffer and src_buffer can not be scalar, should keep len(shape) > 0.";
  }

  int block_size = GetUbBlkSize(dst_info->dtype_);
  CHECK_NE(block_size, 0);

  // check alignment for vtranspose
  if (dst_info->scope_ == SCOPE_UBUF && src_info->scope_ == SCOPE_UBUF && intrin_name == "vtranspose") {
    int half_data_bit = 16;
    bool is_float16 = dst_info->dtype_.is_float() && dst_info->dtype_.bits() == half_data_bit &&
                      src_info->dtype_.is_float() && src_info->dtype_.bits() == half_data_bit;
    bool is_int16 = dst_info->dtype_.is_int() && dst_info->dtype_.bits() == half_data_bit &&
                    src_info->dtype_.is_int() && src_info->dtype_.bits() == half_data_bit;
    if (!is_float16 && !is_int16) {
      // b32 use scalar mov
      intrin_name = intrin_name.assign("vtranspose_scalar");
      return arg_info_map;
    } else {
      int shape0 = GetInt32Const(GetItem(src_info->shape_, -1));
      int shape1 = GetInt32Const(GetItem(src_info->shape_, -2));
      if (shape1 % block_size != 0 || (shape0 % block_size != 0 && (shape1 > block_size || shape0 > block_size))) {
        // use scalar mov
        intrin_name = intrin_name.assign("vtranspose_scalar");
        return arg_info_map;
      }
    }
  }

  auto gen_basic_args = [](const Expr n_burst, const Expr len_burst, const Expr src_stride, const Expr dst_stride) {
    Map<std::string, Expr> mp;
    mp.Set("nBurst", n_burst);
    mp.Set("lenBurst", len_burst);
    mp.Set("srcStride", src_stride);
    mp.Set("dstStride", dst_stride);
    return mp;
  };

  if (intrin_name == "vtranspose") {
    Expr loop_width = truncdiv(GetItem(src_info->strides_, -2), block_size);
    Expr loop_height = truncdiv(GetItem(dst_info->strides_, -2), block_size);
    Expr shape_width = GetItem(src_info->shape_, -1);
    Expr shape_height = GetItem(src_info->shape_, -2);
    if (GetIntConst(loop_width) == 0) {
      loop_width = Expr(1);
    }
    if (GetIntConst(loop_height) == 0) {
      loop_height = Expr(1);
    }
    arg_info_map.Set("loop_width", loop_width);
    arg_info_map.Set("loop_height", loop_height);
    arg_info_map.Set("shape_width", shape_width);

    Expr n_burst = Expr(16);
    Expr len_burst = Expr(1);
    Expr src_stride;
    Expr dst_stride;
    if (GetIntConst(loop_width) > 1) {
      src_stride = make_const(Int(32), GetIntConst(loop_width) - 1);
      dst_stride = Expr(0);
      ub_copy_pre = gen_basic_args(n_burst, len_burst, src_stride, dst_stride);
    }

    if (GetIntConst(loop_height) > 1) {
      src_stride = Expr(0);
      dst_stride = make_const(Int(32), GetIntConst(loop_height) - 1);
      ub_copy_post = gen_basic_args(n_burst, len_burst, src_stride, dst_stride);
    }

    if (GetIntConst(loop_width) == 1 && GetIntConst(loop_height) == 1 &&
        (GetIntConst(shape_width) != block_size || GetIntConst(shape_height) != block_size)) {
      ub_copy_post = gen_basic_args(1, shape_width, 0, 0);
    }

    auto elim_var = GetRange(dst_info->var_, -2, 2);
    src_info.GetNode()->insn_offset_ = GetInsnOffset(src_info, elim_var);
    dst_info.GetNode()->insn_offset_ = GetInsnOffset(dst_info, elim_var);
    CleanForInfoVars(for_info, elim_var);
  } else {
    Expr dst_last_dim = GetItem(dst_info->shape_, -1);
    Expr src_last_dim = GetItem(src_info->shape_, -1);
    Expr real_burst_size = Expr(1);

    // condition should use ub block_size
    int scope_block_size = GetScopeBlockSize(dst_info, src_info);
    CHECK_NE(scope_block_size, 0);
    if (dst_info->var_.empty() && !src_info->var_.empty()) {
      LOG(FATAL) << "Error: DMA Copy a Vector to a Scalar. Please check your algorithm.";
    }

    // default value if src_var and dst_var are both empty
    Expr n_burst = Expr(1);
    Expr len_burst = Expr(1);
    Expr src_stride = Expr(0);
    Expr dst_stride = Expr(0);
    Array<Var> elim_var;

    // check the same loop-var
    if (!dst_info->var_.empty() && !src_info->var_.empty()) {
      if (GetItem(dst_info->var_, -1).get() == GetItem(src_info->var_, -1).get()) {
        // check if data continuous as row
        if (GetInt32Const(GetItem(src_info->strides_, -1)) == 1 &&
            GetInt32Const(GetItem(dst_info->strides_, -1)) == 1 &&
            GetInt32Const(dst_last_dim) == GetInt32Const(src_last_dim)) {
          real_burst_size = dst_last_dim;
          if (dst_info->shape_.size() > 1u && src_info->shape_.size() > 1u &&
              GetItem(dst_info->var_, -2).get() == GetItem(src_info->var_, -2).get() &&
              GetInt32Const(GetItem(src_info->strides_, -2)) % scope_block_size == 0 &&
              GetInt32Const(GetItem(dst_info->strides_, -2)) % scope_block_size == 0 &&
              GetInt32Const(GetItem(src_info->strides_, -2)) >= GetInt32Const(GetItem(src_info->shape_, -1)) &&
              GetInt32Const(GetItem(dst_info->strides_, -2)) >= GetInt32Const(GetItem(dst_info->shape_, -1)) &&
              (intrin_name == INTRIN_NAME_COPY_UB_TO_UB || GetIntConst(dst_last_dim) % scope_block_size == 0)) {
            CHECK(IsTwoItemEqual(dst_info->shape_, src_info->shape_, -2, true))
              << "Shape of dst and src MUST be the same.";
            n_burst = GetItem(dst_info->shape_, -2);
            // if shape[-1] % block_size != 0, len_burst should be ceildiv of block_size
            len_burst = truncdiv(GetItem(dst_info->shape_, -1) + scope_block_size - 1, scope_block_size);
            src_stride = truncdiv(GetItem(src_info->strides_, -2), scope_block_size) - len_burst;
            dst_stride = truncdiv(GetItem(dst_info->strides_, -2), scope_block_size) - len_burst;
            elim_var = GetRange(dst_info->var_, -2, 2);
          } else {
            len_burst = make_const(Int(32), (GetInt32Const(dst_last_dim) + scope_block_size - 1) / scope_block_size);
            elim_var = GetRange(dst_info->var_, -1, 1);
          }
        }
      } else {
        len_burst =
          make_const(Int(32), GetInt32Const(GetItem(dst_info->shape_, -1) + scope_block_size - 1) / scope_block_size);
        elim_var = GetRange(dst_info->var_, -1, 1);
      }
    }
    CleanForInfoVars(for_info, elim_var);

    src_info.GetNode()->insn_offset_ = GetInsnOffset(src_info, elim_var);
    dst_info.GetNode()->insn_offset_ = GetInsnOffset(dst_info, elim_var);
    arg_info_map = gen_basic_args(n_burst, len_burst, src_stride, dst_stride);
    arg_info_map.Set("realBurstSize", GetIntConst(real_burst_size));

    // get mode args
    Expr pad_mode;
    Expr cr_mode;
    if (intrin_name == "copy_gm_to_cbuf") {
      pad_mode = Expr("PAD_NONE");  // padmode 0
    } else if (intrin_name == "copy_matrix_cc_to_ubuf") {
      std::string mode = "CRMODE_NONE";  // crmode 0
      if (dst_info->dtype_ != src_info->dtype_) {
        if (src_info->dtype_ == Float(32) && dst_info->dtype_ == Float(16)) {
          mode = "CRMODE_F32toF16_NONE";
        } else if (src_info->dtype_ == Float(16) && dst_info->dtype_ == Float(32)) {
          mode = "CRMODE_F16toF32_NONE";
        } else if (src_info->dtype_ == Int(32) && dst_info->dtype_ == Float(16)) {
          mode = "CRMODE_S32toF16_NONE";
        } else {
          LOG(FATAL) << "Unsupported data type transform form " << src_info->dtype_ << " to " << dst_info->dtype_;
        }
      }
      cr_mode = Expr(mode);
    }
    arg_info_map.Set("padMode", pad_mode);
    arg_info_map.Set("crMode", cr_mode);
  }
  // fix args
  if (intrin_name == "copy_matrix_cc_to_ubuf") {
    Expr dst_stride = arg_info_map["dstStride"];
    if (dst_info->dtype_ == Float(16)) {
      dst_stride = dst_stride * 16;
    } else if (dst_info->dtype_ == Float(32)) {
      dst_stride = dst_stride * 32;
    } else {
      LOG(FATAL) << "copy_matrix_cc_to_ubuf Unsupport such data type yet.";
    }
    arg_info_map.Set("dstStride", dst_stride);
  }
  arg_info_map.Set("sid", 0);

  return arg_info_map;
}

const char *const DummyLastVar = "cc_last";

TVM_REGISTER_API("cce_util.GetVecMask").set_body([](const TVMArgs args, TVMRetValue *ret) {
  *ret = GetVecMask(args[0], args[1], args[2]);
});
}  // namespace akg
