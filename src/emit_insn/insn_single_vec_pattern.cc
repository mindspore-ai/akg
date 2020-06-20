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
#include <tvm/base.h>
#include <tvm/ir_pass.h>

#include <cmath>
#include <set>

#include "insn_builder.h"
#include "insn_pattern.h"
#include "common/array_api.h"
#include "pass/expr_alg_simplify.h"

namespace akg {
/// Get CCE Single Vector Insn mode
/// \param dst_info_list
/// \param src_info_list
/// \return
std::string GetSingleVecMode(const StmtInfoList &dst_info_list, const StmtInfoList &src_info_list) {
  CHECK(!dst_info_list.empty());
  auto dst_var_list = dst_info_list[0]->var_;
  Array<Var> src_var_list;
  if (!src_info_list.empty()) {
    src_var_list = src_info_list[0]->var_;
  }

  if (IsSame(dst_var_list, src_var_list)) {
    return "elewise";
  } else if (dst_var_list.size() >= src_var_list.size()) {
    return "broadcast";
  }

  return "reduction";
}

/// Get Single Vector Computation Info
/// \param stmt
/// \param intrin_name
/// \param dst_info_list
/// \param src_info_list
/// \param if_info
/// \param for_info
/// \param need_compact
/// \return
std::string GetSingleVecComputationInfo(const Stmt &stmt, const std::string &intrin_name, StmtInfoList &dst_info_list,
                                        StmtInfoList &src_info_list, StmtInfo &if_info, StmtInfo &for_info,
                                        bool need_compact) {
  std::set<std::string> intrin_name_list = {"vadds", "vmuls",      "vrelu", "vabs",  "vln",   "vexp",
                                            "vrec",  "vector_dup", "vnot",  "vsqrt", "vrsqrt"};
  if (intrin_name_list.count(intrin_name) == 0 && intrin_name.find("vconv_") == std::string::npos) {
    LOG(FATAL) << "Error: CCE Single Vector Insn unsupported the given intrin_name. " << intrin_name;
    return "";
  }

  bool same_dtype = intrin_name.find("vconv_") == std::string::npos;
  GetCompactComputationInfo(stmt, dst_info_list, src_info_list, if_info, for_info, same_dtype, need_compact);
  std::string mode = GetSingleVecMode(dst_info_list, src_info_list);

  CHECK(dst_info_list.size() == 1) << "CCE Single Vector only support ONE dst.";

  return mode;
}

/// Get CCE Single vector instructions args.
/// \param dst_info_list
/// \param src_info_list
/// \param for_info
/// \param mode
/// \return
PatternResult SingleVecPatternGenerator::GetInsnArgs() {
  CalcParams();
  Array<Var> elim_var = {};
  float rate3d = Compute3DPatternMaskRate();
  float rate2db = Compute2DBlockPatternMaskRate();
  float rate2d = Compute2DPatternMaskRate();
  float rate1d = Compute1DPatternMaskRate();
  float rate3ds = Compute3DsPatternMaskRate();
  float rate2ds = Compute2DRepeatPatternMaskRate();
  if (mode == "broadcast_last_axis") {
    elim_var = Get1DPattern();
  } else if (rate2ds > 0) {
    elim_var = Get2DRepeatPattern();
  } else if (rate3ds > 0) {
    elim_var = Get3DsPattern();
    arg_info.GetNode()->pattern_ = PATTERN_2D;
  } else if (rate3d >= rate2db && rate3d > 0) {
    elim_var = Get3DPattern();
    arg_info.GetNode()->pattern_ = PATTERN_3D;
  } else if (rate2db >= rate2d && rate2db >= rate1d && rate2db > 0) {
    elim_var = Get2DBlockPattern();
    arg_info.GetNode()->pattern_ = PATTERN_PARTIAL_3D;
  } else if (rate2d > rate1d && rate2d > 0) {
    elim_var = Get2DPattern();
    arg_info.GetNode()->pattern_ = PATTERN_2D;
  } else if (rate1d > 0) {
    elim_var = Get1DPattern();
    arg_info.GetNode()->pattern_ = PATTERN_1D;
  } else {
    LOG(FATAL) << "Error: Cannot emit Single-Vector-Insn with any pattern!";
  }

  std::string mask_rate = "rate3d[" + std::to_string(rate3d) + "], rate2db[" + std::to_string(rate2db) + "], rate2d[" +
                          std::to_string(rate2d) + "], rate1d[" + std::to_string(rate1d) + "]";
  CommentManager::GetInstance().AddComment("Mask_rate", mask_rate);
  if (arg_info->tail_arg_info_.defined()) {
    CommentManager::GetInstance().AddComment("Contain_tail", "true");
  } else {
    CommentManager::GetInstance().AddComment("Contain_tail", "false");
  }

  return GenResult(elim_var);
}

/// Calc params for pattern match
void SingleVecPatternGenerator::CalcParams() {
  Array<StmtStoreInfo> info_list = {dst_info, src_info};
  // check shape len
  for (auto info : info_list) {
    CHECK(!info->shape_.empty())
      << "CCE Vector Insn Error: dst_buffer and src_buffer can not be scalar, should keep len(shape) > 0.";
  }

  FillEmptyVar(info_list);
  dst_info = info_list[0];
  src_info = info_list[1];

  int dst_bits = dst_info->dtype_.bits();
  int src_bits = src_info->dtype_.bits();
  CHECK_NE(dst_bits, 0);
  CHECK_NE(src_bits, 0);
  int dst_block_size = GetUbBlkSize(dst_info->dtype_);
  int src_block_size = GetUbBlkSize(src_info->dtype_);
  CHECK_NE(dst_block_size, 0);
  CHECK_NE(src_block_size, 0);

  data_type = src_bits > dst_bits ? src_info->dtype_ : dst_info->dtype_;

  params.dst_var = dst_info->var_;
  params.src_var = src_info->var_;
  params.dst_shape = dst_info->shape_;
  params.src_shape = src_info->shape_;
  params.dst_strides = dst_info->strides_;
  params.src_strides = src_info->strides_;
  params.dst_block_size = dst_block_size;
  params.src_block_size = src_block_size;
  params.mask_block_size = src_bits > dst_bits ? src_block_size : dst_block_size;
  params.dst_bits = dst_bits;
  params.src_bits = src_bits;
  params.max_bits = FULL_BLOCK_NUM * std::min(dst_bits, src_bits);
  params.dst_vec_max_len = GetVecMaxLen(dst_info->dtype_);
  params.vec_max_len = src_bits > dst_bits ? GetVecMaxLen(src_info->dtype_) : GetVecMaxLen(dst_info->dtype_);
  CHECK_NE(params.dst_vec_max_len, 0);
  CHECK_NE(params.vec_max_len, 0);

  auto GetNonZeroShapeByIdx = [this](int index) -> int {
    if (index <= static_cast<int>(params.dst_var.size())) {
      if (Equal(GetItem(params.dst_var, -index), GetItem(params.src_var, -index))) {
        return GetNonZeroShape(GetItem(params.dst_shape, -index), GetItem(params.src_shape, -index));
      }
    }
    return 1;
  };

  params.non_zero_shape1 = GetNonZeroShapeByIdx(1);
  params.non_zero_shape2 = GetNonZeroShapeByIdx(2);
  params.non_zero_shape3 = GetNonZeroShapeByIdx(3);
  params.all_points = params.non_zero_shape1 * params.non_zero_shape2 * params.non_zero_shape3;

  auto elem_offset_mod = ir::ExprSimplifier().Simplify(Mod::make(dst_info->elem_offset_, dst_block_size));
  if (elem_offset_mod.as<IntImm>()) {
    params.block_offset = elem_offset_mod.as<IntImm>()->value;
  }
}

int SingleVecPatternGenerator::GetLastDimShape(const Expr &dst_shape, const Expr &src_shape) {
  int dst_last_dim = GetInt32Const(dst_shape);
  int src_last_dim = GetInt32Const(src_shape);

  CHECK(dst_last_dim != 0 || src_last_dim != 0);
  if (dst_last_dim == 0) {
    return src_last_dim;
  }
  if (src_last_dim == 0) {
    return dst_last_dim;
  }
  return std::min(dst_last_dim, src_last_dim);
}

bool FindInShape(Array<Expr> &shape, const Expr &target) {
  for (int i = -1; i >= -3; --i) {
    if (Equal(GetItem(shape, i), target)) {
      return true;
    }
  }
  return false;
}

float SingleVecPatternGenerator::Compute2DRepeatPatternMaskRate() {
  if (params.dst_var.size() < 3) {
    return not_this_pattern;
  }

  for (int i = -1; i >= -3; --i) {
    if (!FindInShape(params.src_shape, GetItem(params.dst_shape, i))) {
      return not_this_pattern;
    }
  }

  if (GetInt32Const(GetItem(params.dst_strides, -2)) % params.dst_block_size != 0 ||
      GetInt32Const(GetItem(params.src_strides, -2)) % params.src_block_size != 0) {
    return not_this_pattern;
  }

  if (GetInt32Const(GetItem(params.dst_strides, -3)) % params.dst_block_size != 0 ||
      GetInt32Const(GetItem(params.src_strides, -3)) % params.src_block_size != 0) {
    return not_this_pattern;
  }

  if (GetInt32Const(GetItem(params.dst_strides, -2)) == 0 && GetInt32Const(GetItem(params.src_strides, -2)) == 0) {
    return not_this_pattern;
  }

  if (!Equal(GetItem(params.dst_shape, -3), GetItem(params.src_shape, -2)) ||
      !Equal(GetItem(params.dst_shape, -2), GetItem(params.src_shape, -3))) {
    return not_this_pattern;
  }
  if (GetIntConst(GetItem(params.dst_shape, -2)) > FULL_BLOCK_NUM &&
      GetIntConst(GetItem(params.dst_shape, -2)) % FULL_BLOCK_NUM != 0) {
    return not_this_pattern;
  }
  if (params.dst_block_size == params.src_block_size) {
    return not_this_pattern;
  }
  if (GetInt32Const(GetItem(params.dst_shape, -1)) <= params.dst_block_size &&
      GetInt32Const(GetItem(params.src_shape, -1)) <= params.src_block_size) {
    return not_this_pattern;
  }
  return 1.0;
}

float SingleVecPatternGenerator::Compute3DsPatternMaskRate() {
  if (params.dst_var.size() < 3) {
    return not_this_pattern;
  }
  if (params.dst_block_size != params.src_block_size) {
    return not_this_pattern;
  }
  for (int i = -1; i >= -3; --i) {
    if (!FindInShape(params.src_shape, GetItem(params.dst_shape, i))) {
      return not_this_pattern;
    }
  }
  if (GetInt32Const(GetItem(params.dst_shape, -1)) > params.dst_block_size ||
      GetInt32Const(GetItem(params.src_shape, -1)) > params.src_block_size) {
    return not_this_pattern;
  }

  if (GetInt32Const(GetItem(params.dst_strides, -2)) % params.dst_block_size != 0 ||
      GetInt32Const(GetItem(params.src_strides, -2)) % params.src_block_size != 0) {
    return not_this_pattern;
  }

  if (GetInt32Const(GetItem(params.dst_strides, -3)) % params.dst_block_size != 0 ||
      GetInt32Const(GetItem(params.src_strides, -3)) % params.src_block_size != 0) {
    return not_this_pattern;
  }

  if (GetInt32Const(GetItem(params.dst_strides, -2)) == 0 && GetInt32Const(GetItem(params.src_strides, -2)) == 0) {
    return not_this_pattern;
  }

  if (!Equal(GetItem(params.dst_shape, -3), GetItem(params.src_shape, -2)) ||
      !Equal(GetItem(params.dst_shape, -2), GetItem(params.src_shape, -3))) {
    return not_this_pattern;
  }
  if (GetIntConst(GetItem(params.dst_shape, -2)) > FULL_BLOCK_NUM &&
      GetIntConst(GetItem(params.dst_shape, -2)) % FULL_BLOCK_NUM != 0) {
    return not_this_pattern;
  }
  return 1.0;
}

float SingleVecPatternGenerator::Compute3DPatternMaskRate() {
  // in elemwise mode, the var is already checked to be equal, no need to check
  if (params.dst_var.size() < 3) {
    return not_this_pattern;
  }

  // do not support cast op in 3D pattern
  if (params.dst_block_size != params.src_block_size) {
    return not_this_pattern;
  }

  for (int i = -1; i >= -3; --i) {
    if (!IsTwoItemEqual(params.dst_var, params.src_var, i)) {
      return not_this_pattern;
    }
  }

  if (GetInt32Const(GetItem(params.dst_shape, -1)) > params.dst_block_size ||
      GetInt32Const(GetItem(params.src_shape, -1)) > params.src_block_size) {
    return not_this_pattern;
  }

  if (GetInt32Const(GetItem(params.dst_strides, -2)) % params.dst_block_size != 0 ||
      GetInt32Const(GetItem(params.src_strides, -2)) % params.src_block_size != 0) {
    return not_this_pattern;
  }

  if (GetInt32Const(GetItem(params.dst_strides, -3)) % params.dst_block_size != 0 ||
      GetInt32Const(GetItem(params.src_strides, -3)) % params.src_block_size != 0) {
    return not_this_pattern;
  }

  if (GetInt32Const(GetItem(params.dst_strides, -2)) == 0 && GetInt32Const(GetItem(params.src_strides, -2)) == 0) {
    return not_this_pattern;
  }

  // repeat axis is shape [-3], repeat once, has 8 loops
  bool is3_d = true;
  float rate3d_mode1 = not_this_pattern;
  float rate3d_mode2 = not_this_pattern;
  int repeat_num;
  float repeat_latency;
  StmtInfoList info_list = {dst_info, src_info};
  for (auto info : info_list) {
    if (GetInt32Const(GetItem(info->shape_, -2)) > FULL_BLOCK_NUM ||
        GetInt32Const(GetItem(info->strides_, -2)) / params.dst_block_size >= MAX_STRIDE_M0_SINGLE ||
        GetInt32Const(GetItem(info->strides_, -3)) / params.dst_block_size >= MAX_STRIDE_M1) {
      is3_d = false;
      break;
    }
  }
  if (is3_d) {
    if (GetIntConst(GetItem(params.dst_strides, -2)) == 0) {
      return not_this_pattern;
    }
    repeat_num = params.non_zero_shape3;
    repeat_latency = ((repeat_num - 1) / MAX_REPEAT) * repeat_latency_coef;
    rate3d_mode1 = static_cast<float>(params.all_points) / params.dst_vec_max_len / (repeat_num + repeat_latency);
  }

  is3_d = true;
  // repeat axis is shape[-2]
  for (auto info : info_list) {
    // stride_m0 should less than 65536
    if (GetInt32Const(GetItem(info->shape_, -3)) % FULL_BLOCK_NUM != 0 ||
        GetInt32Const(GetItem(info->strides_, -3)) / params.dst_block_size >= MAX_STRIDE_M0_SINGLE) {
      is3_d = false;
      break;
    }
  }
  if (is3_d) {
    if (GetIntConst(GetItem(params.dst_strides, -3)) == 0) {
      return not_this_pattern;
    }
    repeat_num = params.non_zero_shape2 * (params.non_zero_shape3 / FULL_BLOCK_NUM);
    repeat_latency = ((repeat_num - 1) / MAX_REPEAT) * repeat_latency_coef;
    float offset_latency =
      params.non_zero_shape3 / FULL_BLOCK_NUM > 1 ? params.non_zero_shape3 * offset_latency_coef : 0;
    rate3d_mode2 =
      static_cast<float>(params.all_points) / params.dst_vec_max_len / (repeat_num + repeat_latency + offset_latency);
  }

  return rate3d_mode1 > rate3d_mode2 ? rate3d_mode1 : rate3d_mode2;
}

// Partial 3D Pattern
float SingleVecPatternGenerator::Compute2DBlockPatternMaskRate() {
  // in elemwise mode, the var is already checked to be equal, no need to check
  if (params.dst_var.size() < 2 || params.src_var.size() < 2 || GetInt32Const(GetItem(params.dst_strides, -1)) != 1) {
    return not_this_pattern;
  }

  // do not support cast op in Partial3D pattern
  if (params.dst_block_size != params.src_block_size) {
    return not_this_pattern;
  }

  for (int i = -1; i >= -2; --i) {
    if (!Equal(GetItem(params.dst_var, i), GetItem(params.src_var, i))) {
      return not_this_pattern;
    }
  }

  if (GetInt32Const(GetItem(params.dst_shape, -1)) > params.dst_block_size ||
      GetInt32Const(GetItem(params.src_shape, -1)) > params.src_block_size) {
    return not_this_pattern;
  }

  if (GetInt32Const(GetItem(params.dst_strides, -2)) % params.dst_block_size != 0 ||
      GetInt32Const(GetItem(params.src_strides, -2)) % params.src_block_size != 0) {
    return not_this_pattern;
  }

  if (GetInt32Const(GetItem(params.dst_strides, -2)) == 0 && GetInt32Const(GetItem(params.src_strides, -2)) == 0) {
    return not_this_pattern;
  }

  if (GetInt32Const(GetItem(params.dst_strides, -2)) / params.dst_block_size >= MAX_STRIDE_M0_SINGLE ||
      GetInt32Const(GetItem(params.src_strides, -2)) / params.src_block_size >= MAX_STRIDE_M0_SINGLE) {
    return not_this_pattern;
  }

  if (GetInt32Const(GetItem(params.dst_strides, -2)) == 0) {
    return not_this_pattern;
  }

  int repeat_body_num = params.non_zero_shape2 / FULL_BLOCK_NUM;
  int repeat_tail_num = (params.non_zero_shape2 % FULL_BLOCK_NUM + FULL_BLOCK_NUM - 1) / FULL_BLOCK_NUM;
  int repeat_num = (repeat_body_num + repeat_tail_num) * params.non_zero_shape3;
  float repeat_latency =
    (std::max(repeat_body_num - 1, 0) / MAX_REPEAT + std::max(repeat_tail_num - 1, 0) / MAX_REPEAT) *
    repeat_latency_coef;
  float offset_latency = params.non_zero_shape3 > 1 ? params.non_zero_shape3 * offset_latency_coef : 0;
  float split_latency = (repeat_body_num > 0 && repeat_tail_num > 0) ? split_latency_coef : 0;
  float rate2db = static_cast<float>(params.all_points) / params.dst_vec_max_len /
                  (repeat_num + repeat_latency + offset_latency + split_latency);

  return rate2db;
}

float SingleVecPatternGenerator::Compute2DPatternMaskRate() {
  // in elemwise mode, the var is already checked to be equal, no need to check
  if (params.dst_var.size() < 2 || params.src_var.size() < 2) {
    return not_this_pattern;
  }

  if (src_info->data_alignment_ == 1 && GetInt32Const(GetItem(src_info->strides_, -1)) != params.dst_block_size) {
    return not_this_pattern;
  }

  for (int i = -1; i >= -2; --i) {
    if (!Equal(GetItem(params.dst_var, i), GetItem(params.src_var, i))) {
      return not_this_pattern;
    }
  }

  if (GetInt32Const(GetItem(params.dst_strides, -2)) % params.dst_block_size != 0 ||
      GetInt32Const(GetItem(params.src_strides, -2)) % params.src_block_size != 0) {
    return not_this_pattern;
  }

  if (GetInt32Const(GetItem(params.dst_strides, -2)) / params.dst_block_size >= MAX_STRIDE_M1 ||
      GetInt32Const(GetItem(params.src_strides, -2)) / params.src_block_size >= MAX_STRIDE_M1) {
    return not_this_pattern;
  }

  // check num of insns, select 1D pattern or 2D pattern
  int tail_factor = 0;
  if (params.non_zero_shape1 / params.dst_vec_max_len > 0 && params.non_zero_shape1 % params.dst_vec_max_len > 0) {
    tail_factor = 1;
  }

  int offset_num =
    (params.non_zero_shape1 + params.dst_vec_max_len - 1) / params.dst_vec_max_len * params.non_zero_shape3;
  int repeat_num = offset_num * params.non_zero_shape2;
  float repeat_latency = (std::max(params.non_zero_shape2 - 1, 0) / MAX_REPEAT) * offset_num * repeat_latency_coef;
  float offset_latency = offset_num > 1 ? offset_num * offset_latency_coef : 0;
  float split_latency = tail_factor * split_latency_coef;
  float rate2d = static_cast<float>(params.all_points) / params.dst_vec_max_len /
                 (repeat_num + repeat_latency + offset_latency + split_latency);

  return rate2d;
}

float SingleVecPatternGenerator::Compute1DPatternMaskRate() {
  int tail_factor = 0;
  if (params.non_zero_shape1 / params.dst_vec_max_len > 0 && params.non_zero_shape1 % params.dst_vec_max_len > 0) {
    tail_factor = 1;
  }

  int shape1 = (params.non_zero_shape1 + params.dst_vec_max_len - 1) / params.dst_vec_max_len;
  int repeat_num = shape1 * params.non_zero_shape2 * params.non_zero_shape3;
  float repeat_latency =
    std::max((shape1 - 1) / MAX_REPEAT, 0) * params.non_zero_shape2 * params.non_zero_shape3 * repeat_latency_coef;
  float offset_latency = params.non_zero_shape2 * params.non_zero_shape3 > 1
                           ? params.non_zero_shape2 * params.non_zero_shape3 * offset_latency_coef
                           : 0;
  float split_latency = tail_factor * split_latency_coef;
  float rate1d = static_cast<float>(params.all_points) / params.dst_vec_max_len /
                 (repeat_num + repeat_latency + offset_latency + split_latency);

  return rate1d;
}

Array<Var> SingleVecPatternGenerator::Get2DRepeatPattern() {
  GetShapeInfoAndSwap(params.src_var, params.src_shape, params.src_strides, -2, -3);
  int last_dim_shape = GetNonZeroShape(GetItem(params.dst_shape, -1), GetItem(params.src_shape, -1));
  body_args = VectorArgInfo(make_node<VectorArgInfoNode>());
  body_args.GetNode()->body_num_ = 1;
  body_args.GetNode()->dst_stride_m0_ = 1;
  body_args.GetNode()->src_stride_m0_list_ = {1};
  body_args.GetNode()->dst_stride_m1_ = truncdiv(GetItem(params.dst_strides, -2), params.dst_block_size);
  body_args.GetNode()->src_stride_m1_list_ = {truncdiv(GetItem(params.src_strides, -2), params.src_block_size)};
  body_args.GetNode()->repeat_ = GetItem(params.dst_shape, -2);
  int data_len = CeilTo(last_dim_shape, params.dst_block_size);
  body_args.GetNode()->vec_mask_ = GetVecMask(data_len, 1, dst_info->dtype_);
  return GetRange(params.dst_var, -2, 2);
}

Array<Var> SingleVecPatternGenerator::Get3DsPattern() {
  GetShapeInfoAndSwap(params.src_var, params.src_shape, params.src_strides, -2, -3);
  int last_dim_shape = GetNonZeroShape(GetItem(params.dst_shape, -1), GetItem(params.src_shape, -1));
  body_args = VectorArgInfo(make_node<VectorArgInfoNode>());
  body_args.GetNode()->body_num_ = 1;

  Expr dst_stride_m0 = truncdiv(GetItem(params.dst_strides, -2), params.dst_block_size);
  Expr src_stride_m0 = truncdiv(GetItem(params.src_strides, -2), params.src_block_size);
  body_args.GetNode()->dst_stride_m0_ = dst_stride_m0;
  body_args.GetNode()->src_stride_m0_list_ = {src_stride_m0};

  int block_num = 0;
  int data_len = CeilTo(last_dim_shape, params.mask_block_size);

  if (GetIntConst(GetItem(params.dst_shape, -2)) <= FULL_BLOCK_NUM) {
    block_num = GetIntConst(GetItem(params.dst_shape, -2));
    body_args.GetNode()->dst_stride_m1_ = truncdiv(GetItem(params.dst_strides, -3), params.dst_block_size);
    body_args.GetNode()->src_stride_m1_list_ = {truncdiv(GetItem(params.src_strides, -3), params.src_block_size)};
    body_args.GetNode()->vec_mask_ = GetVecMask(data_len, block_num, dst_info->dtype_);
    auto repeat = GetItem(params.dst_shape, -3);
    if (GetIntConst(repeat) < MAX_STRIDE_M1) {
      body_args.GetNode()->repeat_ = repeat;
      return GetRange(params.dst_var, -3, 3);
    } else {
      body_args.GetNode()->repeat_ = 1;
      return GetRange(params.dst_var, -2, 2);
    }
  } else {
    block_num = FULL_BLOCK_NUM;
    body_args.GetNode()->dst_stride_m1_ = dst_stride_m0 * block_num;
    body_args.GetNode()->src_stride_m1_list_ = {src_stride_m0 * block_num};
    auto repeat = truncdiv(GetItem(params.dst_shape, -2), FULL_BLOCK_NUM);
    body_args.GetNode()->vec_mask_ = GetVecMask(data_len, block_num, dst_info->dtype_);
    if (GetIntConst(repeat) < MAX_STRIDE_M1) {
      body_args.GetNode()->repeat_ = repeat;
      return GetRange(params.dst_var, -2, 2);
    } else {
      return Get1DPattern();
    }
  }
}

Array<Var> SingleVecPatternGenerator::Get3DPattern() {
  if (GetIntConst(GetNonZeroShape(GetItem(params.dst_shape, -2), GetItem(params.src_shape, -2))) > FULL_BLOCK_NUM) {
    // split shape[-3]
    if (GetIntConst(GetNonZeroShape(GetItem(params.dst_shape, -3), GetItem(params.src_shape, -3))) > FULL_BLOCK_NUM) {
      StmtInfoList info_list = {dst_info, src_info};
      SplitAxis(info_list, for_info, GetItem(params.dst_var, -3), FULL_BLOCK_NUM);
      FillEmptyVar(info_list);
      dst_info = info_list[0];
      src_info = info_list[1];

      params.dst_var = dst_info->var_;
      params.dst_shape = dst_info->shape_;
      params.dst_strides = dst_info->strides_;
      params.src_var = src_info->var_;
      params.src_shape = src_info->shape_;
      params.src_strides = src_info->strides_;
    }
    // consider original shape[-2] as repeat axis
    GetShapeInfoAndSwap(params.dst_var, params.dst_shape, params.dst_strides, -2, -3);
    GetShapeInfoAndSwap(params.src_var, params.src_shape, params.src_strides, -2, -3);
  }

  int last_dim_shape = GetNonZeroShape(GetItem(params.dst_shape, -1), GetItem(params.src_shape, -1));
  body_args = VectorArgInfo(make_node<VectorArgInfoNode>());
  body_args.GetNode()->body_num_ = 1;
  body_args.GetNode()->repeat_ =
    make_const(Int(32), GetNonZeroShape(GetItem(params.dst_shape, -3), GetItem(params.src_shape, -3)));
  body_args.GetNode()->dst_stride_m0_ = truncdiv(GetItem(params.dst_strides, -2), params.dst_block_size);
  body_args.GetNode()->dst_stride_m1_ = truncdiv(GetItem(params.dst_strides, -3), params.dst_block_size);
  body_args.GetNode()->src_stride_m0_list_ = {truncdiv(GetItem(params.src_strides, -2), params.src_block_size)};
  body_args.GetNode()->src_stride_m1_list_ = {truncdiv(GetItem(params.src_strides, -3), params.src_block_size)};
  body_args.GetNode()->block_offset_ = make_const(Int(32), params.block_offset);

  int data_len = CeilTo(last_dim_shape, params.mask_block_size);
  int data_num = GetInt32Const(GetItem(params.dst_shape, -2));
  body_args.GetNode()->vec_mask_ = GetVecMask(data_len, data_num, dst_info->dtype_, params.block_offset);

  return GetRange(params.dst_var, -3, 3);
}

Array<Var> SingleVecPatternGenerator::Get2DBlockPattern() {
  int last_dim_shape = GetNonZeroShape(GetItem(params.dst_shape, -1), GetItem(params.src_shape, -1));
  int repeat_len = GetNonZeroShape(GetItem(params.dst_shape, -2), GetItem(params.src_shape, -2));
  int repeat_body = repeat_len / FULL_BLOCK_NUM;
  int repeat_tail = (repeat_len % FULL_BLOCK_NUM + FULL_BLOCK_NUM - 1) / FULL_BLOCK_NUM;
  int data_len = CeilTo(last_dim_shape, params.dst_block_size);

  if (repeat_body > 0) {
    body_args = VectorArgInfo(make_node<VectorArgInfoNode>());
    body_args.GetNode()->body_num_ = 1;
    body_args.GetNode()->repeat_ = make_const(Int(32), repeat_body);
    auto dst_stride_m0 = truncdiv(GetItem(params.dst_strides, -2), params.dst_block_size);
    body_args.GetNode()->dst_stride_m0_ = dst_stride_m0;
    body_args.GetNode()->dst_stride_m1_ = dst_stride_m0 * (params.max_bits / params.src_bits);
    auto src_stride_m0 = truncdiv(GetItem(params.src_strides, -2), params.src_block_size);
    body_args.GetNode()->src_stride_m0_list_ = {src_stride_m0};
    body_args.GetNode()->src_stride_m1_list_ = {src_stride_m0 * (params.max_bits / params.dst_bits)};
    body_args.GetNode()->block_offset_ = make_const(Int(32), params.block_offset);

    int data_num = FULL_BLOCK_NUM;
    body_args.GetNode()->vec_mask_ = GetVecMask(data_len, data_num, dst_info->dtype_, params.block_offset);
  }

  if (repeat_tail > 0) {
    tail_args = VectorArgInfo(make_node<VectorArgInfoNode>());
    tail_args.GetNode()->dst_head_ = GetItem(params.dst_strides, -2) * repeat_body * FULL_BLOCK_NUM;
    tail_args.GetNode()->src_head_list_ = {GetItem(params.src_strides, -2) * repeat_body * FULL_BLOCK_NUM};
    tail_args.GetNode()->repeat_ = Expr(1);
    tail_args.GetNode()->dst_stride_m0_ = truncdiv(GetItem(params.dst_strides, -2), params.dst_block_size);
    tail_args.GetNode()->dst_stride_m1_ = Expr(0);
    tail_args.GetNode()->src_stride_m0_list_ = {truncdiv(GetItem(params.src_strides, -2), params.src_block_size)};
    tail_args.GetNode()->src_stride_m1_list_ = {Expr(0)};
    tail_args.GetNode()->block_offset_ = make_const(Int(32), params.block_offset);

    int data_num = repeat_len % FULL_BLOCK_NUM;
    tail_args.GetNode()->vec_mask_ = GetVecMask(data_len, data_num, dst_info->dtype_, params.block_offset);
  }

  return GetRange(params.dst_var, -2, 2);
}

Array<Var> SingleVecPatternGenerator::Get2DPattern() {
  const int data_num = 1;
  int last_dim_shape = GetNonZeroShape(GetItem(params.dst_shape, -1), GetItem(params.src_shape, -1));
  if (GetInt32Const(GetItem(dst_info->strides_, -1)) == params.dst_block_size &&
      IsTwoItemEqual(dst_info->strides_, src_info->strides_, -1, true)) {
    last_dim_shape *= params.dst_block_size;
  }
  int body_len = FloorTo(last_dim_shape, params.vec_max_len);
  int tail_len = last_dim_shape % params.vec_max_len;

  if (body_len > 0) {
    body_args = VectorArgInfo(make_node<VectorArgInfoNode>());
    body_args.GetNode()->body_num_ = body_len / params.vec_max_len;
    body_args.GetNode()->body_offset_ = params.vec_max_len;
    body_args.GetNode()->repeat_ = GetItem(params.dst_shape, -2);
    body_args.GetNode()->dst_stride_m0_ = Expr(1);
    body_args.GetNode()->dst_stride_m1_ = truncdiv(GetItem(params.dst_strides, -2), params.dst_block_size);
    body_args.GetNode()->src_stride_m0_list_ = {Expr(1)};
    body_args.GetNode()->src_stride_m1_list_ = {truncdiv(GetItem(params.src_strides, -2), params.src_block_size)};
    body_args.GetNode()->block_offset_ = make_const(Int(32), params.block_offset);
    body_args.GetNode()->vec_mask_ = GetVecMask(params.vec_max_len, data_num, data_type, params.block_offset);
  }

  // get tail params
  if (tail_len > 0) {
    tail_args = VectorArgInfo(make_node<VectorArgInfoNode>());
    tail_args.GetNode()->dst_head_ = Expr(body_len);
    tail_args.GetNode()->src_head_list_ = {Expr(body_len)};
    tail_args.GetNode()->repeat_ = GetItem(params.dst_shape, -2);
    tail_args.GetNode()->dst_stride_m0_ = Expr(1);
    tail_args.GetNode()->dst_stride_m1_ = truncdiv(GetItem(params.dst_strides, -2), params.dst_block_size);
    tail_args.GetNode()->src_stride_m0_list_ = {Expr(1)};
    tail_args.GetNode()->src_stride_m1_list_ = {truncdiv(GetItem(params.src_strides, -2), params.src_block_size)};
    tail_args.GetNode()->block_offset_ = make_const(Int(32), params.block_offset);

    int data_len = CeilTo(tail_len, params.mask_block_size);
    tail_args.GetNode()->vec_mask_ = GetVecMask(data_len, data_num, data_type, params.block_offset);
  }

  return GetRange(params.dst_var, -2, 2);
}

Array<Var> SingleVecPatternGenerator::Get1DPattern() {
  int last_dim_shape;
  bool linear_mode = false;
  if ((params.dst_shape.empty() && params.src_shape.empty()) || GetIntConst(GetItem(params.dst_shape, -1)) == 0) {
    last_dim_shape = 1;
  } else if (!IsTwoItemEqual(params.dst_var, params.src_var, -1)) {
    last_dim_shape = 1;
  } else {
    last_dim_shape = GetLastDimShape(GetItem(params.dst_shape, -1), GetItem(params.src_shape, -1));
    linear_mode = params.dst_bits == params.src_bits;
  }

  bool is_scalar_mode = IsScalarMode({dst_info, src_info});
  if (is_scalar_mode && params.dst_bits != params.src_bits) {
    last_dim_shape = 1;
  }
  int vec_max_len = is_scalar_mode ? FULL_BLOCK_NUM : params.vec_max_len;
  int body_len = FloorTo(last_dim_shape, vec_max_len);
  int tail_len = last_dim_shape % vec_max_len;

  auto dst_stride_m0 =
    is_scalar_mode && linear_mode ? truncdiv(GetItem(params.dst_strides, -1), params.dst_block_size) : Expr(1);
  auto src_stride_m0 =
    is_scalar_mode && linear_mode ? truncdiv(GetItem(params.src_strides, -1), params.src_block_size) : Expr(1);
  if (body_len > 0) {
    body_args = VectorArgInfo(make_node<VectorArgInfoNode>());
    body_args.GetNode()->body_num_ = 1;
    body_args.GetNode()->body_offset_ = vec_max_len;
    body_args.GetNode()->repeat_ = Expr(body_len / vec_max_len);
    body_args.GetNode()->dst_stride_m0_ = dst_stride_m0;
    auto dst_block_num = is_scalar_mode ? FULL_BLOCK_NUM : (params.max_bits / params.src_bits);
    body_args.GetNode()->dst_stride_m1_ = dst_stride_m0 * dst_block_num;
    body_args.GetNode()->src_stride_m0_list_ = {src_stride_m0};
    auto src_block_num = is_scalar_mode ? FULL_BLOCK_NUM : (params.max_bits / params.dst_bits);
    body_args.GetNode()->src_stride_m1_list_ = {src_stride_m0 * src_block_num};
    body_args.GetNode()->block_offset_ = make_const(Int(32), params.block_offset);

    // in cast case, data_num should be 1 because dst and src bit is not equal
    int data_len = is_scalar_mode ? 1 : vec_max_len;
    int data_num = is_scalar_mode ? FULL_BLOCK_NUM : 1;
    body_args.GetNode()->vec_mask_ = GetVecMask(data_len, data_num, data_type, params.block_offset);
  }

  // get tail params
  if (tail_len > 0) {
    tail_args = VectorArgInfo(make_node<VectorArgInfoNode>());
    tail_args.GetNode()->body_offset_ = vec_max_len;
    tail_args.GetNode()->body_num_ = 1;
    tail_args.GetNode()->dst_head_ =
      Expr(body_len * (is_scalar_mode ? dst_stride_m0 * params.dst_block_size : Expr(1)));
    tail_args.GetNode()->src_head_list_ = {
      Expr(body_len * (is_scalar_mode ? src_stride_m0 * params.src_block_size : Expr(1)))};
    tail_args.GetNode()->repeat_ = Expr(1);
    tail_args.GetNode()->dst_stride_m0_ = dst_stride_m0;
    tail_args.GetNode()->dst_stride_m1_ = Expr(0);
    tail_args.GetNode()->src_stride_m0_list_ = {src_stride_m0};
    tail_args.GetNode()->src_stride_m1_list_ = {Expr(0)};
    tail_args.GetNode()->block_offset_ = make_const(Int(32), params.block_offset);

    int data_len = is_scalar_mode && linear_mode ? 1 : CeilTo(tail_len, params.mask_block_size);
    int data_num = is_scalar_mode && linear_mode ? tail_len : 1;
    data_num = data_num == 0 ? 1 : data_num;
    tail_args.GetNode()->vec_mask_ = GetVecMask(data_len, data_num, data_type, params.block_offset);
  }

  // compute offset for cce instructions
  Array<Var> elim_var = {};
  if (mode == "elewise" && params.dst_var.size() >= 2 && params.dst_strides.size() >= 2 && for_info.ops_.size() >= 2 &&
      last_dim_shape <= vec_max_len && last_dim_shape >= vec_max_len - params.dst_block_size &&
      GetIntConst(GetItem(params.dst_strides, -2)) == last_dim_shape) {
    // in this case we can merge second last for extent to repeat
    size_t index = 0;
    bool suc = GetIndexOfElement(for_info.vars_, GetItem(params.dst_var, -2), index);
    CHECK(suc);
    auto latest_for = GetItem(for_info.ops_, index).as<For>();
    // there should not be if_op between for loop and compute stmt
    if (latest_for && !latest_for->body->IsInstance<IfThenElse>()) {
      if (!params.dst_var.empty() && (!is_scalar_mode || last_dim_shape != 1)) {
        if (body_args.defined()) {
          // last_dim_shape = vec_max_len
          body_args.GetNode()->repeat_ = body_args->repeat_ * latest_for->extent;
        } else if (tail_args.defined()) {
          // last_dim_shape < vec_max_len
          tail_args.GetNode()->repeat_ = tail_args->repeat_ * latest_for->extent;
        }

        return GetRange(params.dst_var, -2, 2);
      }
    }
  }

  if (!params.dst_var.empty() && (!is_scalar_mode || last_dim_shape != 1 || linear_mode) &&
      GetIntConst(GetItem(params.dst_strides, -1)) > 0 &&
      (params.src_var.empty() || IsTwoItemEqual(params.dst_var, params.src_var, -1))) {
    elim_var = GetRange(params.dst_var, -1, 1);
  }

  return elim_var;
}

PatternResult SingleVecPatternGenerator::GenResult(const Array<Var> &elim_var) {
  arg_info.GetNode()->body_arg_info_ = body_args;
  arg_info.GetNode()->tail_arg_info_ = tail_args;

  dst_info.GetNode()->insn_offset_ = GetInsnOffset(dst_info, elim_var);
  src_info.GetNode()->insn_offset_ = GetInsnOffset(src_info, elim_var);

  CleanForInfoVars(for_info, elim_var);

  StmtInfoList info_list = {dst_info, src_info};
  CleanZeroStrides(info_list);
  dst_info = info_list[0];
  src_info = info_list[1];

  PatternResult result;
  result.dst_info_list = {dst_info};
  result.src_info_list = {src_info};
  result.for_info = for_info;
  result.arg_info = arg_info;

  return result;
}
}  // namespace akg
