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

#include <set>

#include "ir_pass.h"
#include "contrib/cce_parm/cceconf.h"
#include "tvm.h"
#include "common/array_api.h"
#include "insn_pattern.h"
#include "insn_builder.h"

namespace akg {
std::string GetBinaryVecMode(const StmtInfoList &dst_info_list, const StmtInfoList &src_info_list,
                             const std::string &intrin_name, bool enable_bisect = true) {
  std::set<std::string> reduce_bisect_list = {"vadd", "vsub", "vmul", "vmax"};
  std::string mode = "reduction";
  if (IsElementwise(dst_info_list, src_info_list)) {
    mode = "elewise";
  } else if (IsBroadcast(dst_info_list, src_info_list)) {
    mode = "broadcast";
  } else if (IsLastAxisReduction(dst_info_list, src_info_list)) {
    mode = "reduce_last_axis";
  } else if (enable_bisect && reduce_bisect_list.count(intrin_name) != 0 &&
             IsBisectionReduction(dst_info_list, src_info_list)) {
    mode = "reduce_bisection";
  }

  return mode;
}

PatternResult ReduceLastAxisPatternGenerator::GetInsnArgs() {
  CalcParams();
  Array<Var> elim_var;

  float rate2d = Compute2DBlockPatternMaskRate();
  if (rate2d > 1.0f) {
    elim_var = Get2DBlockPattern();
    arg_info.GetNode()->pattern_ = PATTERN_2D_BLOCK;
  } else {
    elim_var = Get1DPattern();
    arg_info.GetNode()->pattern_ = PATTERN_1D;
  }

  return GenResult(elim_var);
}

float ReduceLastAxisPatternGenerator::Compute2DBlockPatternMaskRate() {
  const float is2_dpattern = 1.0f;
  if (intrin_name == "vadd" || intrin_name == "argmax" || intrin_name == "argmin") {
    return not_this_pattern;
  }

  // src var size must larger than 2
  if (params.src_var.size() < 2) {
    return not_this_pattern;
  }

  int body_len = params.last_dim_shape / params.vec_max_len * params.vec_max_len;
  int tail_len = params.last_dim_shape % params.vec_max_len;

  // there is no body in this mode
  if (body_len > 0 || tail_len > params.block_size) {
    return not_this_pattern;
  }

  return is2_dpattern;
}

Array<Var> ReduceLastAxisPatternGenerator::Get2DBlockPattern() {
  int sec_last_dim_shape = GetInt32Const(GetItem(src_info->shape_, -2));
  int body_len = sec_last_dim_shape / FULL_BLOCK_NUM * FULL_BLOCK_NUM;
  int tail_len = sec_last_dim_shape % FULL_BLOCK_NUM;
  int cmd_body_len = 0;

  if (body_len > 0) {
    body_args = VectorArgInfo(make_node<VectorArgInfoNode>());
    body_args.GetNode()->body_num_ = 1;
    body_args.GetNode()->repeat_ = Expr(body_len / FULL_BLOCK_NUM);
    // Here use dst_stride_m1 as dst_stride
    body_args.GetNode()->dst_stride_m1_ = Expr(1);
    body_args.GetNode()->src_stride_m0_list_ = {Expr(1)};
    body_args.GetNode()->src_stride_m1_list_ = {Expr(FULL_BLOCK_NUM)};
    body_args.GetNode()->vec_mask_ = GetVecMask(params.last_dim_shape, FULL_BLOCK_NUM, dst_info->dtype_);
    cmd_body_len += GetInt32Const(body_args->repeat_) * FULL_BLOCK_NUM;
  }
  if (tail_len > 0) {
    tail_args = VectorArgInfo(make_node<VectorArgInfoNode>());
    tail_args.GetNode()->dst_head_ = Expr(cmd_body_len);
    tail_args.GetNode()->src_head_list_ = {Expr(cmd_body_len / FULL_BLOCK_NUM * params.vec_max_len)};
    tail_args.GetNode()->repeat_ = Expr(1);
    tail_args.GetNode()->dst_stride_m1_ = Expr(1);
    tail_args.GetNode()->src_stride_m0_list_ = {Expr(1)};
    tail_args.GetNode()->src_stride_m1_list_ = {Expr(0)};
    tail_args.GetNode()->vec_mask_ = GetVecMask(params.last_dim_shape, tail_len, dst_info->dtype_);
  }

  params.insn_offset_scale_factor = 1;
  return GetRange(params.src_var, -2, 2);
}

Array<Var> ReduceLastAxisPatternGenerator::Get1DPattern() {
  int body_len = params.last_dim_shape / params.vec_max_len * params.vec_max_len;
  int tail_len = params.last_dim_shape % params.vec_max_len;
  int cmd_body_len = 0;
  bool is_vadd = intrin_name == "vadd";
  int repeat_stride = FULL_BLOCK_NUM;
  if (is_vadd) {
    repeat_stride = 1;
  }
  const int fp16_block_size = 16;

  if (body_len > 0) {
    body_args = VectorArgInfo(make_node<VectorArgInfoNode>());
    body_args.GetNode()->body_num_ = 1;
    body_args.GetNode()->body_offset_ = params.vec_max_len;
    body_args.GetNode()->repeat_ = Expr(body_len / params.vec_max_len);
    // Here use dst_stride_m1 as dst_stride
    body_args.GetNode()->dst_stride_m1_ = Expr(1);
    body_args.GetNode()->src_stride_m0_list_ = {Expr(1)};
    body_args.GetNode()->src_stride_m1_list_ = {Expr(FULL_BLOCK_NUM)};
    body_args.GetNode()->vec_mask_ = GetVecMask(params.vec_max_len, 1, dst_info->dtype_);
    cmd_body_len += GetInt32Const(body_args->repeat_) * repeat_stride;
  }
  if (tail_len > 0) {
    tail_args = VectorArgInfo(make_node<VectorArgInfoNode>());
    tail_args.GetNode()->body_offset_ = params.vec_max_len;
    tail_args.GetNode()->dst_head_ = Expr(cmd_body_len);
    tail_args.GetNode()->src_head_list_ = {Expr(body_len)};
    tail_args.GetNode()->repeat_ = Expr(1);
    tail_args.GetNode()->dst_stride_m1_ = Expr(1);
    tail_args.GetNode()->src_stride_m0_list_ = {Expr(1)};
    tail_args.GetNode()->src_stride_m1_list_ = {Expr(0)};
    tail_args.GetNode()->vec_mask_ = GetVecMask(tail_len, 1, dst_info->dtype_);
    if (is_vadd) {
      cmd_body_len += 1;
    } else {
      cmd_body_len += tail_len / fp16_block_size;
      if (tail_len % fp16_block_size != 0) {
        cmd_body_len += 1;
      }
    }
  }
  // cmd_body_len > 1 means vcadd size greater than 128, need to use vcadd again to compute final result
  // if cmd_body_len > 128, then need to recursively emit vcadd
  while (cmd_body_len > 1) {
    int cmd_tail_len = cmd_body_len % params.vec_max_len;
    cmd_body_len = cmd_body_len / params.vec_max_len;
    if (cmd_body_len > 0) {
      VectorArgInfo mix_vec_args = VectorArgInfo(make_node<VectorArgInfoNode>());
      mix_vec_args.GetNode()->repeat_ = Expr(cmd_body_len);
      mix_vec_args.GetNode()->dst_head_ = Expr(0);
      mix_vec_args.GetNode()->src_head_list_ = {Expr(0)};
      mix_vec_args.GetNode()->dst_stride_m1_ = Expr(1);
      mix_vec_args.GetNode()->src_stride_m0_list_ = {Expr(1)};
      mix_vec_args.GetNode()->src_stride_m1_list_ = {Expr(FULL_BLOCK_NUM)};
      mix_vec_args.GetNode()->vec_mask_ = GetVecMask(params.vec_max_len, 1, dst_info->dtype_);
      mix_vec_arg_list.push_back(mix_vec_args);
      if (!is_vadd) {
        cmd_body_len *= FULL_BLOCK_NUM;
      }
    }
    if (cmd_tail_len > 0) {
      VectorArgInfo mix_vec_args = VectorArgInfo(make_node<VectorArgInfoNode>());
      mix_vec_args.GetNode()->repeat_ = Expr(1);
      mix_vec_args.GetNode()->dst_head_ = Expr(cmd_body_len);
      if (is_vadd) {
        mix_vec_args.GetNode()->src_head_list_ = {Expr(cmd_body_len * params.vec_max_len)};
      } else {
        mix_vec_args.GetNode()->src_head_list_ = {Expr(cmd_body_len / FULL_BLOCK_NUM * params.vec_max_len)};
      }
      mix_vec_args.GetNode()->dst_stride_m1_ = Expr(1);
      mix_vec_args.GetNode()->src_stride_m0_list_ = {Expr(1)};
      mix_vec_args.GetNode()->src_stride_m1_list_ = {Expr(FULL_BLOCK_NUM)};
      mix_vec_args.GetNode()->vec_mask_ = GetVecMask(cmd_tail_len, 1, dst_info->dtype_);
      if (is_vadd) {
        cmd_body_len += 1;
      } else {
        cmd_body_len += cmd_tail_len / fp16_block_size;
        if (cmd_tail_len % fp16_block_size != 0) {
          cmd_body_len += 1;
        }
      }
      mix_vec_arg_list.push_back(mix_vec_args);
    }
  }

  params.insn_offset_scale_factor = Expr(params.block_size);
  int max_num = body_len / params.vec_max_len;
  if (intrin_name == "vmax" || intrin_name == "vmin") {
    max_num *= FULL_BLOCK_NUM;
  }
  if (max_num >= params.block_size) {
    params.insn_offset_scale_factor = max_num + params.block_size - 1;
    if (tail_len > 0) {
      params.insn_offset_scale_factor += 1;
    }
    params.insn_offset_scale_factor = truncdiv(params.insn_offset_scale_factor, params.block_size) * params.block_size;
  }

  if (!params.src_var.empty()) {
    return GetRange(params.src_var, -1, 1);
  }

  return {};
}

PatternResult ReduceLastAxisPatternGenerator::GenResult(const Array<Var> &elim_var) {
  dst_info.GetNode()->insn_offset_ = GetInsnOffset(dst_info, elim_var) * params.insn_offset_scale_factor;
  src_info.GetNode()->insn_offset_ = GetInsnOffset(src_info, elim_var);

  if (body_args.defined()) {
    body_args.GetNode()->insn_offset_scale_factor_ = params.insn_offset_scale_factor;
  }
  if (tail_args.defined()) {
    tail_args.GetNode()->insn_offset_scale_factor_ = params.insn_offset_scale_factor;
  }
  for (auto &arg : mix_vec_arg_list) {
    arg.GetNode()->insn_offset_scale_factor_ = params.insn_offset_scale_factor;
  }

  arg_info.GetNode()->body_arg_info_ = body_args;
  arg_info.GetNode()->tail_arg_info_ = tail_args;
  arg_info.GetNode()->reduction_tail_args_ = mix_vec_arg_list;

  CleanForInfoVars(for_info, elim_var);
  arg_info.GetNode()->arg_type_ = ARG_VECTOR_REDUCTION_LAST_AXIS;

  PatternResult result;
  result.dst_info_list = {dst_info};
  result.src_info_list = {src_info};
  result.for_info = for_info;
  result.arg_info = arg_info;

  return result;
}

void ReduceLastAxisPatternGenerator::CalcParams() {
  // check shape len
  if (dst_info->shape_.empty() || src_info->shape_.empty()) {
    LOG(FATAL) << "CCE Vector Insn Error: dst_buffer and src_buffer can not be scalar, should keep len(shape) > 0.";
  }
  // check data type
  if (dst_info->dtype_ != src_info->dtype_) {
    LOG(FATAL) << "CCE Vector Insn Error: dst_buffer and src_buffer can not be different data type.";
  }

  params.src_var = src_info->var_;
  params.block_size = GetUbBlkSize(dst_info->dtype_);
  params.last_dim_shape = GetInt32Const(GetItem(src_info->shape_, -1));
  params.vec_max_len = GetVecMaxLen(dst_info->dtype_);
  CHECK_NE(params.block_size, 0);
  CHECK_NE(params.vec_max_len, 0);
}

/// Get CCE Binary Vector instructions args
/// \return
PatternResult BinaryVecPatternGenerator::GetInsnArgs() {
  CalcParams();
  if (arg_info->arg_type_ == ARG_VECTOR_BROADCAST_LAST_AXIS) {
    PatternResult result;
    result.dst_info_list = {dst_info};
    result.src_info_list = src_info_list;
    result.for_info = for_info;
    result.arg_info = arg_info;
    return result;
  }

  Array<Var> elim_var = {};

  float rate3d = Compute3DPatternMaskRate();
  float rate2db = Compute2DBlockPatternMaskRate();
  float rate2d = Compute2DPatternMaskRate();
  float rate1d = Compute1DPatternMaskRate();

  if (rate3d >= rate2db && rate3d > 0) {
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
    LOG(FATAL) << "Error: Cannot emit Binary-Vector-Insn with any pattern!";
  }

  std::string mask_rate = "rate3d[" + std::to_string(rate3d) + "], rate2db[" + std::to_string(rate2db) + "], rate2d[" +
                          std::to_string(rate2d) + "], rate1d[" + std::to_string(rate1d) + "]";
  CommentManager::GetInstance().AddComment("Mask_rate", mask_rate);
  if (tail_args.defined()) {
    CommentManager::GetInstance().AddComment("Contain_tail", "true");
  } else {
    CommentManager::GetInstance().AddComment("Contain_tail", "false");
  }

  return GenResult(elim_var);
}

float BinaryVecPatternGenerator::Compute3DPatternMaskRate() {
  if (params.non_zero_shape3 == 1 || params.non_zero_shape2 == 1) {
    return not_this_pattern;
  }
  // in elemwise mode, the var is already checked to be equal, no need to check
  if (params.dst_var.size() < 3 || GetIntConst(GetItem(params.dst_shape, -1)) > params.block_size ||
      GetIntConst(GetItem(params.dst_strides, -2)) % params.block_size != 0 ||
      GetIntConst(GetItem(params.dst_strides, -3)) % params.block_size != 0 ||
      (GetIntConst(GetItem(params.dst_strides, -2)) > 0 && GetIntConst(GetItem(params.dst_shape, -1)) > 0 &&
       GetIntConst(GetItem(params.dst_strides, -2)) < GetIntConst(GetItem(params.dst_shape, -1))) ||
      (GetIntConst(GetItem(params.dst_strides, -3)) > 0 && GetIntConst(GetItem(params.dst_shape, -2)) > 0 &&
       GetIntConst(GetItem(params.dst_strides, -3)) < GetIntConst(GetItem(params.dst_shape, -2)))) {
    return not_this_pattern;
  }
  // check dst_stride_m0
  // As described in ISL User Guide t6.3,
  // dst_stride_m0 = 0 is treated as 1
  auto JudgeNot3D = [this](const StmtStoreInfo &info) {
    auto last_shape1 = GetIntConst(GetItem(info->shape_, -1));
    if (info->var_.size() < 3 || last_shape1 > params.block_size) {
      return true;
    }

    auto last_shape2 = GetIntConst(GetItem(info->shape_, -2));
    auto last_stride2 = GetIntConst(GetItem(info->strides_, -2));
    auto last_stride3 = GetIntConst(GetItem(info->strides_, -3));

    return last_stride2 % params.block_size != 0 || last_stride3 % params.block_size != 0 ||
           (last_stride2 > 0 && last_shape1 > 0 && last_stride2 < last_shape1) ||
           (last_stride3 > 0 && last_shape2 > 0 && last_stride3 < last_shape2);
  };
  if (std::any_of(src_info_list.begin(), src_info_list.end(), JudgeNot3D)) {
    return not_this_pattern;
  }

  if (mode == "reduction") {
    // check same alignment
    Array<Expr> shape_list = {GetItem(params.dst_shape, -1)};
    shape_list.push_back(GetItem(params.src_shape0, -1));
    shape_list.push_back(GetItem(params.src_shape1, -1));
    if (!IsNonZeroShapeEqual(shape_list)) {
      return not_this_pattern;
    }
  }

  // repeat axis is shape [-3], repeat once, has 8 loops
  bool is3_d = true;
  float rate3d_mode1 = not_this_pattern;
  float rate3d_mode2 = not_this_pattern;
  int repeat_num;
  float repeat_latency;
  auto info_list = src_info_list;
  Insert(info_list, 0, dst_info);
  for (auto info : info_list) {
    if (GetInt32Const(GetItem(info->shape_, -2)) > FULL_BLOCK_NUM ||
        GetInt32Const(GetItem(info->strides_, -2)) / params.block_size >= MAX_STRIDE_M0 ||
        GetInt32Const(GetItem(info->strides_, -3)) / params.block_size >= MAX_STRIDE_M0) {
      is3_d = false;
      break;
    }
  }
  if (is3_d) {
    if (GetInt32Const(GetItem(params.dst_strides, -2)) == 0) {
      return not_this_pattern;
    }
    repeat_num = params.non_zero_shape3;
    repeat_latency = ((repeat_num - 1) / MAX_REPEAT) * repeat_latency_coef;
    rate3d_mode1 = static_cast<float>(params.all_points) / params.vec_max_len / (repeat_num + repeat_latency);
  }

  is3_d = true;
  // repeat axis is shape[-2]
  for (auto info : info_list) {
    // stride_m0 should be less than 256
    if (GetIntConst(GetItem(info->shape_, -3)) % FULL_BLOCK_NUM != 0 ||
        GetIntConst(GetItem(info->strides_, -3)) / params.block_size >= MAX_STRIDE_M0) {
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
      static_cast<float>(params.all_points) / params.vec_max_len / (repeat_num + repeat_latency + offset_latency);
  }

  return rate3d_mode1 > rate3d_mode2 ? rate3d_mode1 : rate3d_mode2;
}

float BinaryVecPatternGenerator::Compute2DBlockPatternMaskRate() {
  if (params.non_zero_shape2 == 1 || GetIntConst(GetItem(params.dst_strides, -1)) != 1) {
    return not_this_pattern;
  }
  if (params.dst_var.size() < 2 || GetIntConst(GetItem(params.dst_shape, -1)) > params.block_size ||
      GetIntConst(GetItem(params.dst_strides, -2)) % params.block_size != 0 ||
      GetIntConst(GetItem(params.dst_strides, -2)) / params.block_size >= MAX_STRIDE_M0 ||
      (GetIntConst(GetItem(params.dst_strides, -2)) > 0 && GetIntConst(GetItem(params.dst_shape, -1)) > 0 &&
       GetIntConst(GetItem(params.dst_strides, -2)) < GetIntConst(GetItem(params.dst_shape, -1)))) {
    return not_this_pattern;
  }

  for (auto info : src_info_list) {
    if (info->var_.size() < 2 || GetIntConst(GetItem(info->shape_, -1)) > params.block_size ||
        GetIntConst(GetItem(info->strides_, -2)) % params.block_size != 0 ||
        GetIntConst(GetItem(info->strides_, -2)) / params.block_size >= MAX_STRIDE_M0 ||
        (GetIntConst(GetItem(info->strides_, -2)) > 0 && GetIntConst(GetItem(info->shape_, -1)) > 0 &&
         GetIntConst(GetItem(info->strides_, -2)) < GetIntConst(GetItem(info->shape_, -1)))) {
      return not_this_pattern;
    }
  }

  if (GetIntConst(GetItem(params.dst_strides, -2)) == 0) {
    return not_this_pattern;
  }

  if (mode == "reduction") {
    if (params.dst_var.size() > 2) {
      // if not elewise mode, then can not use partial 3D mode
      if (GetIntConst(GetItem(params.dst_shape, -3)) == 0) {
        return not_this_pattern;
      }

      for (auto info : src_info_list) {
        if (GetIntConst(GetItem(info->shape_, -3)) == 0) {
          return not_this_pattern;
        }
      }
    }
    // check same alignment
    Array<Expr> shape_list = {GetItem(params.dst_shape, -1)};
    shape_list.push_back(GetItem(params.src_shape0, -1));
    shape_list.push_back(GetItem(params.src_shape1, -1));
    // check dst_stride_m0
    // As described in ISL User Guide t6.3,
    // dst_stride_m0 = 0 is treated as 1
    if (!IsNonZeroShapeEqual(shape_list)) {
      return not_this_pattern;
    }
  }

  int repeat_body_num = params.non_zero_shape2 / FULL_BLOCK_NUM;
  int repeat_tail_num = (params.non_zero_shape2 % FULL_BLOCK_NUM + FULL_BLOCK_NUM - 1) / FULL_BLOCK_NUM;
  int repeat_num = (repeat_body_num + repeat_tail_num) * params.non_zero_shape3;
  float repeat_latency =
    (std::max(repeat_body_num - 1, 0) / MAX_REPEAT + std::max(repeat_tail_num - 1, 0) / MAX_REPEAT) *
    repeat_latency_coef;
  float offset_latency = params.non_zero_shape3 > 1 ? params.non_zero_shape3 * offset_latency_coef : 0;
  float split_latency = (repeat_body_num > 0 && repeat_tail_num > 0) ? split_latency_coef : 0;
  float rate2db = static_cast<float>(params.all_points) / params.vec_max_len /
                  (repeat_num + repeat_latency + offset_latency + split_latency);

  return rate2db;
}

float BinaryVecPatternGenerator::Compute2DPatternMaskRate() {
  if (params.non_zero_shape2 == 1) {
    return not_this_pattern;
  }
  if (params.dst_var.size() < 2 || GetIntConst(GetItem(params.dst_strides, -2)) % params.block_size != 0 ||
      (GetIntConst(GetItem(params.dst_strides, -2)) < GetIntConst(GetItem(params.dst_shape, -1)) &&
       GetIntConst(GetItem(params.dst_strides, -2) > 0))) {
    return not_this_pattern;
  }

  for (auto info : src_info_list) {
    if (info->var_.size() < 2 || GetIntConst(GetItem(info->strides_, -2)) % params.block_size != 0 ||
        (GetIntConst(GetItem(info->strides_, -2)) < GetIntConst(GetItem(info->shape_, -1)) &&
         GetIntConst(GetItem(info->strides_, -2) > 0))) {
      return not_this_pattern;
    }
  }

  // check num of insns, select 1D pattern or 2D pattern
  int tail_factor = 0;
  if (mode == "reduction") {
    Array<Expr> shape_list = {GetItem(params.dst_shape, -1)};
    shape_list.push_back(GetItem(params.src_shape0, -1));
    shape_list.push_back(GetItem(params.src_shape1, -1));
    if (!IsNonZeroShapeEqual(shape_list)) {
      return not_this_pattern;
    }
  }

  // only cloud allow dst_stride_m1 = 0
  cceconf::CceConf *conf = cceconf::CceConf::getInstance();
  const std::string product_name = conf->getProductName();
  if (GetIntConst(GetItem(params.dst_strides, -2)) == 0 && product_name != "cloud") {
    return not_this_pattern;
  }

  CHECK_NE(params.vec_max_len, 0);
  if (params.non_zero_shape1 / params.vec_max_len > 0 && params.non_zero_shape1 % params.vec_max_len > 0) {
    tail_factor = 1;
  }

  if (GetIntConst(GetItem(dst_info->strides_, -2)) / params.block_size >= MAX_STRIDE_M0) {
    return not_this_pattern;
  }
  for (auto info : src_info_list) {
    if (GetIntConst(GetItem(info->strides_, -2)) / params.block_size >= MAX_STRIDE_M0) {
      return not_this_pattern;
    }
  }

  int shape1 = (params.non_zero_shape1 + params.vec_max_len - 1) / params.vec_max_len;
  int repeat_num = shape1 * params.non_zero_shape2 * params.non_zero_shape3;
  float repeat_latency =
    (std::max(params.non_zero_shape2 - 1, 0) / MAX_REPEAT) * params.non_zero_shape3 * shape1 * repeat_latency_coef;
  float offset_latency =
    shape1 * params.non_zero_shape3 > 1 ? shape1 * params.non_zero_shape3 * offset_latency_coef : 0;
  float split_latency = tail_factor * split_latency_coef;
  float rate2d = static_cast<float>(params.all_points) / params.vec_max_len /
                 (repeat_num + repeat_latency + offset_latency + split_latency);

  return rate2d;
}

float BinaryVecPatternGenerator::Compute1DPatternMaskRate() {
  int tail_factor = 0;
  if (params.non_zero_shape1 / params.vec_max_len > 0 && params.non_zero_shape1 % params.vec_max_len > 0) {
    tail_factor = 1;
  }

  int shape1 = (params.non_zero_shape1 + params.vec_max_len - 1) / params.vec_max_len;
  int repeat_num = shape1 * params.non_zero_shape2 * params.non_zero_shape3;
  float repeat_latency =
    std::max((shape1 - 1) / MAX_REPEAT, 0) * params.non_zero_shape2 * params.non_zero_shape3 * repeat_latency_coef;
  float offset_latency = params.non_zero_shape2 * params.non_zero_shape3 > 1
                           ? params.non_zero_shape2 * params.non_zero_shape3 * offset_latency_coef
                           : 0;
  float split_latency = tail_factor * split_latency_coef;
  float rate1d = static_cast<float>(params.all_points) / params.vec_max_len /
                 (repeat_num + repeat_latency + offset_latency + split_latency);

  return rate1d;
}

Array<Var> BinaryVecPatternGenerator::Get3DPattern() {
  // repeat axis is shape [-2]
  int second_last_shape = GetInt32Const(
    GetNonZeroShape(GetItem(params.dst_shape, -2), GetItem(params.src_shape0, -2), GetItem(params.src_shape1, -2)));
  int third_last_shape = GetInt32Const(
    GetNonZeroShape(GetItem(params.dst_shape, -3), GetItem(params.src_shape0, -3), GetItem(params.src_shape1, -3)));
  if (second_last_shape > 8) {
    // split shape[-3]
    if (third_last_shape > 8) {
      auto info_list = src_info_list;
      Insert(info_list, 0, dst_info);
      SplitAxis(info_list, for_info, GetItem(params.dst_var, -3), FULL_BLOCK_NUM);
      FillEmptyVar(info_list);

      params.dst_var = info_list[0]->var_;
      params.dst_shape = info_list[0]->shape_;
      params.dst_strides = info_list[0]->strides_;
      params.src_var0 = info_list[1]->var_;
      params.src_shape0 = info_list[1]->shape_;
      params.src_strides0 = info_list[1]->strides_;
      params.src_var1 = info_list[2]->var_;
      params.src_shape1 = info_list[2]->shape_;
      params.src_strides1 = info_list[2]->strides_;
    }
    // consider original shape[-2] as repeat axis
    GetShapeInfoAndSwap(params.dst_var, params.dst_shape, params.dst_strides, -2, -3);
    GetShapeInfoAndSwap(params.src_var0, params.src_shape0, params.src_strides0, -2, -3);
    GetShapeInfoAndSwap(params.src_var1, params.src_shape1, params.src_strides1, -2, -3);
  }

  body_args = VectorArgInfo(make_node<VectorArgInfoNode>());
  CHECK(body_args.GetNode());
  body_args.GetNode()->body_num_ = 1;
  body_args.GetNode()->repeat_ = GetItem(params.dst_shape, -3);

  body_args.GetNode()->dst_stride_m0_ = truncdiv(GetItem(params.dst_strides, -2), params.block_size);
  body_args.GetNode()->dst_stride_m1_ = truncdiv(GetItem(params.dst_strides, -3), params.block_size);
  body_args.GetNode()->src_stride_m0_list_ = {truncdiv(GetItem(params.src_strides0, -2), params.block_size),
                                             truncdiv(GetItem(params.src_strides1, -2), params.block_size)};
  body_args.GetNode()->src_stride_m1_list_ = {truncdiv(GetItem(params.src_strides0, -3), params.block_size),
                                             truncdiv(GetItem(params.src_strides1, -3), params.block_size)};

  int data_num = GetInt32Const(GetItem(params.dst_shape, -2));
  if (mode == "reduction") {
    body_args.GetNode()->repeat_ = Expr(
      GetNonZeroShape(GetItem(params.dst_shape, -3), GetItem(params.src_shape0, -3), GetItem(params.src_shape1, -3)));
    data_num =
      GetNonZeroShape(GetItem(params.dst_shape, -2), GetItem(params.src_shape0, -2), GetItem(params.src_shape1, -2));
  }
  int data_len = expand_mask ? CeilTo(params.last_dim_shape, params.block_size) : params.last_dim_shape;
  body_args.GetNode()->vec_mask_ = GetVecMask(data_len, data_num, dst_info->dtype_);

  return GetRange(params.dst_var, -3, 3);
}

Array<Var> BinaryVecPatternGenerator::Get2DBlockPattern() {
  int repeat_len = GetInt32Const(GetItem(params.dst_shape, -2));
  if (mode == "reduction") {
    params.last_dim_shape =
      GetNonZeroShape(GetItem(params.dst_shape, -1), GetItem(params.src_shape0, -1), GetItem(params.src_shape1, -1));
    repeat_len =
      GetNonZeroShape(GetItem(params.dst_shape, -2), GetItem(params.src_shape0, -2), GetItem(params.src_shape1, -2));
  }
  int repeat_body = repeat_len / FULL_BLOCK_NUM;
  int repeat_tail = (repeat_len % FULL_BLOCK_NUM + FULL_BLOCK_NUM - 1) / FULL_BLOCK_NUM;

  if (repeat_body > 0) {
    body_args = VectorArgInfo(make_node<VectorArgInfoNode>());
    CHECK(body_args.GetNode() != nullptr);
    body_args.GetNode()->body_num_ = 1;
    body_args.GetNode()->repeat_ = Expr(repeat_body);
    body_args.GetNode()->dst_stride_m0_ = truncdiv(GetItem(params.dst_strides, -2), params.block_size);
    body_args.GetNode()->dst_stride_m1_ = body_args->dst_stride_m0_ * FULL_BLOCK_NUM;
    Expr src0_stride_m0 = truncdiv(GetItem(params.src_strides0, -2), params.block_size);
    Expr src1_stride_m0 = truncdiv(GetItem(params.src_strides1, -2), params.block_size);
    body_args.GetNode()->src_stride_m0_list_ = {src0_stride_m0, src1_stride_m0};
    body_args.GetNode()->src_stride_m1_list_ = {src0_stride_m0 * FULL_BLOCK_NUM, src1_stride_m0 * FULL_BLOCK_NUM};
    int data_len = expand_mask ? CeilTo(params.last_dim_shape, params.block_size) : params.last_dim_shape;
    body_args.GetNode()->vec_mask_ = GetVecMask(data_len, FULL_BLOCK_NUM, dst_info->dtype_);
  }
  if (repeat_tail > 0) {
    tail_args = VectorArgInfo(make_node<VectorArgInfoNode>());
    CHECK(tail_args.GetNode() != nullptr);
    tail_args.GetNode()->dst_head_ = GetItem(params.dst_strides, -2) * repeat_body * FULL_BLOCK_NUM;
    tail_args.GetNode()->src_head_list_ = {GetItem(params.src_strides0, -2) * repeat_body * FULL_BLOCK_NUM,
                                          GetItem(params.src_strides1, -2) * repeat_body * FULL_BLOCK_NUM};
    tail_args.GetNode()->repeat_ = Expr(1);
    tail_args.GetNode()->dst_stride_m0_ = truncdiv(GetItem(params.dst_strides, -2), params.block_size);
    tail_args.GetNode()->dst_stride_m1_ = Expr(0);
    tail_args.GetNode()->src_stride_m0_list_ = {truncdiv(GetItem(params.src_strides0, -2), params.block_size),
                                               truncdiv(GetItem(params.src_strides1, -2), params.block_size)};
    tail_args.GetNode()->src_stride_m1_list_ = {Expr(0), Expr(0)};
    int data_len = expand_mask ? CeilTo(params.last_dim_shape, params.block_size) : params.last_dim_shape;
    tail_args.GetNode()->vec_mask_ = GetVecMask(data_len, repeat_len % FULL_BLOCK_NUM, dst_info->dtype_);
  }
  return GetRange(params.dst_var, -2, 2);
}

Array<Var> BinaryVecPatternGenerator::Get2DPattern() {
  if (mode == "reduction") {
    params.last_dim_shape =
      GetNonZeroShape(GetItem(params.dst_shape, -1), GetItem(params.src_shape0, -1), GetItem(params.src_shape1, -1));
  }

  int body_len = FloorTo(params.last_dim_shape, params.vec_max_len);
  int tail_len = params.last_dim_shape % params.vec_max_len;

  if (body_len > 0) {
    body_args = VectorArgInfo(make_node<VectorArgInfoNode>());
    CHECK(body_args.GetNode() != nullptr);
    body_args.GetNode()->body_num_ = body_len / params.vec_max_len;
    body_args.GetNode()->body_offset_ = params.vec_max_len;
    body_args.GetNode()->repeat_ = GetItem(params.dst_shape, -2);
    if (mode == "reduction") {
      body_args.GetNode()->repeat_ =
        GetNonZeroShape(GetItem(params.dst_shape, -2), GetItem(params.src_shape0, -2), GetItem(params.src_shape1, -2));
    }
    body_args.GetNode()->dst_stride_m0_ = Expr(1);
    body_args.GetNode()->dst_stride_m1_ = truncdiv(GetItem(params.dst_strides, -2), params.block_size);
    body_args.GetNode()->src_stride_m0_list_ = {Expr(1), Expr(1)};
    body_args.GetNode()->src_stride_m1_list_ = {truncdiv(GetItem(params.src_strides0, -2), params.block_size),
                                               truncdiv(GetItem(params.src_strides1, -2), params.block_size)};
    body_args.GetNode()->vec_mask_ = GetVecMask(params.vec_max_len, 1, dst_info->dtype_);
  }
  if (tail_len > 0) {
    tail_args = VectorArgInfo(make_node<VectorArgInfoNode>());
    CHECK(tail_args.GetNode() != nullptr);
    tail_args.GetNode()->dst_head_ = Expr(body_len);
    tail_args.GetNode()->src_head_list_ = {Expr(body_len), Expr(body_len)};
    tail_args.GetNode()->repeat_ = GetItem(params.dst_shape, -2);
    if (mode == "reduction") {
      tail_args.GetNode()->repeat_ =
        GetNonZeroShape(GetItem(params.dst_shape, -2), GetItem(params.src_shape0, -2), GetItem(params.src_shape1, -2));
    }
    tail_args.GetNode()->dst_stride_m0_ = Expr(1);
    tail_args.GetNode()->dst_stride_m1_ = truncdiv(GetItem(params.dst_strides, -2), params.block_size);
    tail_args.GetNode()->src_stride_m0_list_ = {Expr(1), Expr(1)};
    tail_args.GetNode()->src_stride_m1_list_ = {truncdiv(GetItem(params.src_strides0, -2), params.block_size),
                                               truncdiv(GetItem(params.src_strides1, -2), params.block_size)};
    tail_args.GetNode()->vec_mask_ = GetVecMask(tail_len, 1, dst_info->dtype_);
  }
  return GetRange(params.dst_var, -2, 2);
}

Array<Var> BinaryVecPatternGenerator::Get1DPattern() {
  auto info_list = src_info_list;
  Insert(info_list, 0, dst_info);
  bool is_scalar_mode = IsScalarMode(info_list);
  if (is_scalar_mode) {
    params.last_dim_shape = 1;
  }

  if (mode == "reduction") {
    params.last_dim_shape =
      GetNonZeroShape(GetItem(params.dst_shape, -1), GetItem(params.src_shape0, -1), GetItem(params.src_shape1, -1));
  }
  int body_len = FloorTo(params.last_dim_shape, params.vec_max_len);
  int tail_len = params.last_dim_shape % params.vec_max_len;

  int last_axis = -1;
  if (mode == "broadcast") {
    if (GetIntConst(GetItem(params.src_strides0, -1)) == 0) {
      last_axis = 0;
    }
    if (GetIntConst(GetItem(params.src_strides1, -1)) == 0) {
      last_axis = 1;
    }
  }

  if (body_len > 0) {
    body_args = VectorArgInfo(make_node<VectorArgInfoNode>());
    CHECK(body_args.GetNode() != nullptr);
    body_args.GetNode()->last_axis_info_.src_index_ = last_axis;
    body_args.GetNode()->body_num_ = 1;
    body_args.GetNode()->repeat_ = body_len / params.vec_max_len;
    body_args.GetNode()->dst_stride_m0_ = Expr(1);
    body_args.GetNode()->dst_stride_m1_ = Expr(FULL_BLOCK_NUM);
    body_args.GetNode()->src_stride_m0_list_ = {Expr(1), Expr(1)};
    body_args.GetNode()->src_stride_m1_list_ = {Expr(FULL_BLOCK_NUM), Expr(FULL_BLOCK_NUM)};
    body_args.GetNode()->vec_mask_ = GetVecMask(params.vec_max_len, 1, dst_info->dtype_);
  }
  if (tail_len > 0) {
    tail_args = VectorArgInfo(make_node<VectorArgInfoNode>());
    CHECK(tail_args.GetNode() != nullptr);
    tail_args.GetNode()->last_axis_info_.src_index_ = last_axis;
    tail_args.GetNode()->dst_head_ = Expr(body_len);
    tail_args.GetNode()->src_head_list_ = {Expr(body_len), Expr(body_len)};
    tail_args.GetNode()->repeat_ = Expr(1);
    tail_args.GetNode()->dst_stride_m0_ = Expr(1);
    tail_args.GetNode()->dst_stride_m1_ = Expr(0);
    tail_args.GetNode()->src_stride_m0_list_ = {Expr(1), Expr(1)};
    tail_args.GetNode()->src_stride_m1_list_ = {Expr(0), Expr(0)};
    int data_len = expand_mask ? CeilTo(tail_len, params.block_size) : tail_len;
    tail_args.GetNode()->vec_mask_ = GetVecMask(data_len, 1, dst_info->dtype_);
  }

  // compute offset for cce instructions
  Array<Var> elim_var = {};
  if (mode == "elewise" && params.dst_var.size() >= 2 && params.dst_strides.size() >= 2 &&
      params.last_dim_shape <= params.vec_max_len && for_info.ops_.size() >= 2 &&
      params.last_dim_shape >= params.vec_max_len - params.block_size &&
      GetIntConst(GetItem(params.dst_strides, -2)) == params.last_dim_shape) {
    // in this case we can merge second last for extent to repeat
    size_t idx = 0;
    bool suc = GetIndexOfElement(for_info.vars_, GetItem(params.dst_var, -2), idx);
    CHECK(suc);
    auto latest_for = GetItem(for_info.ops_, idx).as<For>();
    // there should not be if_op between for loop and compute stmt
    if (latest_for && !latest_for->body->IsInstance<IfThenElse>()) {
      if (!params.dst_var.empty() && !is_scalar_mode) {
        if (body_args.defined()) {
          // last_dim_shape = vec_max_len
          body_args.GetNode()->repeat_ = body_args->repeat_ * latest_for->extent;
        } else if (tail_args.defined()) {
          // last_dim_shape < vec_max_len
          tail_args.GetNode()->repeat_ = tail_args->repeat_ * latest_for->extent;
        }
        return elim_var = GetRange(params.dst_var, -2, 2);
      }
    }
  }

  if (!params.dst_var.empty() && !is_scalar_mode) {
    elim_var = GetRange(params.dst_var, -1, 1);
  }

  return elim_var;
}

PatternResult BinaryVecPatternGenerator::GenResult(const Array<Var> &elim_var) {
  arg_info.GetNode()->body_arg_info_ = body_args;
  arg_info.GetNode()->tail_arg_info_ = tail_args;

  auto real_elim_var = elim_var;
  if (!empty_var->name_hint.empty()) {
    bool need_elim = true;
    for (auto e : elim_var) {
      if (e->name_hint == empty_var->name_hint) {
        need_elim = false;
        break;
      }
    }
    if (need_elim) {
      real_elim_var.push_back(empty_var);
    }
  }

  dst_info.GetNode()->insn_offset_ = GetInsnOffset(dst_info, real_elim_var);
  for (auto &info : src_info_list) {
    info.GetNode()->insn_offset_ = GetInsnOffset(info, real_elim_var);
  }

  CleanForInfoVars(for_info, real_elim_var);
  CleanZeroStrides(dst_info);
  CleanZeroStrides(src_info_list);

  if (mode == "elewise") {
    arg_info.GetNode()->arg_type_ = ARG_VECTOR_ELEWISE;
  } else if (mode == "broadcast") {
    arg_info.GetNode()->arg_type_ = ARG_VECTOR_BROADCAST;
  } else if (mode == "reduction") {
    arg_info.GetNode()->arg_type_ = ARG_VECTOR_REDUCTION;
  }

  PatternResult result;
  result.dst_info_list = {dst_info};
  result.src_info_list = src_info_list;
  result.for_info = for_info;
  result.arg_info = arg_info;

  return result;
}

void BinaryVecPatternGenerator::CalcParams() {
  CHECK_GE(src_info_list.size(), 2);
  StmtStoreInfo src_info0 = src_info_list[0];
  StmtStoreInfo src_info1 = src_info_list[1];

  StmtInfoList info_list = {dst_info, src_info0, src_info1};

  // check shape len
  for (auto info : info_list) {
    if (info->shape_.empty()) {
      LOG(FATAL) << "CCE Vector Insn Error: dst_buffer and src_buffer can not be scalar, should keep len(shape) > 0.";
    }
  }

  // check data type
  for (auto src_info : src_info_list) {
    if (dst_info->dtype_ != src_info->dtype_) {
      LOG(FATAL) << "CCE Vector Insn Error: dst_buffer and src_buffer can not be different data type.";
    }
  }

  params.last_dim_shape = GetInt32Const(GetItem(dst_info->shape_, -1));
  AppendEmptyVar(info_list);
  if (arg_info->arg_type_ == ARG_VECTOR_BROADCAST_LAST_AXIS) {
    return;
  }

  if (mode == "reduction" || mode == "broadcast") {
    FillEmptyVar(info_list);
  }
  CHECK_EQ(info_list.size(), 3);
  dst_info = info_list[0];
  src_info0 = info_list[1];
  src_info1 = info_list[2];

  params.vec_max_len = GetVecMaxLen(dst_info->dtype_);
  params.block_size = GetUbBlkSize(dst_info->dtype_);
  CHECK_NE(params.vec_max_len, 0);
  CHECK_NE(params.block_size, 0);

  params.dst_var = dst_info->var_;
  params.dst_shape = dst_info->shape_;
  params.dst_strides = dst_info->strides_;
  params.src_var0 = src_info0->var_;
  params.src_var1 = src_info1->var_;
  params.src_shape0 = src_info0->shape_;
  params.src_shape1 = src_info1->shape_;
  params.src_strides0 = src_info0->strides_;
  params.src_strides1 = src_info1->strides_;

  auto GetNonZeroShapeByIdx = [this](int index) -> int {
    if (index <= static_cast<int>(params.dst_var.size())) {
      if (Equal(GetItem(params.dst_var, -index), GetItem(params.src_var0, -index)) &&
          Equal(GetItem(params.dst_var, -index), GetItem(params.src_var1, -index))) {
        return GetNonZeroShape(GetItem(params.dst_shape, -index), GetItem(params.src_shape0, -index),
                               GetItem(params.src_shape1, -index));
      }
    }
    return 1;
  };

  params.non_zero_shape1 = GetNonZeroShapeByIdx(1);
  params.non_zero_shape2 = GetNonZeroShapeByIdx(2);
  params.non_zero_shape3 = GetNonZeroShapeByIdx(3);
  params.all_points = params.non_zero_shape1 * params.non_zero_shape2 * params.non_zero_shape3;
}

bool BinaryVecPatternGenerator::IsSamePatternComInfo(const StmtStoreInfo &info_a, const StmtStoreInfo &info_b) {
  if (IsSame(info_a->var_, info_b->var_)) {
    if (info_a->shape_.size() != info_b->shape_.size()) {
      return false;
    }
    for (size_t i = 0; i < info_a->shape_.size(); ++i) {
      if (!IsTwoItemEqual(info_a->shape_, info_b->shape_, static_cast<int>(i), true)) {
        return false;
      }
    }
    if (info_a->strides_.size() != info_b->strides_.size()) {
      return false;
    }
    for (size_t i = 0; i < info_a->strides_.size(); ++i) {
      if (!IsTwoItemEqual(info_a->strides_, info_b->strides_, static_cast<int>(i), true)) {
        return false;
      }
    }
    return true;
  }
  return false;
}

bool BinaryVecPatternGenerator::IsNonZeroShapeEqual(const Array<Expr> &shape_list) {
  Array<Expr> non_zero_list;
  for (auto shape : shape_list) {
    if (GetIntConst(shape) != 0) {
      non_zero_list.push_back(shape);
    }
  }
  if (non_zero_list.empty()) {
    LOG(FATAL) << "Error: all shapes are equal to 0.";
  }
  for (auto shape : non_zero_list) {
    if (GetIntConst(shape) != GetIntConst(non_zero_list[0])) {
      return false;
    }
  }
  return true;
}

void BinaryVecPatternGenerator::AppendEmptyVar(StmtInfoList &info_list) {
  auto FillEmptyVarToLast = [](const StmtStoreInfo com_info, const Var &var) -> void {
    com_info.GetNode()->var_.push_back(var);
    com_info.GetNode()->shape_.push_back(Expr(1));
    com_info.GetNode()->strides_.push_back(Expr(1));
    com_info.GetNode()->index_ = com_info->index_ + GetItem(com_info->var_, -1);
  };

  auto src_info0 = src_info_list[0];
  auto src_info1 = src_info_list[1];

  if (mode == "reduction" || mode == "broadcast") {
    // ISA 8.1.2, strides of Xd must be equal to Xm, [Xd = dst, Xn = src0, Xm = src1]
    if (IsSamePatternComInfo(dst_info, src_info0)) {
      auto tmp = src_info0;
      src_info0 = src_info1;
      src_info1 = tmp;
    }

    if (mode == "reduction") {
      if (src_info0->data_alignment_ == 1) {
        empty_var = Var("empty_cc");
        FillEmptyVarToLast(src_info0, empty_var);
      }
    } else if (mode == "broadcast") {
      // last dim broadcast, should use VS insn, such as vadds and vmuls
      bool less_var =
        !dst_info->var_.empty() && !src_info0->var_.empty() && !src_info1->var_.empty() &&
        (!IsTwoItemEqual(dst_info->var_, src_info0->var_, -1) || !IsTwoItemEqual(dst_info->var_, src_info1->var_, -1));
      bool null_var = src_info0->var_.empty() || src_info1->var_.empty();
      if (less_var || null_var) {
        arg_info.GetNode()->arg_type_ = ARG_VECTOR_BROADCAST_LAST_AXIS;
        return;
      } else if (dst_info->data_alignment_ == 1 && src_info0->data_alignment_ == 1) {
        empty_var = Var("empty_cc");
        FillEmptyVarToLast(dst_info, empty_var);
        FillEmptyVarToLast(src_info0, empty_var);
        FillEmptyVarToLast(src_info1, empty_var);
        params.last_dim_shape = 1;
      }
    }
    src_info_list = {src_info0, src_info1};
    info_list = {dst_info, src_info0, src_info1};
  }
}

/// Get CCE Binary Vector Insn Computation Info
/// \param stmt         -  operand stmt
/// \param intrin_name   -  vector intrin name
/// \param dst_info_list  -  dst computation info list
/// \param src_info_list  -  src computation info list
/// \param if_info       -  if info list
/// \param for_info      -  for info list
/// \return intrin args
ArgInfo GetBinaryVecInsnArgs(const Stmt &stmt, std::string intrin_name, StmtInfoList &dst_info_list,
                             StmtInfoList &src_info_list, StmtInfo &if_info, StmtInfo &for_info, bool enable_bisect) {
  // check intrin_name
  std::set<std::string> intrin_name_list = {"vadd", "vmax",  "vmin",   "vmul",   "vdiv",  "vsel",      "vsub", "vand",
                                            "vor",  "vaxpy", "argmax", "argmin", "vmadd", "vmaddrelu", "vmla"};
  if (intrin_name_list.count(intrin_name) == 0) {
    LOG(FATAL) << "Error: CCE Binary Vector Insn doesn't support the given intrin_name.";
  }

  // get and check dst and src
  GetCompactComputationInfo(stmt, dst_info_list, src_info_list, if_info, for_info, true);
  // For vmadd/vmaddrelu/vmla we only need first two src
  if (dst_info_list.size() != 1 || src_info_list.size() < 2) {
    LOG(FATAL) << "CCE Binary Vector Insn only support ONE dst and TWO srcs.";
  }
  src_info_list = GetRange(src_info_list, 0, 2);
  ArgInfo arg_info = ArgInfo(make_node<ArgInfoNode>());

  // detect vector op mode
  std::string mode = GetBinaryVecMode(dst_info_list, src_info_list, intrin_name, enable_bisect);
  if (mode == "reduce_last_axis") {
    size_t src_var_list_size = src_info_list[1]->var_.size();
    if (src_info_list[0]->var_.size() > src_info_list[1]->var_.size()) {
      src_var_list_size = src_info_list[0]->var_.size();
    }

    CHECK(src_var_list_size > 0) << "Error: src can not be a scalar.";
    if (src_var_list_size - dst_info_list[0]->var_.size() == 1) {
      arg_info.GetNode()->arg_type_ = ARG_VECTOR_REDUCTION_LAST_AXIS;
    } else {
      LOG(FATAL) << "Error: cannot support multi-last-axis reduction.";
    }
  } else if (mode == "reduce_bisection") {
    arg_info.GetNode()->arg_type_ = ARG_VECTOR_REDUCTION_BISECTION;
  } else {
    if (mode != "reduction" && mode != "broadcast") {
      FillLastDim(dst_info_list, src_info_list, for_info);
    }

    // vmax/vmin can't expand mask because it may introduce dirty data
    bool can_expand_mask = intrin_name != "vmax" && intrin_name != "vmin";
    BinaryVecPatternGenerator generator =
      BinaryVecPatternGenerator(dst_info_list, src_info_list, for_info, mode, can_expand_mask);
    auto params = generator.GetInsnArgs();
    arg_info = params.arg_info;
    dst_info_list = params.dst_info_list;
    src_info_list = params.src_info_list;
    for_info = params.for_info;
    if (mode == "broadcast") {
      bool has_last_axis = false;
      if ((arg_info->body_arg_info_.defined() && arg_info->body_arg_info_->last_axis_info_.src_index_ != -1) ||
          (arg_info->tail_arg_info_.defined() && arg_info->tail_arg_info_->last_axis_info_.src_index_ != -1)) {
        has_last_axis = true;
      }

      if (has_last_axis && (intrin_name == "vadd" || intrin_name == "vmul")) {
        Array<NodeRef> stores;
        Array<NodeRef> loads;
        GetStoreAndLoads(stmt, stores, loads);
        intrin_name = intrin_name + "s";
        if (arg_info->body_arg_info_.defined()) {
          arg_info.GetNode()->body_arg_info_.GetNode()->last_axis_info_.intrin_name_ = intrin_name;
          arg_info.GetNode()->body_arg_info_.GetNode()->last_axis_info_.src_op_ =
            Downcast<Expr>(loads[arg_info->body_arg_info_->last_axis_info_.src_index_]);
        }
      }
    }
  }

  return arg_info;
}

/// Replace com_info's var with new for loop's var
/// \param info
/// \param old_for_info
/// \param new_for_info
void ReplaceVarWithNewForInfo(StmtStoreInfo &info, const StmtInfo &old_for_info, const StmtInfo &new_for_info) {
  for (size_t i = 0; i < new_for_info.vars_.size(); ++i) {
    for (size_t j = 0; j < info->var_.size(); ++j) {
      if (info->var_[j]->name_hint == new_for_info.vars_[i]->name_hint) {
        SetItem(info.GetNode()->var_, static_cast<int>(j), new_for_info.vars_[i]);
      }
    }
    info.GetNode()->index_ = substitute(old_for_info.vars_[i], new_for_info.vars_[i], info->index_);
  }
}

/// Generete info list for bisection intrin
/// \param dst_info_list
/// \param src_info_list
/// \param for_info
/// \param if_info
/// \param last_axis
/// \param postfix
/// \return
BisectionInfoWrapper SeparateComInfoToBisectionInfoList(const StmtInfoList &dst_info_list,
                                                        const StmtInfoList &src_info_list, const StmtInfo &for_info,
                                                        StmtInfo &if_info, bool last_axis, int postfix = 0) {
  CHECK_EQ(dst_info_list.size(), 1);
  CHECK_EQ(src_info_list.size(), 2);

  BisectionInfoWrapper wrapper;
  // Separate com_info and for_info
  int compare_idx = 1;
  int var_idx = -1;
  if (last_axis) {
    compare_idx = GetLastAxisReductionIdx(dst_info_list, src_info_list);
  } else {
    var_idx = GetBisectionReductionIdx(dst_info_list, src_info_list, compare_idx);
  }
  StmtStoreInfo dst_info = dst_info_list[0];
  CHECK_GE(compare_idx, 0);
  StmtStoreInfo src_info1 = src_info_list[compare_idx];

  Var reduce_var = GetItem(src_info1->var_, var_idx);
  size_t for_idx = 0;
  bool suc = GetIndexOfElement(for_info.vars_, VarExpr(reduce_var), for_idx);
  CHECK(suc);
  auto exist_for = GetItem(for_info.ops_, for_idx).as<For>();
  CHECK(exist_for);
  int extent = GetInt32Const(exist_for->extent);

  std::string prev_name = src_info1->name_;
  Var prev_var = src_info1->data_;
  Buffer prev_buffer = src_info1->buffer_;
  Var bisec_var;
  Buffer bisec_buffer;
  std::string bisec_pre_header = last_axis ? "bisec_last_axis" : "bisec";
  std::string bisec_name = bisec_pre_header + "_local_UB";
  if (postfix > 0) {
    bisec_name = bisec_name + "_" + std::to_string(postfix);
  }
  bool first_round = true;

  int vec_max_len = GetVecMaxLen(dst_info->dtype_);
  int remain_extent = extent;
  int left_extent = 0;

  CHECK_NE(vec_max_len, 0);
  std::vector<int> pow2_list = {0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
  while (extent > 0) {
    int for_extent;
    if (last_axis) {
      left_extent = remain_extent / 2 + remain_extent % 2;
      for (int i : pow2_list) {
        if (left_extent == i) {
          break;
        } else if (left_extent < i) {
          left_extent = i;
          break;
        }
      }
      if (left_extent < vec_max_len) {
        // When left_extent < vec_max_len, stop bisect and generate normal reduce intrin
        left_extent = remain_extent;
      }
      extent = remain_extent - left_extent;
      remain_extent = left_extent;
      for_extent = extent == 0 ? vec_max_len : extent;
    } else {
      for_extent = extent == 1 ? extent : extent / 2;
      extent = extent % 2 == 0 || extent == 1 ? extent / 2 : (extent + 1) / 2;

      for (int i : pow2_list) {
        if (extent == i) {
          break;
        } else if (extent < i) {
          int gap = i - extent;
          extent = i;
          for_extent -= gap;
          break;
        }
      }
    }

    StmtStoreInfo dst_tmp_info = dst_info.Copy();
    StmtStoreInfo src_tmp_info0{src_info1.Copy()};
    StmtStoreInfo src_tmp_info1{src_info1.Copy()};

    if (first_round) {
      auto shape = src_tmp_info1->shape_;
      if (last_axis) {
        int block_size = GetUbBlkSize(dst_info->dtype_);
        SetItem(shape, -1, Expr(CeilTo(GetIntConst(GetItem(shape, -1)), block_size)));
      }
      wrapper.original_shape_ = shape;
      bisec_var = Var(bisec_name, Handle());
      bisec_buffer = BufferNode::make(bisec_var, dst_tmp_info->dtype_, shape, Array<Expr>(), Expr(), bisec_name,
                                      SCOPE_UBUF, 0, 0, BufferType::kDefault);

      if ((last_axis && extent != left_extent) || (!last_axis && extent != for_extent)) {
        // Need to copy input to bisect buffer
        StmtStoreInfo copy_dst_info{src_info1.Copy()};
        StmtStoreInfo copy_src_info{src_info1.Copy()};
        StmtInfoList src_list = {copy_src_info};

        auto for_tmp_info = for_info.Copy();
        auto new_for = GetItem(for_tmp_info.ops_, for_idx).as<For>();
        CHECK(new_for);
        SetItem(for_tmp_info.ops_, static_cast<int>(for_idx),
                For::make(new_for->loop_var, new_for->min, last_axis ? left_extent : extent, new_for->for_type,
                          new_for->device_api, new_for->body));

        ReplaceVarWithNewForInfo(copy_dst_info, for_info, for_tmp_info);
        ReplaceVarWithNewForInfo(copy_src_info, for_info, for_tmp_info);

        SetItem(copy_dst_info.GetNode()->shape_, var_idx, Expr(last_axis ? left_extent : extent));
        SetItem(copy_src_info.GetNode()->shape_, var_idx, Expr(last_axis ? left_extent : extent));

        CompactComputationInfoList(copy_dst_info, src_list, if_info, for_tmp_info);

        copy_dst_info.GetNode()->name_ = bisec_name;
        copy_dst_info.GetNode()->buffer_ = bisec_buffer;
        copy_dst_info.GetNode()->data_ = bisec_var;
        // Replace outside for variable in index
        auto vars = GetVarsInExpr(copy_dst_info->index_);
        for (auto var : vars) {
          if (!IsInArray(copy_dst_info->var_, var)) {
            copy_dst_info.GetNode()->index_ = substitute(var, Expr(0), copy_dst_info->index_);
          }
        }
        wrapper.bisec_info_list_.emplace_back(StmtInfoList{copy_dst_info, copy_src_info});
        wrapper.for_info_list_.push_back(for_tmp_info);
      }
    }

    auto for_tmp_info = for_info.Copy();
    auto new_for = GetItem(for_tmp_info.ops_, for_idx).as<For>();
    CHECK(new_for);
    SetItem(
      for_tmp_info.ops_, static_cast<int>(for_idx),
      For::make(new_for->loop_var, new_for->min, for_extent, new_for->for_type, new_for->device_api, new_for->body));

    ReplaceVarWithNewForInfo(dst_tmp_info, for_info, for_tmp_info);
    ReplaceVarWithNewForInfo(src_tmp_info0, for_info, for_tmp_info);
    ReplaceVarWithNewForInfo(src_tmp_info1, for_info, for_tmp_info);

    SetItem(src_tmp_info0.GetNode()->shape_, var_idx, Expr(for_extent));
    SetItem(src_tmp_info1.GetNode()->shape_, var_idx, Expr(for_extent));

    if (extent > 0) {
      dst_tmp_info.GetNode()->shape_ = src_tmp_info1->shape_;
      dst_tmp_info.GetNode()->strides_ = src_tmp_info1->strides_;
      dst_tmp_info.GetNode()->var_ = src_tmp_info1->var_;
      dst_tmp_info.GetNode()->index_ = src_tmp_info1->index_;
      dst_tmp_info.GetNode()->data_alignment_ = src_tmp_info1->data_alignment_;
      dst_tmp_info.GetNode()->name_ = bisec_name;
      dst_tmp_info.GetNode()->buffer_ = bisec_buffer;
      dst_tmp_info.GetNode()->data_ = bisec_var;
      auto src_extent = Expr(left_extent);
      if (!last_axis) {
        src_extent = GetItem(src_tmp_info1->strides_, var_idx) * extent;
      }
      src_tmp_info1.GetNode()->index_ = src_tmp_info1->index_ + src_extent;
    }

    src_tmp_info0.GetNode()->name_ = prev_name;
    src_tmp_info1.GetNode()->name_ = prev_name;
    src_tmp_info0.GetNode()->buffer_ = prev_buffer;
    src_tmp_info1.GetNode()->buffer_ = prev_buffer;
    src_tmp_info0.GetNode()->data_ = prev_var;
    src_tmp_info1.GetNode()->data_ = prev_var;

    // Replace outside for variable in index
    for (auto &info : {dst_tmp_info, src_tmp_info0, src_tmp_info1}) {
      if (info->name_.find(bisec_pre_header) == std::string::npos) {
        continue;
      }
      auto vars = GetVarsInExpr(info->index_);
      for (auto var : vars) {
        if (!IsInArray(info->var_, var)) {
          info.GetNode()->index_ = substitute(var, Expr(0), info->index_);
        }
      }
    }
    prev_name = bisec_name;
    prev_var = bisec_var;
    prev_buffer = bisec_buffer;

    StmtInfoList src_list = {src_tmp_info0, src_tmp_info1};
    CompactComputationInfoList(dst_tmp_info, src_list, if_info, for_tmp_info);
    wrapper.for_info_list_.emplace_back(for_tmp_info);

    if (extent == 0) {
      // last round should be dst = dst + src_tmp1
      wrapper.bisec_info_list_.emplace_back(StmtInfoList{dst_tmp_info, dst_tmp_info, src_tmp_info1});
    } else {
      // normally is dst_tmp = src_tmp0 + src_tmp1
      wrapper.bisec_info_list_.emplace_back(StmtInfoList{dst_tmp_info, src_tmp_info0, src_tmp_info1});
    }

    first_round = false;
  }

  // Generate arg_info
  for (size_t i = 0; i < wrapper.bisec_info_list_.size(); ++i) {
    auto info_list = wrapper.bisec_info_list_[i];
    auto new_for_info = wrapper.for_info_list_[i];

    ArgInfo arg_info;
    auto dst_list = GetRange(info_list, 0, 1);
    auto src_list = GetRange(info_list, 1, info_list.size() - 1);
    if (info_list.size() == 2) {
      std::string dma_intrin = INTRIN_NAME_COPY_UB_TO_UB;
      wrapper.dma_arg_info_map_ = GetDmaCopyInsnArgs(dma_intrin, dst_list, src_list, new_for_info);
    } else if (last_axis && i == wrapper.bisec_info_list_.size() - 1) {
      auto dst_tmp_info = dst_list[0];
      auto src_tmp_info = src_list[1];
      ReduceLastAxisPatternGenerator generator =
        ReduceLastAxisPatternGenerator(dst_tmp_info, src_tmp_info, new_for_info, "vadd");
      auto result = generator.GetInsnArgs();
      arg_info = result.arg_info;
      dst_tmp_info = result.dst_info_list[0];
      src_tmp_info = result.src_info_list[0];
      new_for_info = result.for_info;
      wrapper.bisec_info_list_[i] = {dst_tmp_info, dst_tmp_info, src_tmp_info};
    } else {
      // Bisect can't expand mask because it has inplace operation
      if (i != wrapper.bisec_info_list_.size() - 1) {
        // Last round dont need to add
        FillLastDim(dst_list, src_list, new_for_info);
      }
      std::string mode = GetBinaryVecMode(dst_list, src_list, "vadd", false);
      BinaryVecPatternGenerator generator = BinaryVecPatternGenerator(dst_list, src_list, new_for_info, mode, false);
      auto params = generator.GetInsnArgs();
      arg_info = params.arg_info;
      dst_list = params.dst_info_list;
      src_list = params.src_info_list;
      new_for_info = params.for_info;
      wrapper.bisec_info_list_[i] = {dst_list[0], src_list[0], src_list[1]};
    }
    wrapper.arg_info_list_.push_back(arg_info);
    wrapper.for_info_list_[i] = new_for_info;
  }

  return wrapper;
}
}  // namespace akg
