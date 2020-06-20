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
#ifndef POLY_STMT_PARSE_H_
#define POLY_STMT_PARSE_H_
#pragma once
#include <tvm/ir.h>
#include <tvm/ir_pass.h>

#include <unordered_map>
#include <vector>
#include <string>

#include "isl.h"
#include "ir_pass.h"
namespace akg {
namespace ir {
namespace poly {
#define FOREACH(X)                \
  X(elewise_single_VS_add)        \
  X(elewise_single_VS_mul)        \
  X(elewise_single_rec)           \
  X(elewise_single_log)           \
  X(elewise_single_exp)           \
  X(elewise_single_sqrt)          \
  X(elewise_single_rsqrt)         \
  X(vec_single_cast)              \
  X(vec_single_floor)             \
  X(vec_single_round)             \
  X(elewise_single_ceil)          \
  X(vec_single_trunc)             \
  X(elewise_single_not)           \
  X(elewise_single_fabs)          \
  X(elewise_single_relu)          \
  X(broadcast)                    \
  X(pandora_select)               \
  X(pandora_cmp)                  \
  X(reg_mov)                      \
  X(mad)                          \
  X(elewise_binary_add)           \
  X(elewise_binary_sub)           \
  X(elewise_binary_mul)           \
  X(elewise_binary_div)           \
  X(elewise_binary_mod)           \
  X(elewise_binary_min)           \
  X(elewise_binary_max)           \
  X(elewise_binary_or)            \
  X(elewise_binary_and)           \
  X(elewise_binary_EQ)            \
  X(elewise_binary_NE)            \
  X(elewise_binary_GT)            \
  X(elewise_binary_GE)            \
  X(elewise_binary_LT)            \
  X(elewise_binary_LE)            \
  X(elewise_binary_scalar_axpy)   \
  X(four2five_nchw)               \
  X(vec_argmax)                   \
  X(elewise_binary_proposal_sort) \
  X(elewise_binary_topk_sort)     \
  X(elewise_binary_nms)           \
  X(with)                         \
  X(vec_argmin)                   \
  X(elewise_binary_dropout)       \
  X(elewise_binary_iou)           \
  X(elewise_binary_unknown)       \
  X(assignment)                   \
  X(im2col)                       \
  X(poly_op_type_max)             \
  X(vmadd)                        \
  X(vmaddrelu)                    \
  X(vaxpy)                        \
  X(vmla)                         \
  X(elewise_binary_bitwise_and)   \
  X(elewise_binary_bitwise_or)    \
  X(elewise_single_bitwise_not)

#define GENERATE_ENUM(ENUM) ENUM,
#define GENERATE_STRING(STRING) #STRING,

enum class PolyOpType : int { FOREACH(GENERATE_ENUM) };

const char *getPolyOpTypeKey(PolyOpType type);

struct StmtOpInfo {
  std::vector<PolyOpType> ops;
  std::vector<isl::id> readtensors;
  bool isCube = false;
  bool isCubeAssign = false;
  bool isWith = false;
  bool isIm2col = false;
  bool isLoad3d = false;
  // only used when isCube/isConv = true;
  std::string A_ = "";
  std::string B_ = "";
  std::string C_ = "";
  ktvm::DataType MadType_ = Float(16);
};

using StmtOpInfoMap = std::unordered_map<isl::id, StmtOpInfo, isl::IslIdIslHash>;
}  // namespace poly
}  // namespace ir
}  // namespace akg
#endif  // POLY_STMT_PARSE_H_
