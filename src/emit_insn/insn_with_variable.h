/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef INSN_WITH_VARIABLE_H
#define INSN_WITH_VARIABLE_H

#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <limits.h>

#include <string>
#include <vector>
#include <set>
#include <unordered_set>

#include "tvm.h"
#include "ir_pass.h"

namespace akg {
namespace ir {

const int64_t FullReduceMaskValue = 6148914691236517205;

class CCEInsn {
 public:
  std::string insn_name;
  Var dst;
  Var src;
};

class CCEInfo {
 public:
  CCEInfo(Var dst, Expr dst_index, Array<Var> src, Array<Expr> src_index, Type type)
      : dst(dst), dst_index(dst_index), src(src), src_index(src_index), type(type) {}

  CCEInfo() = default;
  ~CCEInfo() = default;

  Var dst;
  Expr dst_index;
  Array<Var> src;
  Array<Expr> src_index;
  Type type;
  Stmt ori_stmt;
  Expr imm;
  std::vector<Var> loops_vars_;
  std::vector<Expr> loops_extent_;
  std::vector<Type> src_type;
};

class SelectInfo {
 public:
  Array<Var> tensor_var;
  Array<Expr> tensor_index;
  std::vector<Type> data_type;
  std::vector<int> offset_factor;
};

static const std::unordered_map<std::string, std::pair<std::string, std::string>> SIMDInsnMap = {
  // binary
  {"vec_binary_add", std::make_pair("vadd", "binary")},
  {"vec_binary_sub", std::make_pair("vsub", "binary")},
  {"vec_binary_mul", std::make_pair("vmul", "binary")},
  {"vec_binary_min", std::make_pair("vmin", "binary")},
  {"vec_binary_max", std::make_pair("vmax", "binary")},
  {"vec_binary_div", std::make_pair("vdiv", "binary")},
  {"vec_binary_and", std::make_pair("vand", "binary")},
  {"vec_binary_or", std::make_pair("vor", "binary")},
  {"vec_binary_vmadd", std::make_pair("vmadd", "binary")},
  {"vec_binary_vmaddrelu", std::make_pair("vmaddrelu", "binary")},
  {"vec_binary_vmla", std::make_pair("vmla", "binary")},

  // single
  {"vec_single_fabs", std::make_pair("vabs", "single")},
  {"vec_single_log", std::make_pair("vln", "single")},
  {"vec_single_exp", std::make_pair("vexp", "single")},
  {"vec_single_rec", std::make_pair("vrec", "single")},
  {"vec_single_not", std::make_pair("vnot", "single")},
  {"vec_single_sqrt", std::make_pair("vsqrt", "single")},
  {"vec_single_rsqrt", std::make_pair("vrsqrt", "single")},
  {"vec_single_relu", std::make_pair("vrelu", "single")},
  {"vec_single_not", std::make_pair("vnot", "single")},

  // vector_scalar
  {"vec_single_muls", std::make_pair("vmuls", "vector_scalar")},
  {"vec_single_adds", std::make_pair("vadds", "vector_scalar")},
  {"vec_binary_axpy", std::make_pair("vaxpy", "vector_scalar")},

  // Mov
  {"broadcast", std::make_pair("vector_dup", "vector_dup")},

  // vector_cast
  {"vec_single_cast", std::make_pair("", "cast")},
  {"vec_single_floor", std::make_pair("f", "cast")},
  {"vec_single_round", std::make_pair("r", "cast")},
  {"vec_single_ceil", std::make_pair("c", "cast")},
  {"vec_single_trunc", std::make_pair("z", "cast")},
};
}  // namespace ir
}  // namespace akg

#endif  // INSN_WITH_VARIABLE_H
