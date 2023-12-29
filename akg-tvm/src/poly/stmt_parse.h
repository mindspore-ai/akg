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

#include "isl.h"

#include "ir_pass.h"
#include "poly_util.h"

namespace akg {
namespace ir {
namespace poly {

struct StmtOpInfo {
  std::vector<PolyOpType> ops;
  std::vector<isl::id> readtensors;
  bool isMMU = false;
  bool isMMUAssign = false;
  bool isWith = false;
  bool isIm2col = false;
  bool is_load_im2col = false;
  // only used when isMMU/isConv = true;
  std::string A_ = "";
  std::string B_ = "";
  std::string C_ = "";
  std::string C_IN_ = "";
  air::DataType MadType_ = Float(16);
};

using StmtOpInfoMap = std::unordered_map<isl::id, StmtOpInfo, isl::IslIdIslHash>;
}  // namespace poly
}  // namespace ir
}  // namespace akg
#endif  // POLY_STMT_PARSE_H_
