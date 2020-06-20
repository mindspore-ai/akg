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

#ifndef LANG_CGTYPES_H_
#define LANG_CGTYPES_H_

#include <tvm/dtype.h>
#include <tvm/expr.h>
#include <map>

// the memory type
enum MSType {
  MEM_DEFAULT,  // means its a non-root tile
  HM,
  GM,
  SM,
  CBUF,
  CA,
  CB,
  CC,
  UBUF,
  MEM_END
};

enum class CGTypes : unsigned {
  cg_int8,
  cg_uint8,
  cg_uint16,
  cg_int16,
  cg_int32,
  cg_uint32,
  cg_uint64,
  cg_half,
  cg_float,
  cg_double,
  cg_bool,
  cg_none
};

class Lengh {
 public:
  static const unsigned int ZERO = 0;
  static const unsigned int BYTE = 1;
  static const unsigned int HALF = 2;
  static const unsigned int WORD = 4;
  static const unsigned int DOUBLE_WORD = 8;
};

#endif  // LANG_CGTYPES_H_
