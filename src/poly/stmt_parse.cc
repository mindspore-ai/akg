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
#include "poly/stmt_parse.h"

#include <stdlib.h>
#include <fstream>
#include <cstdlib>
#include <unordered_map>
#include <iostream>

namespace akg {
namespace ir {
namespace poly {
static const char *PolyOpTypeKey[] = {FOREACH(GENERATE_STRING)};

const char *getPolyOpTypeKey(PolyOpType type) {
  int idx = static_cast<int>(type);
  const int num_type_keys = sizeof(PolyOpTypeKey) / sizeof(PolyOpTypeKey[0]);
  CHECK(idx < num_type_keys) << "invalid type " << idx;
  return PolyOpTypeKey[idx];
}
}  // namespace poly
}  // namespace ir
}  // namespace akg
