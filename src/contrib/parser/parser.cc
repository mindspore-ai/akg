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

#include "contrib/parser/parser.h"

#include <tvm/api_registry.h>
#include <tvm/operation.h>

#include <map>

#include "contrib/parser/token.h"
#include "contrib/parser/grammar.h"
#include "contrib/parser/codegen.h"

namespace akg {
namespace ir {
Stmt ParseHalideIRFromFile(const std::string &file, const Map<Tensor, Buffer> &ori_in) {
  auto stat = GetTokStateFromFile(file);

  return ParseHalideIRFromCode(stat.code, ori_in);
}

Stmt ParseHalideIRFromCode(const std::string &code, const Map<Tensor, Buffer> &ori_in) {
  auto stat = GetTokStateFromCode(code);

  auto ast_list = GenAST(stat);

  return GenHalideIR(ast_list, ori_in);
}

TVM_REGISTER_API("ParseHalideIRFromFile").set_body_typed(ParseHalideIRFromFile);

TVM_REGISTER_API("ParseHalideIRFromCode").set_body_typed(ParseHalideIRFromCode);
}  // namespace ir
}  // namespace akg
