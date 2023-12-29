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

#ifndef CONTRIB_PARSER_PARSER_H_
#define CONTRIB_PARSER_PARSER_H_

#include <tvm/expr.h>
#include <tvm/node/container.h>
#include <tvm/tensor.h>
#include <tvm/buffer.h>
#include <tvm.h>

#include <string>

namespace akg {
namespace ir {
Stmt ParseHalideIRFromFile(const std::string &file, const Map<Tensor, Buffer> &ori_in);

Stmt ParseHalideIRFromCode(const std::string &code, const Map<Tensor, Buffer> &ori_in);
}  // namespace ir
}  // namespace akg

#endif  // CONTRIB_PARSER_PARSER_H_
