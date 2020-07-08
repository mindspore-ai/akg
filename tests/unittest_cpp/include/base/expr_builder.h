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
#ifndef UT_BASE_EXPR_BUILDER_H_
#define UT_BASE_EXPR_BUILDER_H_
#include <string>
#include <vector>
#include "tvm/expr.h"
#include "tvm/operation.h"

namespace akg {
class UTExprBuilder {
 public:
  UTExprBuilder() = default;
  ~UTExprBuilder() = default;

  static ktvm::Array<ktvm::Expr> CreateShape(const std::vector<int32_t> &shapes);
  static ktvm::Var CreateVar(const std::string &name);
  static ktvm::Array<ktvm::Expr> CreateVars(const std::vector<std::string> &names);
  static ktvm::Operation PlaceholderOpNode(
      const std::string &name,
      const std::vector<int32_t> &shapes,
      ktvm::DataType dtype = ktvm::Float(16));
  static ktvm::Expr TensorElement(
      const std::string &name,
      const std::vector<int32_t> &shapes,
      const std::vector<std::string> &axis_names,
      ktvm::DataType dtype = ktvm::Float(16));
};  // UTExprBuilder

class UTTensorElementHelper {
 public:
  UTTensorElementHelper(const std::vector<int32_t> &shapes,
                        const std::string &axis_name_prefix = "ax");
  ~UTTensorElementHelper() = default;
  ktvm::Expr Elem(const std::string &name,
                  uint32_t dim,
                  ktvm::DataType dtype = ktvm::Float(16)) const;

 private:
  std::vector<int32_t> shapes_;
  std::string axis_name_prefix_;
  std::vector<std::string> axis_names_;
};  // UTTensorElementHelper

}  // namespace akg
#endif  // UT_BASE_EXPR_BUILDER_H_
