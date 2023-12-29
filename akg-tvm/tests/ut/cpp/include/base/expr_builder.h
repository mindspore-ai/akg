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
#include "tvm/tensor.h"

namespace akg {
class UTExprBuilder {
 public:
  UTExprBuilder() = default;
  ~UTExprBuilder() = default;

  static air::Expr IntImm(int64_t value, air::DataType dtype = air::Int(32));
  static air::Expr UIntImm(uint64_t value, air::DataType dtype = air::UInt(32));
  static air::Expr BoolImm(bool value);
  static air::Array<air::Expr> CreateShape(const std::vector<int32_t> &shapes);
  static air::Var CreateVar(const std::string &name);
  static air::Array<air::Expr> CreateVars(const std::vector<std::string> &names);
  static air::Range CreateRange(int32_t min, int32_t max);
  static air::Region CreateRegion(const std::vector<int32_t> &shapes);
  static air::Region CreateRegion(const air::Array<air::Expr> &shapes);
  static air::Operation PlaceholderOpNode(
      const std::string &name,
      const std::vector<int32_t> &shapes,
      air::DataType dtype = air::Float(16));
  static air::Expr TensorElement(
      const std::string &name,
      const std::vector<int32_t> &shapes,
      const air::Array<air::Expr> &axis_vars,
      air::DataType dtype = air::Float(16));
  static air::Expr ElementOf(
      const air::Operation &op,
      const air::Array<air::Expr> &axis_vars);
  static air::Expr ElementOfPlaceholderOp(
      const air::Operation &op,
      const air::Array<air::Expr> &axis_vars);
  static air::Expr CreateCall(
      const air::ir::FunctionRef func,
      air::Array<air::Expr> args,
      air::ir::Call::CallType call_type = air::ir::Call::Halide,
      int value_index = 0);
  static air::Tensor CreateTensorByPlaceholder(const air::Operation op);
};  // UTExprBuilder

class UTVariablePool {
 public:
  UTVariablePool() = default;
  ~UTVariablePool() = default;
  void AddVar(const std::string &name);
  void AddVars(const std::vector<std::string> &names);
  air::Var GetVar(const std::string &name) const;
  air::Array<air::Expr> GetVars(const std::vector<std::string> &names) const;
  void Reset();

 private:
  std::map<std::string, air::Var> map_name_var_;
};  // class UTVariablePool

class UTTensorElementHelper {
 public:
  UTTensorElementHelper(const std::vector<int32_t> &shapes,
                        const std::string &axis_name_prefix = "ax");
  ~UTTensorElementHelper() = default;
  air::Expr Elem(const std::string &name,
                 uint32_t dim,
                 air::DataType dtype = air::Float(16)) const;
  const UTVariablePool &GetVarPool() const {
    return var_pool_;
  }

 private:
  std::vector<int32_t> shapes_;
  std::string axis_name_prefix_;
  std::vector<std::string> axis_names_;
  UTVariablePool var_pool_;
};  // class UTTensorElementHelper
}  // namespace akg
#endif  // UT_BASE_EXPR_BUILDER_H_
