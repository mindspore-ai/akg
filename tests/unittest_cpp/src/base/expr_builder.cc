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
#include <sstream>
#include "base/expr_builder.h"

namespace akg {
air::Array<air::Expr> UTExprBuilder::CreateShape(const std::vector<int32_t> &shapes) {
  air::Array<air::Expr> res;
  for (int32_t shape : shapes) {
    air::Integer imm = air::IntImm::make(air::Int(32), shape);
    res.push_back(imm);
  }
  return res;
}

air::Var UTExprBuilder::CreateVar(const std::string &name) {
  return air::Var(name);
}

air::Array<air::Expr> UTExprBuilder::CreateVars(const std::vector<std::string> &names) {
  air::Array<air::Expr> vars;
  for (const std::string &name : names) {
    vars.push_back(std::move(CreateVar(name)));
  }
  return vars;
}

air::Operation UTExprBuilder::PlaceholderOpNode(
    const std::string &name,
    const std::vector<int32_t> &shapes,
    air::DataType dtype) {
  air::Array<air::Expr> expr_shapes = CreateShape(shapes);
  return air::PlaceholderOpNode::make(name, expr_shapes, dtype);
}

air::Expr UTExprBuilder::TensorElement(
    const std::string &name,
    const std::vector<int32_t> &shapes,
    const std::vector<std::string> &axis_names,
    air::DataType dtype) {
  return air::ir::Call::make(
      dtype,                                   // type
      name,                                    // name
      CreateVars(axis_names),                  // args
      air::ir::Call::Halide,                  // call_type
      PlaceholderOpNode(name, shapes, dtype),  // func,
      0);                                      // value_index
}

UTTensorElementHelper::UTTensorElementHelper(const std::vector<int32_t> &shapes,
                                             const std::string &axis_name_prefix)
    : shapes_(shapes), axis_name_prefix_(axis_name_prefix) {
  std::stringstream ss;
  for (size_t i = 0; i < shapes_.size(); i++) {
    ss << axis_name_prefix_ << i;
    axis_names_.push_back(ss.str());
    ss.str("");
  }
}

air::Expr UTTensorElementHelper::Elem(const std::string &name,
                                       uint32_t dim,
                                       air::DataType dtype) const {
  uint32_t start = shapes_.size() - dim;
  return UTExprBuilder::TensorElement(
      name,
      std::vector<int32_t>(shapes_.begin() + start, shapes_.end()),
      std::vector<std::string>(axis_names_.begin() + start, axis_names_.end()),
      dtype);
}
}  // namespace akg
