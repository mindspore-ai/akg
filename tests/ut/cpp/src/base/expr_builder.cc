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
#include <tvm/operation.h>
#include "base/expr_builder.h"

namespace akg {
air::Expr UTExprBuilder::IntImm(int64_t value, air::DataType dtype) {
  return air::IntImm::make(dtype, value);
}

air::Expr UTExprBuilder::UIntImm(uint64_t value, air::DataType dtype) {
  return air::ir::UIntImm::make(dtype, value);
}

air::Expr UTExprBuilder::BoolImm(bool value) {
  return air::ir::UIntImm::make(air::Bool(), value ? 1 : 0);
}

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

air::Region UTExprBuilder::CreateRegion(const std::vector<int32_t> &shapes) {
  air::Region region;
  for (int32_t shape : shapes) {
    region.push_back(CreateRange(0, shape));
  }
  return region;
}

air::Region UTExprBuilder::CreateRegion(const air::Array<air::Expr> &shapes) {
  air::Region region;
  for (const air::Expr &shape : shapes) {
    region.push_back(air::Range::make_by_min_extent(IntImm(0), shape));
  }
  return region;
}

air::Range UTExprBuilder::CreateRange(int32_t min, int32_t max) {
  air::Integer imm_min = air::IntImm::make(air::Int(32), min);
  air::Integer imm_max = air::IntImm::make(air::Int(32), max);
  return air::Range(std::move(imm_min), std::move(imm_max));
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
    const air::Array<air::Expr> &axis_vars,
    air::DataType dtype) {
  return air::ir::Call::make(
      dtype,                                   // type
      name,                                    // name
      axis_vars,                               // args
      air::ir::Call::Halide,                   // call_type
      PlaceholderOpNode(name, shapes, dtype),  // func,
      0);                                      // value_index
}

air::Expr UTExprBuilder::ElementOf(
    const air::Operation &op,
    const air::Array<air::Expr> &axis_vars) {
  if (op->template IsInstance<air::PlaceholderOpNode>()) {
    return ElementOfPlaceholderOp(op, axis_vars);
  } else {
    CHECK(false);
    return air::ir::Any::make();
  }
}

air::Expr UTExprBuilder::ElementOfPlaceholderOp(
    const air::Operation &op,
    const air::Array<air::Expr> &axis_vars) {
  const air::PlaceholderOpNode *node = op.as<const air::PlaceholderOpNode>();
  CHECK(node);
  return air::ir::Call::make(
      node->dtype,
      node->name,
      axis_vars,
      air::ir::Call::Halide,
      op,
      0);
}
air::Expr UTExprBuilder::CreateCall(
  const air::ir::FunctionRef func,
  air::Array<air::Expr> args,
  air::ir::Call::CallType call_type,
  int value_index) {
  air::DataType type = air::Float(16);
  const air::OperationNode *node_op = func.as<air::OperationNode>();
  CHECK(node_op);
  std::string name = node_op->name;
  const air::PlaceholderOpNode *node_placeholder = func.as<air::PlaceholderOpNode>();
  if (node_placeholder != nullptr) {
    type = node_placeholder->dtype;
  }
  return air::ir::Call::make(type, name, args, call_type, func, value_index);
}

air::Tensor UTExprBuilder::CreateTensorByPlaceholder(const air::Operation op) {
  const air::PlaceholderOpNode *node = op.as<air::PlaceholderOpNode>();
  CHECK(node);
  return air::TensorNode::make(
      node->shape,
      node->dtype,
      op,
      0);
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
  var_pool_.AddVars(axis_names_);
}

air::Expr UTTensorElementHelper::Elem(const std::string &name,
                                      uint32_t dim,
                                      air::DataType dtype) const {
  uint32_t start = shapes_.size() - dim;
  return UTExprBuilder::TensorElement(
      name,
      std::vector<int32_t>(shapes_.begin() + start, shapes_.end()),
      var_pool_.GetVars(std::vector<std::string>(axis_names_.begin() + start, axis_names_.end())),
      dtype);
}

void UTVariablePool::AddVar(const std::string &name) {
  auto it = map_name_var_.find(name);
  if (it != map_name_var_.end()) {
    std::cerr << "Variable " << name << " has been defined" << std::endl;
    return;
  }
  map_name_var_.insert(std::make_pair(name, UTExprBuilder::CreateVar(name)));
}

void UTVariablePool::AddVars(const std::vector<std::string> &names) {
  for (const std::string &name : names) {
    AddVar(name);
  }
}

air::Var UTVariablePool::GetVar(const std::string &name) const {
  auto it = map_name_var_.find(name);
  CHECK(it != map_name_var_.end());
  return it->second;
}

air::Array<air::Expr> UTVariablePool::GetVars(const std::vector<std::string> &names) const {
  air::Array<air::Expr> vars;
  for (const std::string &name : names) {
    vars.push_back(GetVar(name));
  }
  return vars;
}

void UTVariablePool::Reset() {
  map_name_var_.clear();
}
}  // namespace akg
