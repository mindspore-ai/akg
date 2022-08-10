/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "poly/tiling/hermes/check_visitor.h"
#include "pass/utils.h"

namespace akg {
namespace ir {
namespace poly {
std::vector<std::shared_ptr<Node>> CheckVisitor::nodes_;

void CheckVisitor::Clear() { nodes_.clear(); }

void CheckVisitor::PrintBuildGraphInfo() {
  std::stringstream build_graph_stream;
  build_graph_stream << "build Graph info ----------------------" << std::endl;
  for (auto const &node : nodes_) {
    build_graph_stream << node->name_ << std::endl;
    for (auto const &tos : node->transformed_output_shape_) {
      for (auto const &shape : tos.shape_) {
        build_graph_stream << shape << std::endl;
      }
    }
    for (auto const &iter : node->axis_to_tensor_to_shape_id_map_) {
      for (auto const &l_iter : iter.second) {
        build_graph_stream << iter.first << " map to " << l_iter.second << std::endl;
      }
    }
  }
  LOG(INFO) << build_graph_stream.str();
}

void CheckVisitor::Visit_(const Provide *op) {
  bool exist = false;
  std::string node_name;
  for (auto const &node : nodes_) {
    if (node->name_ == op->func->func_name()) {
      node_name = GetNewNodeName(node, op->args);
      if (node_name.empty()) {
        exist = true;
        cur_node_ = node;
      }
      break;
    }
  }
  if (!exist) {
    std::shared_ptr<Tensor> tensor_of_provide;
    tensor_of_provide = GetTensor(op->func, op->value_index);

    std::shared_ptr<Node> provide_node = std::make_shared<Node>();
    if (node_name.empty()) {
      provide_node->name_ = op->func->func_name();
    } else {
      provide_node->name_ = node_name;
    }
    provide_node->transformed_output_shape_.push_back(*tensor_of_provide);
    if (op->args.size() > 1 || op->args[0].as<Variable>() != nullptr) {
      for (auto const &arg : op->args) {
        std::vector<std::string> vars{};
        if (arg.as<Variable>() == nullptr) {
          vars = GetVarNamesFromExpr(arg);
          if (vars.empty()) {
            continue;
          }
        } else {
          vars.push_back(arg.as<Variable>()->name_hint);
        }
        for (auto const &var : vars) {
          auto iter = realize_node_.axis_to_tensor_to_shape_id_map_.find(var);
          auto res = iter->second.begin();
          int64_t range = res->second;

          Axis axis_of_provide;
          axis_of_provide.name_ = var;
          axis_of_provide.range_ = range;
          provide_node->axis_of_node_.push_back(axis_of_provide);

          std::map<Tensor, int64_t> tensor_to_range_map = {{*tensor_of_provide, range}};
          provide_node->axis_to_tensor_to_shape_id_map_.insert(
            std::make_pair(axis_of_provide.name_, tensor_to_range_map));
        }
        vars.clear();
      }
    }
    provide_node->op_.op_type_ = Op::OpType::Assignment;
    provide_node->output_tensors_.push_back(tensor_of_provide);
    nodes_.push_back(provide_node);
    cur_node_ = nodes_.back();
  }

  bool is_reduce = false;
  if (!loop_var_.empty()) {
    is_reduce = true;
    for (auto const &arg : op->args) {
      if (arg.as<Variable>() != nullptr && arg.as<Variable>()->name_hint == loop_var_) {
        is_reduce = false;
      }
    }
  }
  if (is_reduce) {
    auto op_reduce_type = GetOpReduceType(op->value);
    if (op_reduce_type != AKG_REDUCE_UNSUPPORTED) {
      if (op_reduce_type == AKG_REDUCE_SUM) {
        SetOperatorType(Op::OpType::ReduceSum);
      } else if (op_reduce_type == AKG_REDUCE_MAX) {
        SetOperatorType(Op::OpType::ReduceMax);
      } else if (op_reduce_type == AKG_REDUCE_MIN) {
        SetOperatorType(Op::OpType::ReduceMin);
      } else if (op_reduce_type == AKG_REDUCE_PROD) {
        SetOperatorType(Op::OpType::ReduceProd);
      } else {
        LOG(FATAL) << "Symbolic Tiling's parser does not support this reduce type (" << op_reduce_type << ").";
      }
    }
  }

  IRVisitor::Visit_(op);
}

void CheckVisitor::Visit_(const For *op) {
  Axis axis_for = Axis();
  int64_t range = 0;
  if (op->extent.as<IntImm>() != nullptr) {
    range = static_cast<int64_t>(op->extent.as<IntImm>()->value);
  }
  std::string axis_for_name = op->loop_var->name_hint;
  // check if the axis exist
  merge_map_.insert(std::make_pair(range, axis_for_name));
  axis_for.name_ = axis_for_name;
  std::map<Tensor, int64_t> tensor_to_range_map = {{realize_node_.transformed_output_shape_[0], range}};
  realize_node_.axis_to_tensor_to_shape_id_map_.insert(std::make_pair(axis_for.name_, tensor_to_range_map));
  realize_node_.axis_of_node_.push_back(axis_for);
  loop_var_ = op->loop_var->name_hint;

  IRVisitor::Visit_(op);
  loop_var_.clear();
}

void CheckVisitor::Visit_(const Call *op) {
  DefineCallOpType(op);
  if (cur_node_ && op->func) {
    if (op->name != cur_node_->name_) {
      std::shared_ptr<Tensor> tensor_of_input;
      bool exist = false;
      for (auto const &node : nodes_) {
        if (node->name_ == op->func->func_name()) {
          exist = true;
          UpdateCurrentNode(op->func, op->value_index, node);
          node->succ_.push_back(cur_node_);
          break;
        }
      }
      if (!exist) {
        SetInputNode(op);
      }
    }
  }
  IRVisitor::Visit_(op);
}

void CheckVisitor::Visit_(const Realize *op) {
  realize_node_ = Node();
  Tensor tensor_of_node;
  for (auto bound : op->bounds) {
    CHECK(bound->extent.as<IntImm>());
    tensor_of_node.shape_.push_back(static_cast<int64_t>(bound->extent.as<IntImm>()->value));
  }
  realize_dtype_ = GetDatatypeString(op->type);
  tensor_of_node.datatype_ = DataTypeFromString(realize_dtype_);
  realize_node_.transformed_output_shape_.push_back(tensor_of_node);
  IRVisitor::Visit_(op);
}

void CheckVisitor::Visit_(const Add *op) {
  SetOperatorType(Op::OpType::Add);
  IRVisitor::Visit_(op);
}

void CheckVisitor::Visit_(const Sub *op) {
  SetOperatorType(Op::OpType::Sub);
  IRVisitor::Visit_(op);
}

void CheckVisitor::Visit_(const Mul *op) {
  SetOperatorType(Op::OpType::Mul);
  IRVisitor::Visit_(op);
}

void CheckVisitor::Visit_(const Div *op) {
  SetOperatorType(Op::OpType::RealDiv);
  IRVisitor::Visit_(op);
}

void CheckVisitor::Visit_(const FloorDiv *op) {
  SetOperatorType(Op::OpType::RealDiv);
  IRVisitor::Visit_(op);
}

void CheckVisitor::Visit_(const Min *op) {
  SetOperatorType(Op::OpType::Minimum);
  IRVisitor::Visit_(op);
}

void CheckVisitor::Visit_(const Max *op) {
  SetOperatorType(Op::OpType::Maximum);
  IRVisitor::Visit_(op);
}

void CheckVisitor::Visit_(const EQ *op) {
  SetOperatorType(Op::OpType::Equal);
  IRVisitor::Visit_(op);
}

void CheckVisitor::Visit_(const NE *op) {
  SetOperatorType(Op::OpType::Neg);
  IRVisitor::Visit_(op);
}

void CheckVisitor::Visit_(const Cast *op) {
  dtype_change_.first = true;
  dtype_change_.second = GetDatatypeString(op->type);
  SetOperatorType(Op::OpType::Cast);
  IRVisitor::Visit_(op);
}

void CheckVisitor::Visit_(const Select *op) {
  SetOperatorType(Op::OpType::Select);
  IRVisitor::Visit_(op);
}

void CheckVisitor::SetOperatorType(Op::OpType op_type) {
  for (size_t i = nodes_.size(); i > 0; --i) {
    if (nodes_[i - 1]->name_ == cur_node_->name_ && nodes_[i - 1]->op_.op_type_ != Op::OpType::Input &&
        nodes_[i - 1]->op_.op_type_ == Op::OpType::Assignment) {
      nodes_[i - 1]->op_.op_type_ = op_type;
      break;
    }
  }
}

void CheckVisitor::DefineCallOpType(const Call *op) {
  if (op->call_type == air::ir::Call::CallType::PureIntrinsic) {
    if (op->name == "mad") {
      if (op->args[0].as<Call>()->args.size() == 4) {
        SetOperatorType(Op::OpType::MatMul);
      } else {
        SetOperatorType(Op::OpType::BatchMatMul);
      }
    } else {
      SetOperatorType(Op::OpTypeFromBufferName(op->name));
    }
  }
  if (!nodes_.empty() && nodes_[nodes_.size() - 1]->op_.op_type_ == Op::OpType::Assignment) {
    int idx_axis = 0;
    for (auto const &arg : op->args) {
      if (arg.as<IntImm>() != nullptr) {
        continue;
      }
      if (arg.as<Variable>() != nullptr &&
          arg.as<Variable>()->name_hint != realize_node_.axis_of_node_[idx_axis].name_) {
        SetOperatorType(Op::OpType::TransData);
        break;
      }
      idx_axis++;
    }
  }
}

void CheckVisitor::SetInputNode(const Call *op) {
  std::shared_ptr<Tensor> input_tensor;
  input_tensor = GetTensor(op->func, op->value_index);

  std::shared_ptr<Node> input_node = std::make_shared<Node>();
  input_node->name_ = op->func->func_name();
  input_node->transformed_output_shape_.push_back(*input_tensor);

  for (auto const &arg : op->args) {
    std::vector<std::string> vars{};
    if (arg.as<Variable>() == nullptr) {
      vars = GetVarNamesFromExpr(arg);
      if (vars.empty()) {
        continue;
      }
    } else {
      vars.push_back(arg.as<Variable>()->name_hint);
    }
    for (auto const &var : vars) {
      auto iter = realize_node_.axis_to_tensor_to_shape_id_map_.find(var);
      auto res = iter->second.begin();
      int64_t range = res->second;

      Axis axis_call;
      axis_call.name_ = var;
      axis_call.range_ = range;
      input_node->axis_of_node_.push_back(axis_call);

      std::map<Tensor, int64_t> tensor_to_range_map = {{*input_tensor, range}};
      input_node->axis_to_tensor_to_shape_id_map_.insert(std::make_pair(axis_call.name_, tensor_to_range_map));
    }
    vars.clear();
  }

  input_node->op_.op_type_ = Op::OpType::Input;
  input_node->output_tensors_.push_back(input_tensor);
  input_node->succ_.push_back(cur_node_);

  cur_node_->input_tensors_.push_back(input_tensor);
  cur_node_->pred_.push_back(input_node);
  nodes_.push_back(input_node);
}

void CheckVisitor::UpdateCurrentNode(const air::ir::FunctionRef &func, int value_index,
                                     const std::shared_ptr<Node> &node) {
  std::shared_ptr<Tensor> input_tensor;
  input_tensor = GetTensor(func, value_index);
  cur_node_->input_tensors_.push_back(input_tensor);
  if (dtype_change_.first) {
    cur_node_->output_tensors_.back()->datatype_ = DataTypeFromString(dtype_change_.second);
    dtype_change_.first = false;
  }
  cur_node_->pred_.push_back(node);
}

std::string CheckVisitor::GetNewNodeName(const std::shared_ptr<Node> &node, const air::Array<air::Expr> &op_args) {
  std::vector<std::string> all_args{};
  for (auto const &arg : op_args) {
    std::vector<std::string> vars{};
    if (arg.as<Variable>() == nullptr) {
      vars = GetVarNamesFromExpr(arg);
    } else {
      vars.push_back(arg.as<Variable>()->name_hint);
    }
    if (!vars.empty()) {
      all_args.insert(std::end(all_args), std::begin(vars), std::end(vars));
      vars.clear();
    }
  }

  size_t count_axis_not_in_args = node->axis_of_node_.size();
  std::vector<std::string> args_in_common{};
  for (auto const &axis : node->axis_of_node_) {
    auto arg_it = all_args.begin();
    while (arg_it != all_args.end()) {
      if (axis.name_ == *arg_it) {
        arg_it = all_args.erase(arg_it);
        args_in_common.push_back(axis.name_);
        --count_axis_not_in_args;
        break;
      }
      ++arg_it;
    }
  }

  std::string node_name;
  if (!all_args.empty()) {
    node_name = node->name_;
    for (auto const &arg : all_args) {
      node_name.append("_" + arg);
    }
  } else if (count_axis_not_in_args > 0) {
    node_name = node->name_;
    for (auto const &arg : args_in_common) {
      node_name.append("_" + arg);
    }
  }

  return node_name;
}

std::shared_ptr<Tensor> CheckVisitor::GetTensor(const air::ir::FunctionRef &func, int value_index) {
  std::shared_ptr<Tensor> tensor = std::make_shared<Tensor>();
  air::Tensor op_tensor = Downcast<Operation>(func).output(value_index);

  for (auto shape : op_tensor->shape) {
    CHECK(shape.as<IntImm>());
    auto range = static_cast<int64_t>(shape.as<IntImm>()->value);
    tensor->shape_.push_back(range);
  }
  tensor->name_ = func->func_name();
  tensor->datatype_ = Tensor::GetDataTypeFromTVM(op_tensor->dtype);

  return tensor;
}

std::shared_ptr<Tensor> CheckVisitor::GetTensor(const air::Array<air::Expr> &args, const std::string &name) {
  std::shared_ptr<Tensor> tensor = std::make_shared<Tensor>();
  for (auto const &arg : args) {
    if (arg.as<Variable>() == nullptr) {
      tensor->shape_.push_back(1);
      continue;
    }
    auto iter = realize_node_.axis_to_tensor_to_shape_id_map_.find(arg.as<Variable>()->name_hint);
    auto res = iter->second.begin();
    int64_t range = res->second;
    tensor->shape_.push_back(range);
  }
  tensor->name_ = name;
  return tensor;
}

std::string CheckVisitor::GetDatatypeString(air::DataType op_dtype) {
  std::ostringstream oss;
  oss << op_dtype;
  return oss.str();
}

std::vector<std::string> CheckVisitor::GetVarNamesFromExpr(const Expr &expr) {
  std::stringstream sstream;
  sstream << expr;
  std::string expr_str = sstream.str();

  std::replace_if(
    expr_str.begin(), expr_str.end(), [](const char c) -> bool { return (c != '_' && std::isalnum(c) == 0); }, ' ');

  std::stringstream all_vars(expr_str);
  std::vector<std::string> vars{};
  std::string var;
  while (std::getline(all_vars, var, ' ')) {
    bool is_number =
      std::find_if(var.begin(), var.end(), [](unsigned char c) { return std::isdigit(c) == 0; }) == var.end();
    if (var.empty() || is_number) {
      continue;
    }
    vars.push_back(var);
  }
  return vars;
}
}  // namespace poly
}  // namespace ir
}  // namespace akg
