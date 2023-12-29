/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include <algorithm>
#include <dmlc/logging.h>

#include "poly/tiling/hermes/op.h"

namespace akg {
namespace ir {
namespace poly {
const std::vector<std::tuple<Op::OpType, std::string, std::string>> Op::ops_{
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::Add, "Add", "add"},
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::TensorAdd, "TensorAdd", "add"},
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::AddN, "AddN", "elemwise_sum"},
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::Cast, "Cast", "cast"},
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::Round, "Round", "round"},
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::Trunc, "Trunc", "trunc"},
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::Trunc, "Floor", "floor"},
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::Ceil, "Ceil", "ceil"},
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::Exp, "Exp", "exp"},
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::Fabs, "Fabs", "fabs"},
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::Log, "Log", "log"},
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::Pow, "Pow", "pow"},
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::Mul, "Mul", "multiply"},
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::Maximum, "Maximum", "maximum"},
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::Minimum, "Minimum", "minimum"},
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::FArgMax, "FArgMax", "fargmax"},
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::FArgMin, "FArgMin", "fargmin"},
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::Relu, "Relu", "relu"},
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::Neg, "Neg", "negative"},
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::RealDiv, "RealDiv", "divide"},
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::Reciprocal, "Reciprocal", "divide"},
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::Sub, "Sub", "subtract"},
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::Sqrt, "Sqrt", "sqrt"},
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::Rsqrt, "Rsqrt", "rsqrt"},
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::Tanh, "Tanh", "tanh"},
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::Equal, "Equal", "equal"},
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::Greater, "Greater", "greater"},
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::Lower, "Lower", "lower"},
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::Select, "Select", "select"},
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::BitwiseAnd, "BitwiseAnd", "bitwise_and"},
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::BitwiseOr, "BitwiseOr", "bitwise_or"},
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::BitwiseNot, "BitwiseNot", "bitwise_not"},
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::BroadcastTo, "BroadCastTo", "broadcast"},
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::TransData, "TransData", "transpose"},
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::InplaceAssign, "InplaceAssign", "inplace_assign"},
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::Reshape, "Reshape", "reshape"},
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::FourToFiveNCHW, "FourToFiveNCHW", "four2five_nchw"},
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::Input, HInputOp::input, ""},
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::Constant, HInputOp::cst, ""},
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::BatchMatMul, "BatchMatMul", "batchmatmul"},
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::MatMul, "MatMul", "batchmatmul"},
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::ReduceSum, "ReduceSum", ""},
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::ReduceMin, "ReduceMin", ""},
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::ReduceMax, "ReduceMax", ""},
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::ReduceProd, "ReduceProd", ""},
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::ReduceSRC, "ReduceSRC", ""},
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::ReduceDST, "ReduceDST", ""},
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::ReduceY, "ReduceY", ""},
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::ReduceX, "ReduceX", ""},
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::AllReduce, "AllReduce", ""},
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::VMAdd, "VMAdd", "vmadd"},
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::VMLA, "VMLA", "vmla"},
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::IoU, "IoU", "iou"},
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::IuF, "IuF", "iuf"},
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::NMS, "NMS", "nms"},
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::LHS, "LHS", "lhs"},
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::RHS, "RHS", "rhs"},
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::Dropout, "Dropout", "dropout"},
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::Orig, "Orig", "orig"},
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::With, "With", "with"},
  std::tuple<Op::OpType, std::string, std::string>{Op::OpType::Assignment, "Assignment", ""},
};

Op::Op() : op_type_{Op::OpType::Assignment} {}

Op::Op(OpType op_type) : op_type_{op_type} {}

Op::Op(const std::string &op_name) : op_type_{OpTypeFromString(op_name)} {}

Op::OpType Op::OpTypeFromString(const std::string &op_type) {
  auto it =
    std::find_if(ops_.begin(), ops_.end(), [&op_type](const std::tuple<Op::OpType, std::string, std::string> &op) {
      return std::get<Op::Source::Info>(op) == op_type;
    });
  if (it == ops_.end()) {
    LOG(FATAL) << "[OpTypeFromString] The primitive operator " + op_type + " is not taken into account yet";
  }

  return std::get<Op::Source::Enum>(*it);
}

std::string Op::ToString() const {
  auto it = std::find_if(ops_.begin(), ops_.end(), [this](const std::tuple<Op::OpType, std::string, std::string> &op) {
    return std::get<Op::Source::Enum>(op) == op_type_;
  });
  if (it == ops_.end()) {
    LOG(FATAL) << "[Op::ToString] This op_type is not taken into account yet";
  }

  return std::get<Op::Source::Info>(*it);
}

std::string Op::BufferName() const {
  auto it = std::find_if(ops_.begin(), ops_.end(), [this](const std::tuple<Op::OpType, std::string, std::string> &op) {
    return std::get<Op::Source::Enum>(op) == op_type_;
  });
  if (it == ops_.end()) {
    LOG(FATAL) << "[Op::BufferName] This op_type is not taken into account yet";
  }

  return std::get<Op::Source::IR>(*it);
}

Op::OpType Op::OpTypeFromBufferName(const std::string &op_type) {
  auto it =
    std::find_if(ops_.begin(), ops_.end(), [&op_type](const std::tuple<Op::OpType, std::string, std::string> &op) {
      return std::get<Op::Source::IR>(op) == op_type;
    });
  if (it == ops_.end()) {
    LOG(FATAL) << "[OpTypeFromBufferName] The primitive operator " + op_type + " is not taken into account yet";
  }

  return std::get<Op::Source::Enum>(*it);
}

bool Op::IsReduce() const {
  switch (op_type_) {
    case Op::OpType::ReduceSum:
      return true;
    case Op::OpType::ReduceMax:
      return true;
    case Op::OpType::ReduceMin:
      return true;
    case Op::OpType::AllReduce:
      return true;
    case Op::OpType::ReduceX:
      return true;
    case Op::OpType::ReduceY:
      return true;
    default:
      return false;
  }
}

bool Op::IsConstant() const {
  switch (op_type_) {
    case Op::OpType::Constant:
      return true;
    default:
      return false;
  }
}

// Not used
bool Op::IsNameless() const {
  switch (op_type_) {
    case Op::OpType::Constant:
      return true;
    case Op::OpType::Reshape:
      return true;
    default:
      return false;
  }
}

bool Op::RemoveUselessInput() const {
  switch (op_type_) {
    case Op::OpType::InplaceAssign:
      return true;
    default:
      return false;
  }
}

bool Op::IsInput() const {
  switch (op_type_) {
    case Op::OpType::Input:
      return true;
    case Op::OpType::Constant:
      return true;
    default:
      return false;
  }
}

bool Op::IsLonely() const {
  switch (op_type_) {
    case Op::OpType::Exp:
      return true;
    case Op::OpType::Fabs:
      return true;
    case Op::OpType::Ceil:
      return true;
    case Op::OpType::Sqrt:
      return true;
    case Op::OpType::Log:
      return true;
    case Op::OpType::BroadcastTo:
      return true;
    case Op::OpType::TransData:
      return true;
    case Op::OpType::AddN:
      return true;
    default:
      return false;
  }
}

bool hasReduceName(const std::string &name) { return !(name.find("red", name.size() - 4) == std::string::npos); }

bool hasName(const std::string &name, const std::string &opBuffName) {
  return (!(name.rfind(opBuffName, 0) == std::string::npos));
}

bool hasOpName(const std::string &name, const std::string &opBuffName) { return hasName(name, ("T_" + opBuffName)); }

bool Op::FitBufferName(const std::string &name, bool cstInput) const {
  std::string bufName = BufferName();
  if (IsReduce()) {
    return hasReduceName(name);
  }
  if (IsLonely() && cstInput) {
    return hasName(name, bufName);
  }
  if (IsInput() || IsConstant()) {
    return true;
  }
  return hasOpName(name, bufName);
}

Op::OpCategory Op::Category() const {
  switch (op_type_) {
    case Op::OpType::Add:
    case Op::OpType::AddN:
    case Op::OpType::TensorAdd:
    case Op::OpType::Cast:
    case Op::OpType::Round:
    case Op::OpType::Trunc:
    case Op::OpType::Floor:
    case Op::OpType::Exp:
    case Op::OpType::Fabs:
    case Op::OpType::Ceil:
    case Op::OpType::Pow:
    case Op::OpType::Equal:
    case Op::OpType::Greater:
    case Op::OpType::Lower:
    case Op::OpType::Select:
    case Op::OpType::BitwiseAnd:
    case Op::OpType::BitwiseOr:
    case Op::OpType::BitwiseNot:
    case Op::OpType::Log:
    case Op::OpType::Mul:
    case Op::OpType::Maximum:
    case Op::OpType::Minimum:
    case Op::OpType::Neg:
    case Op::OpType::FArgMax:
    case Op::OpType::FArgMin:
    case Op::OpType::Relu:
    case Op::OpType::RealDiv:
    case Op::OpType::Reciprocal:
    case Op::OpType::Sub:
    case Op::OpType::Sqrt:
    case Op::OpType::Rsqrt:
    case Op::OpType::Tanh:
    case Op::OpType::InplaceAssign:
    case Op::OpType::VMAdd:
    case Op::OpType::VMLA:
    case Op::OpType::IoU:
    case Op::OpType::IuF:
    case Op::OpType::NMS:
    case Op::OpType::LHS:
    case Op::OpType::RHS:
    case Op::OpType::Dropout:
    case Op::OpType::Orig:
    case Op::OpType::With:
      return Op::OpCategory::Injective;

    case Op::OpType::BroadcastTo:
      return Op::OpCategory::Broadcast;

    case Op::OpType::TransData:
      return Op::OpCategory::Transpose;

    case Op::OpType::Reshape:
    case Op::OpType::FourToFiveNCHW:
      return Op::OpCategory::Reshape;

    case Op::OpType::Input:
    case Op::OpType::Constant:
      return Op::OpCategory::Input;

    case Op::OpType::ReduceSum:
    case Op::OpType::ReduceMax:
    case Op::OpType::ReduceMin:
    case Op::OpType::ReduceProd:
    case Op::OpType::ReduceDST:
    case Op::OpType::ReduceSRC:
    case Op::OpType::AllReduce:
      return Op::OpCategory::AllReduce;

    case Op::OpType::ReduceY:
      return Op::OpCategory::ReduceY;

    case Op::OpType::ReduceX:
      return Op::OpCategory::ReduceX;

    case Op::OpType::MatMul:
    case Op::OpType::BatchMatMul:
      return Op::OpCategory::MatMul;

    case Op::OpType::Assignment:
      return Op::OpCategory::Assignment;

    default:
      LOG(FATAL) << "[Op::category] This op_type is not taken into account yet";
  }
  return Op::OpCategory::Injective;
}

int Op::Priority(Op::OpCategory category) {
  Op op;
  auto priority = op.op_category_priority_map_.find(category);
  if (priority != op.op_category_priority_map_.end()) {
    return static_cast<int>(priority->second);
  }
  LOG(FATAL) << "[Op.cc] This OpCategory has no priority assigned";
  return 0;
}

Op::OpCategory Op::DominantCategory(Op::OpCategory cat_1, Op::OpCategory cat_2) {
  if (Op::Priority(cat_1) > Op::Priority(cat_2)) {
    return cat_1;
  }
  return cat_2;
}

std::string StringOfCategory(Op::OpCategory cat) {
  switch (cat) {
    case Op::OpCategory::Input:
      return "Input";
    case Op::OpCategory::Injective:
      return "Injective";
    case Op::OpCategory::Broadcast:
      return "Broadcast";
    case Op::OpCategory::Reshape:
      return "Reshape";
    case Op::OpCategory::Transpose:
      return "Transpose";
    case Op::OpCategory::AllReduce:
      return "AllReduce";
    case Op::OpCategory::ReduceX:
      return "ReduceX";
    case Op::OpCategory::ReduceY:
      return "ReduceY";
    case Op::OpCategory::MatMul:
      return "MatMul";
    case Op::OpCategory::Assignment:
      return "Assignment";
    default:
      LOG(FATAL) << "[StringOfCategory] This OpCategory is not taken into account yet";
  }
  return "";
}
}  // namespace poly
}  // namespace ir
}  // namespace akg
