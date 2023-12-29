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
#include "base/ir_checker.h"
#include <cinttypes>
#include <string>
#include "base/dump_helper.h"
#include "base/expr_builder.h"

namespace akg {
int64_t UTIRCheckHelper::GetValueFromImm(const air::Expr &expr) {
  const air::IntImm *imm_int = expr.as<air::IntImm>();
  if (imm_int != nullptr) {
    return imm_int->value;
  }
  const air::ir::UIntImm *imm_uint = expr.as<air::ir::UIntImm>();
  if (imm_uint != nullptr) {
    CHECK(imm_uint->value < INT64_MAX);
    return static_cast<int64_t>(imm_uint->value);
  }
  return 0;
}

void UTProvideChecker::Visit_(const air::ir::For *op) {
  uint64_t count_top = for_count_stack_.back();
  int64_t min = UTIRCheckHelper::GetValueFromImm(op->min);
  int64_t extent = UTIRCheckHelper::GetValueFromImm(op->extent);
  CHECK(extent > min);
  count_top *= static_cast<uint64_t>(extent);
  for_count_stack_.push_back(count_top);
  IRVisitor::Visit_(op);
  for_count_stack_.pop_back();
}

bool UTProvideChecker::CompareDump(
    const std::string &dump,
    const std::string &target) {
  if (dump.compare(target) == 0) {
    return true;
  }
  if (ignore_args_) {
    size_t npos = dump.find("(");
    return dump.substr(0, npos).compare(target) == 0;
  }
  return false;
}

std::vector<std::tuple<std::string, const air::ir::Provide*, uint64_t>> UTProvideCheckerForAssign::Find(
    const air::NodeRef &node,
    const std::string &dump_rhs) {
  dump_rhs_ = dump_rhs;
  infos_lhs_.clear();
  for_count_stack_.clear();
  for_count_stack_.push_back(1);
  Visit(node);
  return infos_lhs_;
}

void UTProvideCheckerForAssign::Visit_(const air::ir::Provide *op) {
  std::string dump_expr = UTDumpHelper::Dump(op->value);
  if (CompareDump(dump_expr, dump_rhs_)) {
    air::Expr expr_call = UTExprBuilder::CreateCall(op->func, op->args);
    infos_lhs_.push_back(std::make_tuple(UTDumpHelper::Dump(expr_call), op, for_count_stack_.back()));
  }
}

std::vector<std::tuple<std::string, const air::ir::Provide*, uint64_t>> UTProvideCheckerForBinary::Find(
    const air::NodeRef &node,
    UTProvideCheckerForBinary::BinaryOpType op_type,
    const std::string &dump_rhs1,
    const std::string &dump_rhs2) {
  op_type_ = op_type;
  dump_rhs1_ = dump_rhs1;
  dump_rhs2_ = dump_rhs2;
  infos_lhs_.clear();
  for_count_stack_.clear();
  for_count_stack_.push_back(1);
  if (dump_rhs1_.empty() && dump_rhs2_.empty()) {
    return infos_lhs_;
  }
  Visit(node);
  return infos_lhs_;
}

void UTProvideCheckerForBinary::Visit_(const air::ir::Provide *op) {
  switch (op_type_) {
    case BinaryOpType::kAdd:
      CheckBinary<air::ir::Add>(op);
      break;
    case BinaryOpType::kSub:
      CheckBinary<air::ir::Sub>(op);
      break;
    case BinaryOpType::kMul:
      CheckBinary<air::ir::Mul>(op);
      break;
    case BinaryOpType::kDiv:
      CheckBinary<air::ir::Div>(op);
      break;
    case BinaryOpType::kMod:
      CheckBinary<air::ir::Mod>(op);
      break;
    default:
      break;
  }
}
}  // namespace akg
