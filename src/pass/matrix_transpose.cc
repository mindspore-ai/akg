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

#include <tvm/ir_pass.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_visitor.h>
#include <tvm.h>
#include "ir_pass.h"

namespace akg {
namespace ir {
static constexpr auto PROMOTE_TRANSPOSE = "promoted_transpose";
static constexpr auto MATRIX_TRANSPOSE = "MatrixTranspose";
static constexpr auto INT32 = 32;
static constexpr auto PARAMETER_NUM = 4;

class MatrixTransposeMutator : public IRMutator {
 public:
  explicit MatrixTransposeMutator() {}

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == PROMOTE_TRANSPOSE) {
      return CallTransposeInterface(op, s);
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    auto provide = IRMutator::Mutate_(op, s);
    provide_ = provide;
    return provide;
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    extents_.push_back(op->extent);
    return IRMutator::Mutate_(op, s);
  }

 private:
  Stmt CallTransposeInterface(const AttrStmt *op, const Stmt &s) {
    extents_.clear();
    auto stmt = IRMutator::Mutate(op->body);
    for (auto extent : extents_) {
      auto extent_value = extent.as<IntImm>()->value;
      if (extent_value & (extent_value - 1)) {
        return s;
      }
    }

    Array<Expr> shapes;
    std::reverse(extents_.begin(), extents_.end());
    shapes.assign(extents_.begin(), extents_.end());
    extents_.clear();

    auto provide = provide_.as<Provide>();
    CHECK(provide->value.as<Call>());
    auto pro_value = provide->value.as<Call>();
    auto pro_func = provide->func;

    Array<Expr> indices;
    for (auto arg : pro_value->args) {
      indices.push_back(make_zero(Int(INT32)));
    }

    // Array<Expr> indices;
    Array<Expr> args;
    args.push_back(make_zero(Int(INT32)));
    args.push_back(make_zero(Int(INT32)));
    for (auto shape : shapes) {
      args.push_back(shape);
    }
    CHECK(args.size() == PARAMETER_NUM) << "The number of input parameters of the transpose interface must be 4.";
    Expr dst_call = Call::make(pro_value->type, pro_func->func_name(), indices, pro_value->call_type, pro_func,
                               pro_value->value_index);
    Expr src_call = Call::make(pro_value->type, pro_value->name, indices, pro_value->call_type, pro_value->func,
                               pro_value->value_index);

    Expr dst_addr = Call::make(Handle(), air::ir::intrinsic::tvm_address_of, {dst_call}, Call::PureIntrinsic);
    Expr src_addr = Call::make(Handle(), air::ir::intrinsic::tvm_address_of, {src_call}, Call::PureIntrinsic);
    args.Set(0, dst_addr);
    args.Set(1, src_addr);
    return Evaluate::make(Call::make(Handle(), MATRIX_TRANSPOSE, args, Call::Intrinsic));
  }

 private:
  std::vector<Expr> extents_;
  Stmt provide_;
};

Stmt MatrixTranspose(const Stmt &stmt) { return MatrixTransposeMutator().Mutate(stmt); }
}  // namespace ir
}  // namespace akg
