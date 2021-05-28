/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "composite/optimize/clean_zero_align.h"
#include "pass/utils.h"

namespace akg {

// The minimum unit from ub to gm on ascend device is 32B.
// Therefore, 32B alignment is also required for clean zero to output, for example, atomic_clean.
// For example:
// // attr [{"input_0_format": "DefaultFormat", "shape": [1]}] attrs = 1
// output_0_0(1) = BroadcastTo(0f):int32:PI
//
// To ===>
//
// // attr [{"input_0_format": "DefaultFormat", "shape": [8]}] attrs = 1
// output_0_0(8) = BroadcastTo(0f):int32:PI
class CleanZeroAlign : public IRMutator {
 public:
  explicit CleanZeroAlign(BuildInfo &info) : info_(info) {}

  Stmt Align(const Stmt &stmt) {
    PostOrderVisit(stmt, [&](const NodeRef &node) {
      if (auto provide = node.as<Provide>()) {
        if (auto call = provide->value.as<Call>()) {
          if (call->name == "BroadcastTo" && call->args.size() == 1 && IsZero(call->args[0])) {
            clean_zero_funcs_.insert(provide->func);
          }
        }
      }
      // If it is used in other places later, it will not be aligned
      if (auto call = node.as<Call>()) {
        if (clean_zero_funcs_.count(call->func)) {
          clean_zero_funcs_.erase(call->func);
        }
      }
    });
    return Mutate(stmt);
  }

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) override {
    auto new_body = Mutate(op->body);
    if (new_body.same_as(op->body)) {
      return s;
    }
    CHECK_EQ(op->attr_key, "attrs");
    auto attrs = Downcast<Map<std::string, NodeRef>>(op->node);
    CHECK(attrs.count("shape"));
    Array<Expr> shape = Downcast<Array<Expr>>(attrs["shape"]);
    auto align_shape = AlignShape(shape);
    attrs.Set("shape", align_shape);
    return AttrStmt::make(attrs, op->attr_key, op->value, new_body);
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) override {
    if (!clean_zero_funcs_.count(op->func)) {
      return s;
    }
    dtype_byte_ = op->value.type().bytes();
    auto shape_align = AlignShape(op->args);
    if (shape_align.same_as(op->args)) {
      return s;
    }
    return Provide::make(op->func, op->value_index, op->value, shape_align);
  }

 private:
  Array<Expr> AlignShape(const Array<Expr> &shape) {
    auto ascend_align_byte = 32;
    auto shape_align_unit = ascend_align_byte / dtype_byte_;
    auto shape_total = Expr(1);
    for (auto dim : shape) {
      shape_total = shape_total * dim;
    }
    shape_total = Simplify(shape_total);
    auto shape_align_trunc = Simplify(truncdiv(shape_total, shape_align_unit) * shape_align_unit);
    auto shape_align =
      if_then_else(shape_align_trunc == shape_total, shape_total, Simplify(shape_align_trunc + shape_align_unit));
    if (shape_align.same_as(shape_total)) {
      return shape;
    }
    return Array<Expr>{shape_align};
  }

  bool IsZero(const Expr &expr) {
    if (auto *op = expr.as<FloatImm>()) {
      if (op->value == 0) {
        return true;
      }
    }
    if (is_const_int(expr, 0)) {
      return true;
    }
    return false;
  }

  int dtype_byte_ = 4;
  std::unordered_set<FunctionRef, NodeHash, NodeEqual> clean_zero_funcs_;
  BuildInfo info_;
};

Stmt CleanZeroAligner::Run(const Stmt &stmt) { return CleanZeroAlign(info_).Align(stmt); }
}  // namespace akg
