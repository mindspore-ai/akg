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

#include <dmlc/common.h>
#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_visitor.h>
#include <tvm/arithmetic.h>
#include <tvm.h>
#include <ir_pass.h>
#include <pass/ir_util.h>
#include <pass/utils.h>

namespace akg {
namespace ir {

/*
 This pass tries to fuse continuous axis of tensor.
 The example (note `i` and `k1` have been fused to `i_k1_fused`):
  realize T_csr_reduce_sum_input_1_1<float32>([0, 2708], [0, 8]) {
    produce T_csr_reduce_sum_input_1_1 {
      for (i, 0, 2708) {
        for (k1, 0, 8) {
          T_csr_reduce_sum_input_1_1(i, k1) = 0f
          for (j, 0, (input_3((i + 1)) - input_3(i))) {
            ....
          }
        }
      }
    }
  }
 ...............
 ========>
 // attr [extern(T_csr_reduce_sum_input_1_1, 0x556e033292b0)] realize_scope = ""
  realize T_csr_reduce_sum_input_1_1<float32>([0, 2708], [0, 8]) {
    produce T_csr_reduce_sum_input_1_1 {
      for (i_k1_fused, 0, 21664) {
        T_csr_reduce_sum_input_1_1(floordiv(i_k1_fused, 8), floormod(i_k1_fused, 8)) = 0f
        for (j, 0, (input_3((floordiv(i_k1_fused, 8) + 1)) - input_3(floordiv(i_k1_fused, 8)))) {
          ....
        }
      }
    }
  }
 ...............
 */

class FuseAxisExtern : public IRMutator {
 public:
  std::unordered_map<std::string, std::pair<Var, Range>> var_name_with_range;
  explicit FuseAxisExtern(std::unordered_map<std::string, std::pair<Var, Range>> &var_and_range)
      : var_name_with_range{var_and_range} {}
  Stmt Mutate_(const For *op, const Stmt &s) final {
    auto next_for_op = op->body.as<For>();
    if (next_for_op) {
      auto outer_var_name = op->loop_var->name_hint;
      auto inner_var_name = next_for_op->loop_var->name_hint;
      auto fuse_name = outer_var_name + "_" + inner_var_name + "_fused";
      if (CheckFusible(outer_var_name) && CheckFusible(inner_var_name)) {
        Range range = Range::make_by_min_extent(0, op->extent * next_for_op->extent);
        auto fused_var = Variable::make(op->loop_var.type(), fuse_name);
        auto fused_outer_var = FloorDiv::make(fused_var, next_for_op->extent);
        auto fused_inner_var = FloorMod::make(fused_var, next_for_op->extent);
        std::unordered_map<const Variable *, Expr> loop_var_map{{}};
        loop_var_map[op->loop_var.as<Variable>()] = fused_outer_var;
        loop_var_map[next_for_op->loop_var.as<Variable>()] = fused_inner_var;
        auto ret = Substitute(next_for_op->body, loop_var_map);
        ret = Mutate(ret);
        return For::make(fused_var, range->min, range->extent, op->for_type, op->device_api, ret);
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  bool CheckFusible(std::string name_hint) {
    auto idx = name_hint.find("fused");
    return var_name_with_range.count(name_hint) || idx != std::string::npos;
  }

  Expr Mutate_(const FloorDiv *op, const Expr &e) final { return air::ir::CanonicalSimplify(e); }

  Expr Mutate_(const FloorMod *op, const Expr &e) final { return air::ir::CanonicalSimplify(e); }
};

class FusionVarCollector : public IRVisitor {
 public:
  std::unordered_map<std::string, std::pair<Var, Range>> var_name_with_range;
  std::vector<std::string> not_fused_var_name;

  void Visit_(const For *op) override {
    std::string var_name = op->loop_var->name_hint;
    if (op->min.as<IntImm>() && op->extent.as<IntImm>()) {
      auto range = Range::make_by_min_extent(op->min, op->extent);
      var_name_with_range[var_name] = std::make_pair(op->loop_var, range);
    }
    IRVisitor::Visit_(op);
  }

  void Visit_(const AttrStmt *op) override {
    if (op->attr_key == air::ir::attr::reduce_update) {
      auto iter_vars = Downcast<Array<IterVar>>(op->node);
      if (iter_vars.defined()) {
        for (auto iter_var : iter_vars) {
          not_fused_var_name.push_back(iter_var->var->name_hint);
        }
      }
    }
    IRVisitor::Visit_(op);
  }
};

Stmt FuseAxisExternOp(Stmt stmt, air::Schedule sch) {
  constexpr size_t MAX_FUSE_TIMES = 100;
  auto bounds = air::schedule::InferBound(sch);
  auto fusion_var_vollector = FusionVarCollector();
  fusion_var_vollector.Visit(stmt);
  auto var_name_with_range{fusion_var_vollector.var_name_with_range};
  for (auto var_name : fusion_var_vollector.not_fused_var_name) {
    var_name_with_range.erase(var_name);
  }
  auto fuse_axis_extern = FuseAxisExtern(var_name_with_range);
  // prevent infinite loop
  for (size_t i{0}; i < MAX_FUSE_TIMES; ++i) {
    auto fused_stmt = fuse_axis_extern.Mutate(stmt);
    if (fused_stmt.same_as(stmt)) {
      return stmt;
    }
    stmt = fused_stmt;
  }
  return stmt;
}
}  // namespace ir
}  // namespace akg
