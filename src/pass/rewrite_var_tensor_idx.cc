/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

/**
 * Replace variables in tensor index with broadcast and reduce.
 * Make sure tensor index is an affine expr of constants and loop vars.
 */
#include <tvm/ir.h>
#include <tvm/tensor.h>
#include <tvm/ir_functor_ext.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_pass.h>
#include <tvm.h>
#include <ir_pass.h>
#include <pass/ir_util.h>

/*
 * Example before this pass:

    input   var3([0, 1000], [0, 8], [0, 4]);
    input   var1([0, 32]);
    input   var2([0, 1000], [0, 16]);
    realize compute([0, 1000], [0, 32], [0, 16]) {
      for (cc0, 0, 1000) {
        for (cc1, 0, 32) {
          for (cc2, 0, 16) {
            compute(cc0, cc1, cc2) = var3(cc0, var1(cc1), var2(cc0, cc2));
          }
        }
      }
    }

  How this pass works:

  First, we identify the tensor (var3) with variable tensor index(es)
  (var1(cc1) and var2(cc0, cc2) in the example).
  Second, we add loops to select this variable tensor index.
  The loop bounds are determined from the realize range.

  After this pass:

  input   var3([0, 1000], [0, 8], [0, 4]);
  input   var1([0, 32]);
  input   var2([0, 1000], [0, 16]);
  realize compute([0, 1000], [0, 32], [0, 16]) {
    for (cc0, 0, 1000) {
      for (cc1, 0, 32) {
        for (cc2, 0, 16) {
          for (cc3, 0, 8) {
            for (cc4, 0, 4) {
              if (cc3 == var1(cc1) && cc4 == var2(cc0, cc2)) {
                compute(cc0, cc1, cc2) = var3(cc0, cc3, cc4);
              }
            }
          }
        }
      }
    }
  }
 */

namespace akg {
namespace ir {
using Region = Array<Range>;
using TensorName = std::string;
using LoopVarName = std::string;
using LoopVarRange = std::pair<Var, Range>;
using LoopVarsPromotedRange = std::vector<LoopVarRange>;
using LoopVarPromotion = std::pair<Var, Expr>;
using LoopVarsPromotion = std::vector<LoopVarPromotion>;

class RewriteVarTensorIdxMutator : public IRMutator {
 public:
  Stmt run(const Stmt &stmt, const Map<Tensor, Buffer> &extern_buffer) {
    for (auto buffer : extern_buffer) {
      Region region;
      auto buffer_shape = buffer.second->shape;
      std::transform(buffer_shape.begin(), buffer_shape.end(), std::back_inserter(region.CopyOnWrite()->data),
                     [](const Expr &extent) { return (Range::make_by_min_extent(0, extent)); });
      tensor_shape[buffer.first->op->name] = region;
      tensor_type[buffer.first->op->name] = buffer.second->dtype;
    }
    return Mutate(stmt);
  }

 private:
  Stmt Mutate_(const Realize *op, const Stmt &s) override {
    tensor_shape[op->func->func_name()] = op->bounds;
    tensor_type[op->func->func_name()] = op->type;
    auto stmt = IRMutator::Mutate_(op, s);
    tensor_shape.erase(op->func->func_name());
    tensor_type.erase(op->func->func_name());
    return stmt;
  }

  Stmt Mutate_(const For *op, const Stmt &s) override {
    loop_vars.insert(op->loop_var->name_hint);
    auto stmt = IRMutator::Mutate_(op, s);
    loop_vars.erase(op->loop_var->name_hint);
    return stmt;
  }

  template <class T>
  Array<Expr> parseCallArgs(const T *op) {
    // outermost (non-nested) call
    in_call = true;
    std::string name = op->func->func_name();
    if (tensor_shape.count(name) == 0) {
      LOG(FATAL) << "realize scope of tensor " << name << " not found, please check realize";
    }
    auto realize_shape = tensor_shape.find(name)->second;
    CHECK(realize_shape.size() == op->args.size());
    LoopVarName new_loop_var_prefix = "cc";
    size_t new_loop_var_count = 0;
    Array<Expr> new_call_args = op->args;
    for (size_t call_arg_index = 0; call_arg_index < op->args.size(); ++call_arg_index) {
      arg_need_promote = false;
      Expr new_arg = IRMutator::Mutate(op->args[call_arg_index]);
      if (arg_need_promote) {
        LoopVarName candidate_name;
        while (1) {
          candidate_name = new_loop_var_prefix + std::to_string(new_loop_var_count);
          if (loop_vars.count(candidate_name) == 0 && new_loop_vars.count(candidate_name) == 0) break;
          ++new_loop_var_count;
        }
        new_loop_vars.insert(candidate_name);
        const int var_size_bits = 32;
        Var new_loop_var = Variable::make(Int(var_size_bits), candidate_name);
        new_call_args.Set(call_arg_index, new_loop_var);

        loop_vars_promotion.emplace_back(LoopVarPromotion(new_loop_var, new_arg));
        loop_vars_promoted_range.emplace_back(LoopVarRange(new_loop_var, realize_shape[call_arg_index]));
        provide_need_promote = true;
      }
    }

    in_call = false;
    return new_call_args;
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) override {
    in_provide = true;
    provide_need_promote = false;
    new_loop_vars.clear();
    loop_vars_promotion.clear();
    loop_vars_promoted_range.clear();

    auto new_call_args = parseCallArgs(op);
    auto new_value = IRMutator::Mutate(op->value);
    auto stmt = Provide::make(op->func, op->value_index, new_value, new_call_args);
    in_provide = false;

    if (!provide_need_promote) {
      return stmt;
    }

    Array<Expr> selection_conds;
    CHECK_GT(loop_vars_promotion.size(), 0);
    for (const auto &loop_var_pair : loop_vars_promotion) {
      selection_conds.push_back(EQ::make(loop_var_pair.first, loop_var_pair.second));
    }

    Expr selection_cond = selection_conds[0];
    for (size_t cond_idx = 1; cond_idx < selection_conds.size(); cond_idx++) {
      selection_cond = And::make(selection_cond, selection_conds[cond_idx]);
    }
    Stmt loop_body = IfThenElse::make(selection_cond, stmt);

    ForType for_type = ForType::Serial;
    air::ir::DeviceAPI device_api = air::ir::DeviceAPI::None;
    for (const auto &loop_var_pair : loop_vars_promoted_range) {
      const Range &loop_range = loop_var_pair.second;
      const Var &loop_var_name = loop_var_pair.first;
      loop_body = For::make(loop_var_name, loop_range->min, loop_range->extent, for_type, device_api, loop_body);
    }

    return loop_body;
  }

  Expr Mutate_(const Call *op, const Expr &e) override {
    if (!in_provide) {
      return IRMutator::Mutate_(op, e);
    }
    if (op->call_type != Call::CallType::Halide) {
      // ordinary function call (example: round(expr)), not a tensor reference
      return IRMutator::Mutate_(op, e);
    }
    if (in_call) {  // variable loop index, need promote
      arg_need_promote = true;
      return IRMutator::Mutate_(op, e);
    }

    auto new_call_args = parseCallArgs(op);
    return Call::make(op->type, op->name, new_call_args, op->call_type, op->func, op->value_index);
  }

  bool in_call{false};
  bool in_provide{false};
  bool arg_need_promote{false};
  bool provide_need_promote{false};
  std::unordered_map<TensorName, Region> tensor_shape;
  std::unordered_map<TensorName, Type> tensor_type;
  std::unordered_set<LoopVarName> loop_vars;
  std::unordered_set<LoopVarName> new_loop_vars;
  LoopVarsPromotedRange loop_vars_promoted_range;
  LoopVarsPromotion loop_vars_promotion;
};

Stmt RewriteVarTensorIdx(Stmt stmt, const Map<Tensor, Buffer> &extern_buffer) {
  stmt = RewriteVarTensorIdxMutator().run(stmt, extern_buffer);
  return stmt;
}
}  // namespace ir
}  // namespace akg
