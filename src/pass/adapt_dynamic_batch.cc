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

#include <dmlc/common.h>
#include <tvm/ir.h>
#include <tvm/expr.h>
#include <tvm/operation.h>
#include <tvm/ir_mutator.h>
#include <tvm/expr_operator.h>
#include <tvm/ir_pass.h>
#include <tvm/buffer.h>
#include <tvm/build_module.h>
#include <unordered_map>
#include <unordered_set>
#include <regex>
#include "common/common_util.h"
#include "pass/utils.h"
#include "ir_pass.h"

namespace akg {
namespace ir {
std::vector<size_t> parse_str(const std::string &s) {
  std::regex delimiters(",");
  std::vector<std::string> index(std::sregex_token_iterator(s.begin(), s.end(), delimiters, -1),
                                 std::sregex_token_iterator());
  std::vector<size_t> out_index;
  for (size_t i = 0; i < index.size(); i++) {
    out_index.push_back(std::stoul(index[i]));
  }
  return out_index;
}

/* Add batch axis for output and target input
 * === Example 1 ===
 * target input: input_1
 * for (ax0, 0, 16)
 *   for (ax1, 0, 16)
 *     T(ax0, ax1) = input0(ax0, ax1)+input1(ax0,ax1)
 * -->
 * for(bs0, 0, batch(0))
 *  for (ax0, 0, 16)
 *   for (ax1, 0, 16)
 *     T(bs0 * T.shape[0] + ax0, ax1) = input0(ax0, ax1) + input1(bs0 * input1.shape[0] + ax0, ax1)
 */
class AddBatchMutator : public IRMutator {
 public:
  AddBatchMutator(const Array<Tensor> &dyn_inputs, const Map<Tensor, Buffer> &binds, const Tensor &bs) {
    Array<Expr> args;
    args.push_back(Expr(0));
    bs_type_ = bs->dtype;
    bs_ = Call::make(bs->dtype, bs->op->name, args, Call::CallType::Halide, bs->op);
    for (auto &i : dyn_inputs) {
      (void)dyn_inputs_.emplace(i->op->name, i);
    }
    for (auto &kv : binds) {
      (void)binds_.emplace(kv.first->op->name, kv.first);
    }
  }
  ~AddBatchMutator() = default;

  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (ongoing_for_ == 0) {
      ongoing_for_++;
      add_batch_already_ = true;
      use_batch_already_ = false;
      loop_var_ = Variable::make(bs_type_, "bs" + std::to_string(new_var_count_));
      new_var_count_++;
      Stmt body = IRMutator::Mutate(s);
      add_batch_already_ = false;
      ongoing_for_--;
      if (!use_batch_already_) {
        return body;
      }
      return For::make(loop_var_, Expr(0), bs_, ForType::Parallel, DeviceAPI::None, body);
    } else {
      ongoing_for_++;
      auto stmt = IRMutator::Mutate_(op, s);
      ongoing_for_--;
      return stmt;
    }
  }

  Expr Mutate_(const Call *op, const Expr &e) final {
    if (op->func.defined()) {
      auto it = dyn_inputs_.find(op->func->func_name());
      if (it == dyn_inputs_.end()) {
        it = binds_.find(op->func->func_name());
        if (it == binds_.end() || tensor_with_batch_.count(op->func->func_name()) == 0) {
          return IRMutator::Mutate_(op, e);
        }
      }
      if (!add_batch_already_) {
        LOG(FATAL) << "Cannot add arg: bs, bs is not defined";
      }
      (void)tensor_with_batch_.insert(op->func->func_name());
      use_batch_already_ = true;
      Array<Expr> update_args;
      for (size_t i = 0; i < op->args.size(); i++) {
        auto cur_arg = IRMutator::Mutate(op->args[i]);
        if (i == 0) {
          update_args.push_back(Add::make(Mul::make(loop_var_, it->second->shape[0]), cur_arg));
        } else {
          update_args.push_back(cur_arg);
        }
      }
      return Call::make(op->type, op->name, update_args, op->call_type, op->func);
    }
    return IRMutator::Mutate_(op, e);
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    auto it = binds_.find(op->func->func_name());
    if (it == binds_.end()) {
      auto value = IRMutator::Mutate(op->value);
      return Provide::make(op->func, op->value_index, value, op->args);
    }
    if (!add_batch_already_) {
      LOG(FATAL) << "Cannot add arg: bs, bs is not defined";
    }
    (void)tensor_with_batch_.insert(op->func->func_name());
    use_batch_already_ = true;
    Array<Expr> update_args;
    for (size_t i = 0; i < op->args.size(); i++) {
      auto cur_arg = IRMutator::Mutate(op->args[i]);
      if (i == 0) {
        update_args.push_back(Add::make(Mul::make(loop_var_, it->second->shape[0]), cur_arg));
      } else {
        update_args.push_back(cur_arg);
      }
    }
    auto value = IRMutator::Mutate(op->value);
    return Provide::make(op->func, op->value_index, value, update_args);
  }

 private:
  Expr bs_;
  Type bs_type_;
  Var loop_var_;
  std::unordered_map<std::string, Tensor> dyn_inputs_;
  std::unordered_map<std::string, Tensor> binds_;
  std::unordered_set<std::string> tensor_with_batch_;
  int new_var_count_{0};
  int ongoing_for_{0};
  bool add_batch_already_{false};
  bool use_batch_already_{false};
};

Stmt AdaptDynamicBatch(const Stmt &stmt, const Array<NodeRef> &args, const Map<Tensor, Buffer> &binds, const Tensor &bs,
                       const NodeRef &dynamic_input_index) {
  std::vector<size_t> dyn_index = parse_str(dynamic_input_index.as<StringImm>()->value);
  if (dyn_index.empty()) {
    return stmt;
  }
  Array<Tensor> dyn_inputs;
  for (const auto &i : dyn_index) {
    CHECK_GT(args.size(), i);
    auto buf = args[i];
    for (auto &kv : binds) {
      if (kv.second == buf) {
        dyn_inputs.push_back(kv.first);
      }
    }
  }
  AddBatchMutator add_batch_mutator(dyn_inputs, binds, bs);
  return add_batch_mutator.Mutate(stmt);
}
}  // namespace ir
}  // namespace akg
