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
 The example:
 // attr [compute(input_1_red, 0x56179ef3f290)] realize_scope = ""
 realize input_1_red<float32>([0, 2], [0, 4]) {
  produce input_1_red {
    for (ax0, 0, 2) {
      for (ax1, 0, 4) {
        input_1_red(ax0, ax1) = 0f
        for (k2, 0, 8) {
          for (k3, 0, 16) {
            for (k4, 0, 32) {
              // attr [[iter_var(k2, range(min=0, ext=8)), iter_var(k3, range(min=0, ext=16)),
              // iter_var(k4, range(min=0, ext=32))]] reduce_update = ""
              input_1_red(ax0, ax1) = (input_1_red(ax0, ax1) + input_1(ax0, ax1, k2, k3, k4))
 ...............
 ========>
 // attr [compute(input_1_red, 0x56179eb4cac0)] realize_scope = ""
 realize input_1_red<float32>([0, 8]) {
  produce input_1_red {
    for (cc0, 0, 8) {
      input_1_red(cc0) = 0f
      for (cc2, 0, 4096) {
        // attr [[iter_var(cc2, range(min=0, ext=4096))]] reduce_update = ""
        input_1_red(cc0) = (input_1_red(cc0) + input_1(cc0, cc2))
 ...............
 */

#define DEBUG_FUSE_AXIS 0

using VarPair = std::pair<const Variable *, const Variable *>;
using IterVarPair = std::pair<const IterVarNode *, const IterVarNode *>;

struct ArrayIterVarHash {
  size_t operator()(const Array<IterVar> &arr) const {
    size_t ret = 0;
    for (auto &iter_var : arr) {
      ret = dmlc::HashCombine(ret, ExprHash()(iter_var));
    }
    return ret;
  }
};

std::string VarPairStr(const VarPair &var_pair) {
  std::stringstream repr;
  repr << "(" << var_pair.first->name_hint << "(" << var_pair.first << ")"
       << ", " << var_pair.second->name_hint << "(" << var_pair.second << ")"
       << ")";
  return repr.str();
}

bool RangeIsZeroToOne(const Range &r) { return is_zero(r->min) && is_one(r->extent); }

// To Find the VarPair that can be fused, for the example:
// For the first time run: (ax0, ax1), (k2, k3), (k3,k4)
// For the second time run: (cc1, k4)
class FindVarPair : public IRVisitor {
 public:
  FindVarPair() = default;
  ~FindVarPair() override = default;

  void Visit_(const For *op) final {
    std::unordered_map<const Variable *, Range> for_var_range;
    std::vector<const Variable *> continue_loop_vars;
    auto last_op = op;

    while (op) {
      const Variable *var = op->loop_var.get();
      loop_var_name_var[var->name_hint] = var;
      loop_range_[var] = Range::make_by_min_extent(op->min, op->extent);
      continue_loop_vars.push_back(var);
      last_op = op;
      op = last_op->body.as<For>();
    }
    auto var_pairs = GetVarPairs(continue_loop_vars);
    cur_fuse_var_pairs_.insert(var_pairs.begin(), var_pairs.end());
    Visit(last_op->body);
    for (auto var_pair : var_pairs) {
      if (cur_fuse_var_pairs_.erase(var_pair) && !disable_fuse_var_.count(var_pair.first) &&
          !disable_fuse_var_.count(var_pair.second)) {
        fuse_var_pairs_.emplace(var_pair);
      }
    }
  }

  void Visit_(const AttrStmt *op) final {
    if (auto compute_op = op->node.as<ComputeOpNode>()) {
      if (compute_op->reduce_axis.size() > 1) {
        auto func = GetRef<FunctionRef>(compute_op);
        func_reduce_axis_[func] = compute_op->reduce_axis;
      }
    } else if (op->attr_key == air::ir::attr::reduce_update) {
      auto iter_vars = Downcast<Array<IterVar>>(op->node);
      std::vector<const Variable *> loop_vars;
      for (const auto &iter_var : iter_vars) {
        if (loop_var_name_var.count(iter_var->var->name_hint)) {
          loop_vars.push_back(loop_var_name_var.at(iter_var->var->name_hint));
        } else {
          CHECK(RangeIsZeroToOne(iter_var->dom));
          loop_vars.push_back(nullptr);
        }
      }
      reduce_axis_var_[iter_vars].push_back(loop_vars);
    }
    IRVisitor::Visit_(op);
  }

  void Visit_(const Provide *op) final {
    if (op->func.defined() && op->func.as<OperationNode>()) {
      UpdateFuseVarPairsByArgs<Provide>(op);
      Visit(op->value);
    } else {
      IRVisitor::Visit_(op);
    }
  }

  void Visit_(const Call *op) final {
    if (op->func.defined() && op->func.as<OperationNode>()) {
      UpdateFuseVarPairsByArgs<Call>(op);
    } else {
      IRVisitor::Visit_(op);
    }
  }

  void Visit_(const Variable *op) final { disable_fuse_var_.emplace(op); }

  std::unordered_set<VarPair, PairHash> fuse_var_pairs_;
  // record insert order of func_vec_vars_
  std::list<FunctionRef> func_vec_vars_keys_;
  std::unordered_map<FunctionRef, std::vector<std::vector<const Variable *>>, ExprHash, ExprEqual> func_vec_vars_;

  // get var of the args for the reduce axis of func:
  std::unordered_map<FunctionRef, Array<IterVar>, ExprHash, ExprEqual> func_reduce_axis_;
  std::unordered_map<Array<IterVar>, std::vector<std::vector<const Variable *>>, ArrayIterVarHash> reduce_axis_var_;

 private:
  std::unordered_map<const Variable *, Range> loop_range_;
  std::unordered_map<std::string, const Variable *> loop_var_name_var;
  std::unordered_set<const Variable *> disable_fuse_var_;
  std::unordered_set<VarPair, PairHash> cur_fuse_var_pairs_;
  std::unordered_map<FunctionRef, Array<Expr>, ExprHash, ExprEqual> func_axis_extent_;

  template <typename T>
  void UpdateFuseVarPairsByArgs(const T *op) {
    std::vector<const Variable *> tensor_args_var;
    auto args = op->args;
    auto bounds = GetExtents(op->func);
    CHECK_EQ(args.size(), bounds.size());
    for (size_t i = 0; i < args.size(); ++i) {
      if (auto var = args[i].template as<Variable>()) {
        CHECK(loop_range_.count(var));
        if (Equal(loop_range_.at(var)->extent, bounds[i])) {
          tensor_args_var.push_back(var);
          continue;
        }
      } else {
        tensor_args_var.push_back(nullptr);
        Visit(args[i]);
      }
    }
    UpdateFuseVarPair(tensor_args_var);
    if (!func_vec_vars_.count(op->func)) {
      func_vec_vars_keys_.push_back(op->func);
    }
    func_vec_vars_[op->func].emplace_back(tensor_args_var);
  }

  Array<Expr> GetExtents(const FunctionRef &func) {
    if (!func_axis_extent_.count(func)) {
      Array<Expr> extents;
      if (auto compute_op = func.as<ComputeOpNode>()) {
        auto axis = compute_op->axis;
        for (auto &a : axis) {
          extents.push_back(a->dom->extent);
        }
      } else if (auto place_op = func.as<PlaceholderOpNode>()) {
        extents = place_op->shape;
      } else {
        LOG(FATAL) << func << " not been proccessed!" << std::endl;
      }
      func_axis_extent_[func] = extents;
    }
    return func_axis_extent_.at(func);
  }

  void UpdateFuseVarPair(const std::vector<const Variable *> &vars) {
    auto vars_pairs = GetVarPairs(vars);
    std::unordered_set<const Variable *> vars_valid;
    for (auto var : vars) {
      if (var) {
        vars_valid.emplace(var);
      }
    }
    if (vars_valid.empty()) {
      return;
    }

    for (auto it = cur_fuse_var_pairs_.begin(); it != cur_fuse_var_pairs_.end();) {
      if ((vars_valid.count(it->first) || vars_valid.count(it->second)) && !vars_pairs.count(*it)) {
        it = cur_fuse_var_pairs_.erase(it);
      } else {
        ++it;
      }
    }
  }

  std::unordered_set<VarPair, PairHash> GetVarPairs(const std::vector<const Variable *> &vars) {
    std::unordered_set<VarPair, PairHash> var_pairs;
    if (vars.size() <= 1) {
      return var_pairs;
    }
    for (size_t i = 1; i < vars.size(); ++i) {
      if (vars[i - 1] && vars[i]) {
        CHECK(loop_range_.count(vars[i - 1]));
        auto pre_loop_var_range = loop_range_.at(vars[i - 1]);
        CHECK(loop_range_.count(vars[i]));
        auto loop_var_range = loop_range_.at(vars[i]);
        if (is_zero(pre_loop_var_range->min) && is_zero(loop_var_range->min)) {
          var_pairs.emplace(VarPair(vars[i - 1], vars[i]));
        }
      }
    }
    return var_pairs;
  }
};

// Get the index of axis that to be fused for tensor:
// For the first time run:
// input_1_red : [0]; input_1 : [0,2]
// For the second time run:
// input_1 : [1]
class FuseAxisVisit : public IRVisitor {
 public:
  FuseAxisVisit() = default;
  ~FuseAxisVisit() override = default;

  void Run(const NodeRef &stmt) {
    GetFuseIndexVarPair(stmt);
    Visit(stmt);
  }

  void Visit_(const AttrStmt *op) override {
    if (auto compute_op = op->node.as<ComputeOpNode>()) {
      // Get fuse_index by fuse_index_var_pair
      auto func = GetRef<Operation>(compute_op);
      std::unordered_set<size_t> fuse_index;
      if (func_fuse_index_var_pair_.count(func)) {
        fuse_index = GetFuseIndex(func_fuse_index_var_pair_.at(func));
        AddFusedVars(compute_op->axis, fuse_index);
      }
      std::unordered_set<size_t> reduce_fuse_index;
      if (func_reduce_fuse_index_var_pair_.count(func)) {
        reduce_fuse_index = GetFuseIndex(func_reduce_fuse_index_var_pair_.at(func));
        AddFusedVars(compute_op->reduce_axis, reduce_fuse_index);
      }
      if (!fuse_index.empty()) {
        func_fuse_index_[func] = fuse_index;
      }
      if (!reduce_fuse_index.empty()) {
        func_reduce_fuse_index_[func] = reduce_fuse_index;
      }
      VisitArray(compute_op->body);
      visited_.insert(func);
    }
    return IRVisitor::Visit_(op);
  }

  void Visit_(const Provide *op) override {
    if (!(op->func.defined() && op->func.as<OperationNode>())) {
      return IRVisitor::Visit_(op);
    }
    CHECK(visited_.count(op->func));
    return Visit(op->value);
  }

  void Visit_(const Call *op) override {
    auto func = op->func;
    if (!(func.defined() && func.as<OperationNode>())) {
      return IRVisitor::Visit_(op);
    }
    if (visited_.count(func)) {
      return;
    }
    if (func.as<PlaceholderOpNode>()) {
      std::unordered_set<size_t> fuse_index = GetFuseIndexByArgs(func, op->args);
      if (!fuse_index.empty()) {
        func_fuse_index_[func] = fuse_index;
      }
    } else if (auto compute_op = func.as<ComputeOpNode>()) {
      std::unordered_set<size_t> fuse_index = GetFuseIndexByArgs(func, op->args);
      AddFusedVars(compute_op->axis, fuse_index);
      CHECK_EQ(compute_op->reduce_axis.size(), 0)
        << "check fail for " << func << " with reduce_axis " << compute_op->reduce_axis;
      VisitArray(compute_op->body);
      if (!fuse_index.empty()) {
        func_fuse_index_[func] = fuse_index;
      }
    } else {
      LOG(FATAL) << func << " not been proccessed!" << std::endl;
    }
    visited_.insert(func);
  }

  void Visit_(const Variable *op) override {
    if (fused_vars_.count(op)) {
      LOG(FATAL) << "var: " << op->name_hint << " should be fused!" << std::endl;
    }
    return IRVisitor::Visit_(op);
  }

  std::string DumpInfo() {
    std::stringstream info;
    info << "==== output finder fuse_var_pairs_" << std::endl;
    for (auto var_pair : finder.fuse_var_pairs_) {
      info << VarPairStr(var_pair) << std::endl;
    }
    info << "==== output finder fuse_var_pairs_" << std::endl;
    info << "==== output enable_fuse_var_pairs_" << std::endl;
    for (auto var_pair : enable_fuse_var_pairs_) {
      info << VarPairStr(var_pair) << std::endl;
    }
    info << "==== output enable_fuse_var_pairs_ end" << std::endl;
    info << "==== output func_fuse_index_var_pair_" << std::endl;
    info << DumpFuncFuseIndexVarPairInfo(func_fuse_index_var_pair_);
    info << "==== output func_fuse_index_var_pair_ end" << std::endl;
    info << "==== output func_reduce_fuse_index_var_pair_" << std::endl;
    info << DumpFuncFuseIndexVarPairInfo(func_reduce_fuse_index_var_pair_);
    info << "==== output func_reduce_fuse_index_var_pair_ end" << std::endl;
    info << "==== output func_fuse_index_" << std::endl;
    info << DumpFuncFuseIndexInfo(func_fuse_index_);
    info << "==== output func_fuse_index_ end" << std::endl;
    info << "==== output func_reduce_fuse_index_" << std::endl;
    info << DumpFuncFuseIndexInfo(func_reduce_fuse_index_);
    info << "==== output func_reduce_fuse_index_ end" << std::endl;
    return info.str();
  }

  std::unordered_map<FunctionRef, std::unordered_set<size_t>, ExprHash, ExprEqual> func_fuse_index_;
  std::unordered_map<FunctionRef, std::unordered_set<size_t>, ExprHash, ExprEqual> func_reduce_fuse_index_;
  std::unordered_set<const Variable *> fused_vars_;
  std::unordered_set<VarPair, PairHash> fused_var_pairs_;

 private:
  FindVarPair finder;
  std::unordered_set<VarPair, PairHash> enable_fuse_var_pairs_;
  std::unordered_map<FunctionRef, std::vector<std::unordered_set<VarPair, PairHash>>, ExprHash, ExprEqual>
    func_fuse_index_var_pair_;
  std::unordered_map<FunctionRef, std::vector<std::unordered_set<VarPair, PairHash>>, ExprHash, ExprEqual>
    func_reduce_fuse_index_var_pair_;
  std::unordered_map<FunctionRef, std::unordered_set<size_t>, ExprHash, ExprEqual> disable_fuse_index_;
  std::unordered_set<FunctionRef, ExprHash, ExprEqual> visited_;

  void GetFuseIndexVarPair(const NodeRef &stmt) {
    finder.Visit(stmt);
    enable_fuse_var_pairs_ = finder.fuse_var_pairs_;
    UpdateEnableFuseVarPair();
    // Get FuseIndexVarPair
    const auto &func_vec_vars_keys = finder.func_vec_vars_keys_;
    const auto &func_vec_vars = finder.func_vec_vars_;
    for (const auto &func : func_vec_vars_keys) {
      const auto &vec_vars = func_vec_vars.at(func);
      auto vec_fuse_var_pairs = GetFuncIndexVarPairs(vec_vars);
      func_fuse_index_var_pair_[func] = vec_fuse_var_pairs;
    }
    // Get ReduceFuseIndexVarPair
    const auto &func_reduce_axis = finder.func_reduce_axis_;
    const auto &reduce_axis_var = finder.reduce_axis_var_;
    for (auto &f_kv : func_reduce_axis) {
      auto func = f_kv.first;
      auto reduce_axis = f_kv.second;
      CHECK(reduce_axis_var.count(reduce_axis));
      auto &vec_vars = reduce_axis_var.at(reduce_axis);
      auto vec_fuse_var_pairs = GetFuncIndexVarPairs(vec_vars);
      func_reduce_fuse_index_var_pair_[func] = vec_fuse_var_pairs;
    }
  }

  std::vector<std::unordered_set<VarPair, PairHash>> GetFuncIndexVarPairs(
    const std::vector<std::vector<const Variable *>> &vec_vars) {
    auto index_size = vec_vars[0].size();
    std::vector<std::unordered_set<VarPair, PairHash>> vec_fuse_var_pairs(index_size);
    for (size_t i = 0; i < index_size - 1; ++i) {
      for (auto vars : vec_vars) {
        auto var_pair = VarPair(vars[i], vars[i + 1]);
        if (enable_fuse_var_pairs_.count(var_pair)) {
          vec_fuse_var_pairs[i].emplace(var_pair);
        } else {
          vec_fuse_var_pairs[i].clear();
          break;
        }
      }
    }
    return vec_fuse_var_pairs;
  }

  // From the perspective of func axis to update the VarPairs
  void UpdateEnableFuseVarPair() {
    const auto &func_vec_vars_keys = finder.func_vec_vars_keys_;
    const auto &func_vec_vars = finder.func_vec_vars_;
    CHECK_EQ(func_vec_vars_keys.size(), func_vec_vars.size());
    std::unordered_set<VarPair, PairHash> pre_enable_fuse_var_pairs;
    do {
      pre_enable_fuse_var_pairs = enable_fuse_var_pairs_;
      for (const auto &func : func_vec_vars_keys) {
        const auto &vec_vars = func_vec_vars.at(func);
        auto index_size = vec_vars[0].size();
        std::unordered_set<size_t> cur_disable_index;
        for (size_t i = 0; i < index_size - 1; ++i) {
          if (disable_fuse_index_.count(func) && disable_fuse_index_.at(func).count(i)) {
            continue;
          }
          for (auto args : vec_vars) {
            auto var_pair = VarPair(args[i], args[i + 1]);
            if (!enable_fuse_var_pairs_.count(var_pair)) {
              cur_disable_index.insert(i);
              disable_fuse_index_[func].insert(i);
              break;
            }
          }
        }
        for (auto i : cur_disable_index) {
          for (auto args : vec_vars) {
            auto var_pair = VarPair(args[i], args[i + 1]);
            enable_fuse_var_pairs_.erase(var_pair);
          }
        }
      }
    } while (enable_fuse_var_pairs_.size() != pre_enable_fuse_var_pairs.size());
  }

  std::unordered_set<size_t> GetFuseIndexByArgs(const FunctionRef &func, const Array<Expr> &args) {
    std::unordered_set<size_t> res;
    if (args.size() <= 1) {
      return res;
    }
    for (size_t i = 0; i < args.size(); ++i) {
      if (fused_vars_.count(args[i].as<Variable>())) {
        CHECK(i < args.size() - 1);
        auto var_pair = VarPair(args[i].as<Variable>(), args[i + 1].as<Variable>());
        CHECK(fused_var_pairs_.count(var_pair))
          << "fail to find fused info " << VarPairStr(var_pair) << "for args of " << func;
        res.insert(i);
        i += 1;
      }
    }
    return res;
  }

  std::unordered_set<size_t> GetFuseIndex(const std::vector<std::unordered_set<VarPair, PairHash>> &index_var_pairs) {
    std::unordered_set<size_t> fuse_index;
    for (size_t i = 0; i < index_var_pairs.size() - 1; ++i) {
      if (!index_var_pairs[i].empty()) {
        auto fused = true;
        for (auto &var_pair : index_var_pairs[i]) {
          if (!enable_fuse_var_pairs_.count(var_pair) ||
              (!fused_var_pairs_.count(var_pair) &&
               (fused_vars_.count(var_pair.first) || fused_vars_.count(var_pair.second)))) {
            fused = false;
            break;
          } else {
            fused_var_pairs_.insert(var_pair);
            fused_vars_.insert(var_pair.first);
            fused_vars_.insert(var_pair.second);
          }
        }
        if (fused) {
          fuse_index.insert(i);
          for (auto &var_pair : index_var_pairs[i + 1]) {
            enable_fuse_var_pairs_.erase(var_pair);
          }
          i += 1;
        } else {
          for (auto &var_pair : index_var_pairs[i]) {
            enable_fuse_var_pairs_.erase(var_pair);
          }
        }
      }
    }
    return fuse_index;
  }

  // Add var of axis to the fused var info (fused_var_pairs_ and fused_vars_)
  void AddFusedVars(const Array<IterVar> &axis, const std::unordered_set<size_t> &fuse_index) {
    for (auto index : fuse_index) {
      auto var1 = axis[index]->var.get();
      CHECK_LT(index + 1, axis.size());
      auto var2 = axis[index + 1]->var.get();
      fused_var_pairs_.insert(VarPair(var1, var2));
      fused_vars_.insert(var1);
      fused_vars_.insert(var2);
    }
  }

  void VisitArray(const Array<Expr> &arr) {
    for (auto &e : arr) {
      Visit(e);
    }
  }

  static std::string DumpFuncFuseIndexVarPairInfo(
    const std::unordered_map<FunctionRef, std::vector<std::unordered_set<VarPair, PairHash>>, ExprHash, ExprEqual>
      &func_fuse_index_var_pair) {
    std::stringstream info;
    for (const auto &kv : func_fuse_index_var_pair) {
      auto func = kv.first;
      auto fuse_index_var_pair = kv.second;
      info << func << ": ";
      for (size_t i = 0; i < fuse_index_var_pair.size(); ++i) {
        if (fuse_index_var_pair[i].empty()) {
          continue;
        }
        info << i << ", ";
        for (const auto &var_pair : fuse_index_var_pair[i]) {
          info << VarPairStr(var_pair) << "; ";
        }
      }
      info << std::endl;
    }
    return info.str();
  }

  static std::string DumpFuncFuseIndexInfo(
    const std::unordered_map<FunctionRef, std::unordered_set<size_t>, ExprHash, ExprEqual> &func_fuse_index) {
    std::stringstream info;
    for (const auto &kv : func_fuse_index) {
      auto func = kv.first;
      info << func << ": ";
      auto fuse_index = kv.second;
      info << "[";
      for (const auto &index : fuse_index) {
        info << index << ",";
      }
      info << "]";
      info << std::endl;
    }
    return info.str();
  }
};

bool UpdateBinds(const std::unordered_map<FunctionRef, FunctionRef, ExprHash, ExprEqual> &func_remap,
                 const Array<NodeRef> &arg_list, const Map<Tensor, Buffer> &binds, Array<NodeRef> &new_arg_list,
                 Map<Tensor, Buffer> &new_binds) {
  Map<Buffer, Buffer> buffer_map;
  for (auto kv : binds) {
    auto tensor = kv.first;
    auto buffer = kv.second;
    FunctionRef func = tensor->op;
    if (!func_remap.count(func)) {
      new_binds.Set(tensor, buffer);
      continue;
    }
    if (tensor->shape != buffer->shape) {
      LOG(INFO) << "For tensor " << tensor << ", tensor and buffer with different "
                << "shape that is not supported!" << std::endl;
      return false;
    }
    const auto &new_func = func_remap.at(func);
    auto new_tensor = Downcast<Operation>(new_func).output(tensor->value_index);
    auto new_buffer =
      BufferNode::make(buffer->data, buffer->dtype, new_tensor->shape, buffer->strides, buffer->elem_offset,
                       buffer->name, buffer->scope, buffer->data_alignment, buffer->offset_factor, buffer->buffer_type);
    new_binds.Set(new_tensor, new_buffer);
    buffer_map.Set(buffer, new_buffer);
  }

  for (const auto &arg : arg_list) {
    if (auto buffer_node = arg.as<BufferNode>()) {
      auto buffer = GetRef<Buffer>(buffer_node);
      if (buffer_map.count(buffer)) {
        new_arg_list.push_back(buffer_map[buffer]);
      } else {
        new_arg_list.push_back(buffer);
      }
    } else {
      // arg_list must be Buffer or Var
      CHECK(arg.as<Variable>());
      new_arg_list.push_back(arg);
    }
  }
  CHECK_EQ(arg_list.size(), new_arg_list.size());
  return true;
}

// For the first time run:
// input_1_red(ax0, ax1) ---> input_1_red(cc0); input_1(ax0, ax1, k2, k3, k4) ---> (cc0, cc1, k4)
// For the second time run:
// input_1(cc0, cc1, k4) ---> (cc0, cc2)
class FuseAxisMutate : public IRMutator {
 public:
  explicit FuseAxisMutate(const FuseAxisVisit &fuse_axis_visit, int fuse_var_index) {
    func_fuse_index_ = fuse_axis_visit.func_fuse_index_;
    func_reduce_fuse_index_ = fuse_axis_visit.func_reduce_fuse_index_;
    fused_vars_ = fuse_axis_visit.fused_vars_;
    fused_var_pairs_ = fuse_axis_visit.fused_var_pairs_;
    fuse_var_index_ = fuse_var_index;
  }

  ~FuseAxisMutate() override = default;

  Array<NodeRef> Run(const Stmt &stmt, const Array<NodeRef> &arg_list, const Map<Tensor, Buffer> &extern_buffer) {
    if (func_fuse_index_.empty()) {
      return Array<NodeRef>({stmt, arg_list, extern_buffer});
    }
    auto new_stmt = Mutate(stmt);
    Array<NodeRef> new_arg_list;
    Map<Tensor, Buffer> new_extern_buffer;
    bool update_success = UpdateBinds(func_remap_, arg_list, extern_buffer, new_arg_list, new_extern_buffer);
    if (!update_success) {
      LOG(ERROR) << "As fail to update_success, the FuseAxis pass may be didn't work!";
      return Array<NodeRef>({stmt, arg_list, extern_buffer});
    }
    return Array<NodeRef>({new_stmt, new_arg_list, new_extern_buffer});
  }

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) override {
    if (op->attr_key == air::ir::attr::reduce_update) {
      auto iter_vars = Downcast<Array<IterVar>>(op->node);
      if (axis_remap_.count(iter_vars)) {
        auto new_iter_vars = axis_remap_.at(iter_vars);
        auto ret = Mutate(op->body);
        return AttrStmt::make(new_iter_vars, op->attr_key, op->value, ret);
      }
    } else if (auto compute_op = op->node.as<ComputeOpNode>()) {
      auto func = GetRef<Operation>(compute_op);
      auto new_axis = compute_op->axis;
      if (func_fuse_index_.count(func) || func_reduce_fuse_index_.count(func)) {
        std::unordered_set<size_t> fused_index;
        if (func_fuse_index_.count(func)) {
          fused_index = func_fuse_index_.at(func);
        }
        new_axis = FuseFuncAxis(compute_op->axis, fused_index);
        std::unordered_set<size_t> reduce_fused_index;
        if (func_reduce_fuse_index_.count(func)) {
          reduce_fused_index = func_reduce_fuse_index_.at(func);
        }
        FuseFuncAxis(compute_op->reduce_axis, reduce_fused_index);
      }
      auto new_body = MutateArray(compute_op->body);
      if (!new_axis.same_as(compute_op->axis) || !new_body.same_as(compute_op->body)) {
        auto new_func =
          ComputeOpNode::make(GetFuseName(compute_op->name), compute_op->tag, compute_op->attrs, new_axis, new_body);
        func_remap_[func] = new_func;
        auto ret = Mutate(op->body);
        return AttrStmt::make(new_func, op->attr_key, op->value, ret);
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Realize *op, const Stmt &s) override {
    if (func_remap_.count(op->func)) {
      auto new_func = func_remap_.at(op->func);
      Region new_bouds;
      if (func_fuse_index_.count(op->func)) {
        auto fused_args_index = func_fuse_index_.at(op->func);
        for (size_t i = 0; i < op->bounds.size(); ++i) {
          if (fused_args_index.count(i)) {
            auto bound_1 = op->bounds[i];
            auto bound_2 = op->bounds[i + 1];
            auto new_bound =
              Range::make_by_min_extent(bound_1->min * bound_2->extent, bound_1->extent * bound_2->extent);
            new_bouds.push_back(new_bound);
            i += 1;
          } else {
            new_bouds.push_back(op->bounds[i]);
          }
        }
      } else {
        new_bouds = op->bounds;
      }
      auto new_condition = Mutate(op->condition);
      auto new_body = Mutate(op->body);
      return Realize::make(new_func, op->value_index, op->type, new_bouds, new_condition, new_body);
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const ProducerConsumer *op, const Stmt &s) override {
    if (func_remap_.count(op->func)) {
      auto new_func = func_remap_.at(op->func);
      auto ret = Mutate(op->body);
      return ProducerConsumer::make(new_func, op->is_producer, ret);
    }
    return IRMutator::Mutate_(op, s);
  }

  Expr Mutate_(const Reduce *op, const Expr &e) override {
    if (axis_remap_.count(op->axis)) {
      auto new_axis = axis_remap_[op->axis];
      Array<Expr> new_source = MutateArray(op->source);
      Expr new_cond = Mutate(op->condition);
      return Reduce::make(op->combiner, new_source, new_axis, new_cond, op->value_index);
    }
    return IRMutator::Mutate_(op, e);
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    auto next_for_op = op->body.as<For>();
    if (next_for_op) {
      auto var_pair = VarPair(op->loop_var.get(), next_for_op->loop_var.get());
      if (fused_var_pairs_.count(var_pair)) {
        auto new_var = FuseVar(var_pair);
        auto ret = Mutate(next_for_op->body);
        return For::make(new_var, op->min * next_for_op->extent, op->extent * next_for_op->extent, op->for_type,
                         op->device_api, ret);
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    if (func_remap_.count(op->func)) {
      auto new_func = func_remap_.at(op->func);
      Array<Expr> new_args;
      if (func_fuse_index_.count(op->func)) {
        new_args = FuseArgs(op->func, op->args, func_fuse_index_.at(op->func));
      } else {
        new_args = op->args;
      }
      auto new_value = IRMutator::Mutate(op->value);
      return Provide::make(new_func, op->value_index, new_value, new_args);
    }
    return IRMutator::Mutate_(op, s);
  }

  Expr Mutate_(const Call *op, const Expr &e) final {
    if (!op->func.defined()) {
      return IRMutator::Mutate_(op, e);
    }
    if (func_fuse_index_.count(op->func) || func_remap_.count(op->func)) {
      std::unordered_set<size_t> fuse_axis_index;
      if (func_fuse_index_.count(op->func)) {
        fuse_axis_index = func_fuse_index_.at(op->func);
      }
      FunctionRef new_func;
      Array<Expr> new_args;
      if (func_remap_.count(op->func)) {
        new_func = func_remap_.at(op->func);
      } else if (auto place_op = op->func.as<PlaceholderOpNode>()) {
        auto new_shape = GetFuseShape(place_op->shape, fuse_axis_index);
        new_func = PlaceholderOpNode::make(GetFuseName(place_op->name), new_shape, place_op->dtype);
        func_remap_[op->func] = new_func;
      } else if (auto compute_op = op->func.as<ComputeOpNode>()) {
        auto new_axis = FuseFuncAxis(compute_op->axis, fuse_axis_index);
        CHECK_EQ(compute_op->reduce_axis.size(), 0);
        auto body = MutateArray(compute_op->body);
        new_func =
          ComputeOpNode::make(GetFuseName(compute_op->name), compute_op->tag, compute_op->attrs, new_axis, body);
        func_remap_[op->func] = new_func;
      } else {
        LOG(FATAL) << op->func << " not been proccessed!" << std::endl;
      }
      new_args = FuseArgs(op->func, op->args, fuse_axis_index);
      return Call::make(op->type, GetFuseName(op->name), new_args, op->call_type, new_func, op->value_index);
    } else if (auto compute_op = op->func.as<ComputeOpNode>()) {
      auto new_body = MutateArray(compute_op->body);
      if (!new_body.same_as(compute_op->body)) {
        auto new_func = ComputeOpNode::make(GetFuseName(compute_op->name), compute_op->tag, compute_op->attrs,
                                            compute_op->axis, new_body);
        return Call::make(op->type, GetFuseName(op->name), op->args, op->call_type, new_func, op->value_index);
      }
    }
    return IRMutator::Mutate_(op, e);
  }

  Expr Mutate_(const Variable *op, const Expr &e) final {
    if (fused_vars_.count(op)) {
      LOG(FATAL) << "var: " << op->name_hint << " should be fused!" << std::endl;
    }
    return IRMutator::Mutate_(op, e);
  }

  int fuse_var_index_{0};

 private:
  std::string new_var_prefix = "cc";
  std::unordered_map<FunctionRef, std::unordered_set<size_t>, ExprHash, ExprEqual> func_fuse_index_;
  std::unordered_map<FunctionRef, std::unordered_set<size_t>, ExprHash, ExprEqual> func_reduce_fuse_index_;
  std::unordered_set<const Variable *> fused_vars_;
  std::unordered_set<VarPair, PairHash> fused_var_pairs_;
  std::unordered_map<VarPair, Var, PairHash> var_pair_remap_;
  std::unordered_map<std::pair<std::string, std::string>, std::string, PairHash> pair_name_remap_;
  std::unordered_map<std::string, std::string> name_remap_;
  std::unordered_map<IterVarPair, IterVar, PairHash> itervar_pair_remap_;
  std::unordered_map<Array<IterVar>, Array<IterVar>, ArrayIterVarHash> axis_remap_;
  std::unordered_map<FunctionRef, FunctionRef, ExprHash, ExprEqual> func_remap_;

  static Array<Expr> GetFuseShape(const Array<Expr> &shape, const std::unordered_set<size_t> &fuse_args_index) {
    Array<Expr> fuse_shape;
    for (size_t i = 0; i < shape.size(); ++i) {
      if (fuse_args_index.count(i)) {
        fuse_shape.push_back(shape[i] * shape[i + 1]);
        i += 1;
      } else {
        fuse_shape.push_back(shape[i]);
      }
    }
    return fuse_shape;
  }

  Array<Expr> MutateArray(const Array<Expr> &arr) {
    return air::ir::UpdateArray(arr, [this](const Expr &e) { return this->Mutate(e); });
  }

  Array<IterVar> FuseFuncAxis(const Array<IterVar> &axis, const std::unordered_set<size_t> &fused_index) {
    if (axis_remap_.count(axis)) {
      return axis_remap_.at(axis);
    }
    if (fused_index.empty()) {
      return axis;
    }
    Array<IterVar> new_axis;
    for (size_t i = 0; i < axis.size(); ++i) {
      if (fused_index.count(i)) {
        auto new_iter_var = FuseIterVar(axis[i], axis[i + 1]);
        new_axis.push_back(new_iter_var);
        i += 1;
      } else {
        new_axis.push_back(axis[i]);
      }
    }
    axis_remap_[axis] = new_axis;
    return new_axis;
  }

  IterVar FuseIterVar(const IterVar &iter_var1, const IterVar &iter_var2) {
    auto iter_var_pair = IterVarPair(iter_var1.as<IterVarNode>(), iter_var2.as<IterVarNode>());
    if (itervar_pair_remap_.count(iter_var_pair)) {
      return itervar_pair_remap_.at(iter_var_pair);
    }
    auto var_pair = VarPair(iter_var1->var.as<Variable>(), iter_var2->var.as<Variable>());
    auto fuse_var = FuseVar(var_pair);
    auto new_dom = Range::make_by_min_extent(iter_var1->dom->min * iter_var2->dom->extent,
                                             iter_var1->dom->extent * iter_var2->dom->extent);
    IterVar new_iter_var = IterVarNode::make(new_dom, fuse_var, iter_var1->iter_type, iter_var1->thread_tag);
    var_pair_remap_[var_pair] = fuse_var;
    itervar_pair_remap_[iter_var_pair] = new_iter_var;
    return new_iter_var;
  }

  Array<Expr> FuseArgs(const FunctionRef &func, const Array<Expr> &args,
                       const std::unordered_set<size_t> &fused_index) {
    if (fused_index.empty()) {
      return args;
    }
    Array<Expr> new_args;
    for (size_t i = 0; i < args.size(); ++i) {
      if (fused_index.count(i)) {
        CHECK_LT(i, args.size() - 1);
        auto var_pair = VarPair(args[i].as<Variable>(), args[i + 1].as<Variable>());
        CHECK(var_pair_remap_.count(var_pair))
          << "fail to find fuse info for args " << VarPairStr(var_pair) << " of func " << func << std::endl;
        Expr new_var = var_pair_remap_.at(var_pair);
        new_args.push_back(new_var);
        i += 1;
      } else {
        new_args.push_back(args[i]);
      }
    }
    return new_args;
  }

  Var FuseVar(const VarPair &var_pair) {
    if (var_pair_remap_.count(var_pair)) {
      return var_pair_remap_.at(var_pair);
    } else {
      auto fuse_var_name = GetFuseName(var_pair.first->name_hint, var_pair.second->name_hint);
      auto fuse_var = Var(fuse_var_name);
      var_pair_remap_[var_pair] = fuse_var;
      return fuse_var;
    }
  }

  std::string GetFuseName(const std::string &name1, const std::string &name2) {
    auto pair_name = std::pair<std::string, std::string>(name1, name2);
    if (pair_name_remap_.count(pair_name)) {
      return pair_name_remap_.at(pair_name);
    } else {
      auto new_name = new_var_prefix + std::to_string(fuse_var_index_++);
      pair_name_remap_[pair_name] = new_name;
      return new_name;
    }
  }

  std::string GetFuseName(const std::string &name_) {
    std::string new_name_prefix;
    if (DEBUG_FUSE_AXIS) {
      new_name_prefix = "fuse_";
    }
    if (name_remap_.count(name_)) {
      return name_remap_.at(name_);
    } else {
      auto new_name = new_name_prefix + name_;
      name_remap_[name_] = new_name;
      return new_name;
    }
  }
};

Array<NodeRef> FuseAxis(Stmt stmt, const Array<NodeRef> &arg_list, const Map<Tensor, Buffer> &extern_buffer) {
  if (DEBUG_FUSE_AXIS) {
    LOG(INFO) << stmt << std::endl;
  }

  // prevent infinite loop
  auto max_fuse_times = 100;

  auto fuse_times = 1;
  auto fuse_var_start_index = 0;
  Stmt stmt_ = stmt;
  Array<NodeRef> arg_list_ = arg_list;
  Map<Tensor, Buffer> extern_buffer_ = extern_buffer;
  do {
    // Visit
    auto fuse_axis_visit = FuseAxisVisit();
    fuse_axis_visit.Run(stmt_);
    if (DEBUG_FUSE_AXIS) {
      LOG(INFO) << fuse_axis_visit.DumpInfo() << std::endl;
    }
    // Mutate
    auto fuse_mutate = FuseAxisMutate(fuse_axis_visit, fuse_var_start_index);
    auto res = fuse_mutate.Run(stmt_, arg_list_, extern_buffer_);
    CHECK_EQ(res.size(), 3);
    auto new_stmt = Downcast<Stmt>(res[0]);
    if (DEBUG_FUSE_AXIS) {
      LOG(INFO) << fuse_times << std::endl << new_stmt << std::endl;
    }
    if (new_stmt.same_as(stmt_)) {
      return res;
    }
    if (fuse_times >= max_fuse_times) {
      LOG(FATAL) << "Too many times to fuse for the FuseAxis pass!" << std::endl;
    }
    // update var
    stmt_ = new_stmt;
    arg_list_ = Downcast<Array<NodeRef>>(res[1]);
    extern_buffer_ = Downcast<Map<Tensor, Buffer>>(res[2]);
    fuse_times++;
    fuse_var_start_index = fuse_mutate.fuse_var_index_;
  } while (true);
}
}  // namespace ir
}  // namespace akg
