/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include <tvm/ir_visitor.h>
#include <tvm/operation.h>
#include <tvm/schedule_pass.h>
#include <tvm.h>
#include <dmlc/common.h>

#include "pass/utils.h"

struct FuncIndex {
  air::ir::FunctionRef f;
  size_t arg_index;

  inline bool operator==(const FuncIndex &other) const { return f == other.f && arg_index == other.arg_index; }
  inline std::string GetStr() const {
    std::ostringstream os;
    os << f->func_name() << "_arg_" << arg_index;
    return os.str();
  }
};

namespace std {
template <>
struct hash<FuncIndex> {
  std::size_t operator()(const FuncIndex &k) const {
    size_t lhs = ::air::NodeHash()(k.f);
    size_t rhs = k.arg_index;
    return dmlc::HashCombine(lhs, rhs);
  }
};
}  // namespace std

namespace akg {
namespace schedule {

#define DEBUG_AUTO_FUSE 0

class FuseOpAxis {
 public:
  FuseOpAxis(const Schedule &sch, const std::unordered_map<IterVar, std::unordered_set<size_t>> &axis_reduce_group_ids,
             const std::vector<size_t> &split_config, std::vector<size_t> &split_index)
      : sch_(sch),
        axis_reduce_group_ids_(axis_reduce_group_ids),
        split_config_(split_config),
        split_index_(split_index) {}

  void Run() {
    for (auto op : sch_->outputs) {
      TraverseCheck(op);
    }
    if (!enable_fuse) {
      return;
    }
    for (auto op : sch_->outputs) {
      TraverseFuse(op);
    }
  }

  void TraverseCheck(const Operation &op) {
    if (!op.defined() || op->IsInstance<PlaceholderOpNode>() || check_visited.count(op)) {
      return;
    }
    RunCheck(op);
    for (auto t : op->InputTensors()) {
      TraverseCheck(t->op);
    }
    check_visited.insert(op);
  }

  void TraverseFuse(const Operation &op) {
    if (!op.defined() || op->IsInstance<PlaceholderOpNode>() || fuse_visited.count(op)) {
      return;
    }
    RunFuse(op);
    for (auto t : op->InputTensors()) {
      TraverseFuse(t->op);
    }
    fuse_visited.insert(op);
  }

 private:
  Schedule sch_;
  bool enable_fuse{true};
  std::unordered_set<Operation> check_visited;
  std::unordered_set<Operation> fuse_visited;
  std::unordered_map<IterVar, std::unordered_set<size_t>> axis_reduce_group_ids_;
  std::vector<size_t> split_config_;
  std::vector<size_t> &split_index_;

  void RunCheck(const Operation &op) {
    // skip op that has been inlined
    if (sch_[op]->attach_type == air::kInline) {
      return;
    }
    auto tensor = op.output(0);
    // reduce axis of op should be same
    auto compute_op = sch_[tensor]->op.as<air::ComputeOpNode>();
    CHECK_NOTNULL(compute_op);
    if (compute_op->reduce_axis.size() > 1) {
      if (SplitAxisToGroups(compute_op->reduce_axis).size() > 1) {
        LOG(WARNING) << "The scenes where the reduce_axis cannot be fused into one axis currently not supported."
                     << std::endl;
        enable_fuse = false;
      }
    }
  }
  void RunFuse(const Operation &op) {
    // skip op that has been inlined
    if (sch_[op]->attach_type == air::kInline) {
      return;
    }
    auto tensor = op.output(0);
    // fuse reduce axis of op
    auto compute_op = sch_[tensor]->op.as<air::ComputeOpNode>();
    CHECK_NOTNULL(compute_op);
    if (compute_op->reduce_axis.size() > 1) {
      IterVar fused_reduce_axis;
      sch_[tensor].fuse(compute_op->reduce_axis, &fused_reduce_axis);
      // reduce by the fused_reduce_axis
      auto data_rf = sch_.rfactor(tensor, fused_reduce_axis);
      if (data_rf.size() == 1) {
        sch_[data_rf[0]].compute_inline();
      }
    }
    // fuse axis of op
    compute_op = sch_[tensor]->op.as<air::ComputeOpNode>();
    CHECK_NOTNULL(compute_op);
    if (compute_op->axis.size() > 1) {
      auto axis_groups = SplitAxisToGroups(compute_op->axis);
      // As the fuse can only fuse iterVar that are consecutive between each other,
      // the axis that needs the fuse must be consecutive.
      // To make the axis are consecutive by reorder.
      Array<IterVar> axis_order;
      for (const auto &axis_group : axis_groups) {
        for (const auto &ax : axis_group) {
          axis_order.push_back(ax);
        }
      }
      sch_[tensor].reorder(axis_order);
      for (const auto &axis_group : axis_groups) {
        IterVar fused_axis;
        sch_[tensor].fuse(axis_group, &fused_axis);
      }
    }
  }

  std::vector<Array<IterVar>> SplitAxisToGroups(const Array<IterVar> &axis) {
    std::vector<Array<IterVar>> groups;
    if (!split_config_.empty()) {
      split_index_.clear();
      split_index_ = split_config_;
      for (size_t i = 0; i < split_index_.size() - 1; ++i) {
        Array<IterVar> cur_axis_group;
        for (auto j = split_index_[i]; j < split_index_[i + 1]; ++j) {
          cur_axis_group.push_back(axis[j]);
        }
        groups.push_back(cur_axis_group);
      }
      return groups;
    }

    std::unordered_map<IterVar, std::unordered_set<size_t>> reduce_group_ids;
    for (auto ax : axis) {
      if (axis_reduce_group_ids_.count(ax)) {
        reduce_group_ids[ax] = axis_reduce_group_ids_.at(ax);
      } else {
        reduce_group_ids[ax] = std::unordered_set<size_t>();
      }
    }
    for (auto ax : axis) {
      auto group_id = groups.size();
      for (size_t j = 0; j < groups.size(); ++j) {
        if (reduce_group_ids[ax] == reduce_group_ids[groups[j][0]]) {
          group_id = j;
          break;
        }
      }
      if (group_id == groups.size()) {
        Array<IterVar> cur_axis_group = {ax};
        groups.push_back(cur_axis_group);
      } else {
        groups[group_id].push_back(ax);
      }
    }
    return groups;
  }
};

class FuseCheck {
 public:
  FuseCheck(const Schedule &sch, const std::vector<size_t> &split_config) : sch_(sch), split_config_(split_config) {}

  bool NeedToFuse() {
    if (!split_config_.empty()) return true;
    if (HasExternOp()) {
      return false;
    }
    ReduceCheck();
    if (has_reduce_ && !has_matmul_) {
      OutputBroadcastRecord();
      return true;
    }
    return false;
  }

  bool HasExternOp() {
    for (const auto &s : sch_->stages) {
      auto op = s->op;
      CHECK(op.defined());
      if (op.as<air::ExternOpNode>()) {
        return true;
      }
    }
    return false;
  }

  void ReduceCheck() {
    for (const auto &s : sch_->stages) {
      auto op = s->op;
      CHECK(op.defined());
      auto compute_op = op.as<air::ComputeOpNode>();
      if (compute_op && !compute_op->reduce_axis.empty()) {
        has_reduce_ = true;
        if (IsMatmul(op)) {
          has_matmul_ = true;
        }
      }
    }
  }

  void OutputBroadcastRecord() {
    GetOutputBroadcastPair();
    if (!output_broadcast_pairs_.empty()) {
      has_output_broadcast_ = true;
    }
    // For scenarios where there is a broadcast relationship between outputs,
    // the current solution requires the compute_at process after fuse.
    compute_at_pairs_ = output_broadcast_pairs_;
  }

  std::unordered_map<Operation, Operation> compute_at_pairs_;

 private:
  Schedule sch_;
  std::vector<size_t> split_config_;
  bool has_reduce_{false};
  bool has_matmul_{false};
  bool has_output_broadcast_{false};
  std::unordered_map<Operation, std::unordered_set<Operation>> op_input_ops;
  std::unordered_map<Operation, Operation> output_broadcast_pairs_;

  bool IsMatmul(const Operation &op) {
    // judge according to the tag of the op
    if (op->tag == "dense" || op->tag == "batch_matmul" || op->tag == "matmul") {
      return true;
    }

    // judge according to the format of the compute op
    auto compute_op = op.as<ComputeOpNode>();
    auto reduce = compute_op->body[0].as<Reduce>();
    CHECK_NOTNULL(reduce);
    // combiner should be `lhs + rhs`
    auto combiner = reduce->combiner;
    if (combiner->lhs.size() != 1 || combiner->rhs.size() != 1 || combiner->result.size() != 1 ||
        !combiner->result[0].as<Add>()) {
      return false;
    }
    // size of reduce_axis should be 1
    auto reduce_axis = reduce->axis;
    if (reduce_axis.size() != 1) {
      return false;
    }
    // source should be such as: left[..., i, k] * right[..., j, k]
    auto source = reduce->source;
    if (source.size() != 1 || !source[0].as<Mul>()) {
      return false;
    }
    auto mul = source[0].as<Mul>();
    auto left = mul->a.as<Call>();
    auto right = mul->b.as<Call>();
    if (!left || !right || left->args.size() != right->args.size()) {
      return false;
    }
    auto args_size = left->args.size();
    if (args_size < 2) {
      return false;
    }
    for (size_t i = 0; i < args_size - 2; ++i) {
      if (!left->args[i].same_as(right->args[i])) {
        return false;
      }
    }
    auto reduce_var = reduce_axis[0]->var.get();
    if ((left->args[args_size - 1].as<Variable>() != reduce_var &&
         left->args[args_size - 2].as<Variable>() != reduce_var) ||
        (right->args[args_size - 1].as<Variable>() != reduce_var &&
         right->args[args_size - 2].as<Variable>() != reduce_var)) {
      return false;
    }
    return true;
  }

  void GetOutputBroadcastPair() {
    if (sch_->outputs.size() < 2) {
      return;
    }
    std::unordered_map<Operation, std::vector<Operation>> enable_output_broadcast_pairs;
    std::vector<const ComputeOpNode *> output_compute_ops;
    for (auto op : sch_->outputs) {
      if (auto compute_op = op.as<ComputeOpNode>()) {
        output_compute_ops.push_back(compute_op);
      }
    }
    for (auto compute_op : output_compute_ops) {
      if (compute_op->reduce_axis.size() > 0) {
        continue;
      }
      for (auto compute_op_other : output_compute_ops) {
        if (compute_op_other != compute_op && EnableBroadcast(compute_op, compute_op_other)) {
          enable_output_broadcast_pairs[GetRef<Operation>(compute_op)].push_back(GetRef<Operation>(compute_op_other));
        }
      }
    }
    if (enable_output_broadcast_pairs.empty()) {
      return;
    }
    GetOpInputOps();
    for (auto kv : enable_output_broadcast_pairs) {
      auto op = kv.first;
      auto enable_broadcast_ops = kv.second;
      for (auto enable_broadcast_op : enable_broadcast_ops) {
        CHECK(op_input_ops.count(enable_broadcast_op));
        if (op_input_ops[enable_broadcast_op].count(op)) {
          if (output_broadcast_pairs_.count(op)) {
            // otherwise there will be a problem of poly stuck
            LOG(WARNING) << "As output " << op->func_name() << "needs broadcast to output "
                         << output_broadcast_pairs_.at(op)->func_name() << " and output "
                         << enable_broadcast_op->func_name() << ", this scenario does not currently support the fuse."
                         << std::endl;
            output_broadcast_pairs_.clear();
            return;
          }
          output_broadcast_pairs_[op] = enable_broadcast_op;
        }
      }
    }
  }

  void GetOpInputOps() {
    for (auto stage : sch_->stages) {
      auto op = stage->op;
      std::unordered_set<Operation> input_ops;
      for (auto input_op : op->InputTensors()) {
        if (sch_[input_op]->attach_type == air::kInline) {
          CHECK(op_input_ops.count(input_op->op));
          input_ops.insert(op_input_ops[input_op->op].begin(), op_input_ops[input_op->op].end());
        } else {
          input_ops.insert(input_op->op);
        }
      }
      op_input_ops[op] = input_ops;
    }
  }

  bool EnableBroadcast(const ComputeOpNode *op1, const ComputeOpNode *op2) {
    auto axis_1 = op1->axis;
    auto axis_2 = op2->axis;
    if (axis_1.size() != axis_2.size()) {
      return false;
    }
    auto size = axis_1.size();
    bool enable_broadcast = false;
    std::vector<int64_t> axis_1_extent;
    for (size_t i = 0; i < size; ++i) {
      if (is_one(axis_1[i]->dom->extent) && !is_one(axis_2[i]->dom->extent)) {
        enable_broadcast = true;
      } else if (!AxisRangeEqual(axis_1[i], axis_2[i])) {
        return false;
      }
    }
    return enable_broadcast;
  }

  bool AxisRangeEqual(IterVar ax_1, IterVar ax_2) {
    auto ax_1_extent = GetExprIntVal(ax_1->dom->extent);
    auto ax_2_extent = GetExprIntVal(ax_2->dom->extent);
    if (ax_1_extent < 0 || ax_2_extent < 0) {
      return false;
    }
    return ax_1_extent == ax_2_extent;
  }

  int64_t GetExprIntVal(Expr expr) {
    auto expr_int = as_const_int(expr);
    auto expr_uint = as_const_uint(expr);
    // -1 indicates that the val of the expr is invalid
    int64_t expr_val = -1;
    if (expr_int) {
      expr_val = *expr_int;
    } else if (expr_uint) {
      expr_val = static_cast<int64_t>(*expr_uint);
    }
    return expr_val;
  }
};

class ComputeInfo : public IRVisitor {
 public:
  explicit ComputeInfo(const Schedule &sch) { sch_ = sch; }

  void Run() {
    for (size_t i = 0; i < sch_->stages.size(); ++i) {
      auto op = sch_->stages[i]->op;
      stage_id_ = i;
      if (auto compute_op = op.as<air::ComputeOpNode>()) {
        GetAxisInfo(compute_op->axis, compute_op->reduce_axis);
        VisitComputeOp(op);
      }
      if (DEBUG_AUTO_FUSE) {
        LOG(INFO) << " stage_id: " << stage_id_ << " op: " << op->func_name() << std::endl;
      }
    }
    GetAxisReduceGroup();
    if (DEBUG_AUTO_FUSE) {
      std::stringstream info;
      info << "==== axis_reduce_group_ids_ start" << std::endl;
      for (size_t i = 0; i < sch_->stages.size(); ++i) {
        auto op = sch_->stages[i]->op;
        if (auto compute_op = op.as<air::ComputeOpNode>()) {
          std::vector<IterVar> all_axis(compute_op->axis.begin(), compute_op->axis.end());
          for (auto ax : compute_op->reduce_axis) {
            all_axis.push_back(ax);
          }
          for (const auto &ax : all_axis) {
            if (axis_reduce_group_ids_.count(ax)) {
              info << compute_op->func_name() << ", " << ax << ": ";
              auto reduce_groups_ids = axis_reduce_group_ids_.at(ax);
              info << "[";
              for (auto id : reduce_groups_ids) {
                info << sch_->stages[id]->op->func_name() << ",";
              }
              info << "]" << std::endl;
            }
          }
        }
      }
      info << "==== axis_reduce_group_ids_ end" << std::endl;
      LOG(INFO) << info.str();
    }
  }

  std::unordered_map<IterVar, std::unordered_set<size_t>> axis_reduce_group_ids_;

 private:
  Schedule sch_;
  size_t stage_id_{0};
  std::unordered_set<const Variable *> reduce_axis_var_;
  std::unordered_set<const Variable *> axis_var_;
  std::unordered_map<const Variable *, IterVar> all_axis_var_axis_;
  Map<Var, Range> simplify_info_;
  std::vector<FuncIndex> func_index_keys_;
  std::unordered_map<FuncIndex, std::unordered_set<IterVar>> func_index_axis_;
  std::unordered_map<FuncIndex, std::unordered_set<size_t>> func_index_reduce_group_ids_;
  std::unordered_map<IterVar, std::unordered_set<FuncIndex>> axis_func_indexs_;

  void VisitComputeOp(const Operation &op) {
    auto compute_op = op.as<air::ComputeOpNode>();
    auto func_dim = compute_op->axis.size();
    for (size_t i = 0; i < func_dim; ++i) {
      FuncIndex func_index = FuncIndex{op, i};
      if (!func_index_axis_.count(func_index)) {
        func_index_keys_.push_back(func_index);
      }
      func_index_axis_[func_index].insert(compute_op->axis[i]);
    }
    GetSimplifyInfo(compute_op->axis);
    for (auto expr : compute_op->body) {
      Visit(expr);
    }
  }

  void Visit_(const Call *op) override {
    auto func = op->func;
    if (!(func.defined() && func.as<OperationNode>())) {
      return IRVisitor::Visit_(op);
    }
    auto func_dim = op->args.size();
    for (size_t i = 0; i < func_dim; ++i) {
      auto func_index = FuncIndex{func, i};
      auto arg = op->args[i];
      if (!arg.as<Variable>()) {
        arg = Simplify(arg, simplify_info_);
      }
      std::vector<const Variable *> arg_vars;
      if (!arg.as<Variable>()) {
        auto arg_var_refs = akg::ir::GetVarsInExpr(arg);
        for (auto var_ref : arg_var_refs) {
          arg_vars.push_back(var_ref.get());
        }
      } else {
        arg_vars.push_back(arg.as<Variable>());
      }

      for (auto var : arg_vars) {
        if (reduce_axis_var_.count(var) || axis_var_.count(var)) {
          CHECK(all_axis_var_axis_.count(var));
          auto ax = all_axis_var_axis_.at(var);
          if (!func_index_axis_.count(func_index)) {
            func_index_keys_.push_back(func_index);
          }
          func_index_axis_[func_index].insert(ax);
          if (reduce_axis_var_.count(var)) {
            func_index_reduce_group_ids_[func_index].insert(stage_id_);
          }
        }
      }
    }
  }

  void GetAxisInfo(const Array<IterVar> &axis, const Array<IterVar> &reduce_axis) {
    axis_var_.clear();
    reduce_axis_var_.clear();
    all_axis_var_axis_.clear();
    for (auto ax : axis) {
      auto var = ax->var.get();
      axis_var_.insert(var);
      CHECK_EQ(all_axis_var_axis_.count(var), 0);
      all_axis_var_axis_[var] = ax;
    }
    for (auto ax : reduce_axis) {
      auto var = ax->var.get();
      reduce_axis_var_.insert(var);
      CHECK_EQ(all_axis_var_axis_.count(var), 0);
      all_axis_var_axis_[var] = ax;
    }
  }

  void GetAxisReduceGroup() {
    // record map that axis to func_indexs
    for (const auto &kv : func_index_axis_) {
      auto func_index = kv.first;
      auto axis = kv.second;
      for (auto ax : axis) {
        axis_func_indexs_[ax].insert(func_index);
      }
    }
    // update reduce group by func_index
    std::unordered_map<FuncIndex, std::unordered_set<FuncIndex>> func_index_map_func_indexs;
    for (auto kv : func_index_axis_) {
      auto func_index = kv.first;
      auto axis = kv.second;
      std::unordered_set<FuncIndex> func_indexs;
      for (const auto &ax : axis) {
        auto cur_func_indexs = axis_func_indexs_.at(ax);
        func_indexs.insert(cur_func_indexs.begin(), cur_func_indexs.end());
      }
      // remove self from the map
      func_indexs.erase(func_index);
      if (!func_indexs.empty()) {
        func_index_map_func_indexs[func_index] = func_indexs;
      }
    }
    std::unordered_set<FuncIndex> last_updated;
    for (const auto &kv : func_index_reduce_group_ids_) {
      last_updated.insert(kv.first);
    }
    std::vector<FuncIndex> func_index_keys_has_map_;
    for (const auto &func_index : func_index_keys_) {
      if (func_index_map_func_indexs.count(func_index)) {
        func_index_keys_has_map_.push_back(func_index);
      }
    }
    do {
      std::unordered_set<FuncIndex> updated;
      for (const auto &func_index : func_index_keys_has_map_) {
        std::unordered_set<FuncIndex> map_func_indexs = func_index_map_func_indexs.at(func_index);
        std::unordered_set<size_t> reduce_group_ids;
        if (func_index_reduce_group_ids_.count(func_index)) {
          reduce_group_ids = func_index_reduce_group_ids_.at(func_index);
        }
        auto pre_size = reduce_group_ids.size();
        for (const auto &cur_map_func_index : map_func_indexs) {
          if (last_updated.count(cur_map_func_index)) {
            auto cur_map_reduce_group_ids = func_index_reduce_group_ids_.at(cur_map_func_index);
            reduce_group_ids.insert(cur_map_reduce_group_ids.begin(), cur_map_reduce_group_ids.end());
          }
        }
        if (reduce_group_ids.size() > pre_size) {
          func_index_reduce_group_ids_[func_index] = reduce_group_ids;
          updated.insert(func_index);
        }
      }
      last_updated = updated;
    } while (!last_updated.empty());
    // get reduce_group ids for axis
    for (const auto &kv : axis_func_indexs_) {
      auto ax = kv.first;
      auto func_indexs = kv.second;
      std::unordered_set<size_t> reduce_group_ids;
      for (const auto &func_index : func_indexs) {
        if (func_index_reduce_group_ids_.count(func_index)) {
          auto cur_reduce_group_ids = func_index_reduce_group_ids_.at(func_index);
          reduce_group_ids.insert(cur_reduce_group_ids.begin(), cur_reduce_group_ids.end());
        }
      }
      if (!reduce_group_ids.empty()) {
        axis_reduce_group_ids_[ax] = reduce_group_ids;
      }
    }
  }

  void GetSimplifyInfo(const Array<IterVar> &axis) {
    Map<Var, Range> simplify_info;
    for (const auto &iter_var : axis) {
      simplify_info.Set(iter_var->var, iter_var->dom);
    }
    simplify_info_ = simplify_info;
  }
};

class ComputeAtProcess {
 public:
  explicit ComputeAtProcess(Schedule sch) { sch_ = sch; }

  void Run(const std::unordered_map<Operation, Operation> &compute_at_pair) {
    for (auto kv : compute_at_pair) {
      auto op1 = kv.first;
      auto op2 = kv.second;
      auto leaf_iter_vars_size = sch_[op2]->leaf_iter_vars.size();
      auto compute_at_itervar = sch_[op2]->leaf_iter_vars[leaf_iter_vars_size - 1];
      sch_[op1].compute_at(sch_[op2], compute_at_itervar);
      // For the output, its is_output attribute should be set to false after the compute_at.
      sch_[op1]->is_output = false;
    }
  }

 private:
  Schedule sch_;
};

void AutoFuse(Schedule sch, const std::string &split_str, std::vector<size_t> &split_index) {
  auto split_vec = dmlc::Split(split_str, ' ');
  std::vector<size_t> split_config;
  for (const auto &c : split_vec) {
    char *endptr = nullptr;
    const int radix = 10;
    size_t split = strtol(c.c_str(), &endptr, radix);
    split_config.emplace_back(split);
  }
  auto fuse_check = FuseCheck(sch, split_config);
  if (!fuse_check.NeedToFuse()) {
    return;
  }
  auto compute_info = ComputeInfo(sch);
  compute_info.Run();
  FuseOpAxis(sch, compute_info.axis_reduce_group_ids_, split_config, split_index).Run();
  if (!fuse_check.compute_at_pairs_.empty()) {
    ComputeAtProcess(sch).Run(fuse_check.compute_at_pairs_);
  }
}
}  // namespace schedule
}  // namespace akg
