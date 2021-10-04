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

#include "common/common_util.h"
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
enum AxisType { ELEWISE, REDUCE };
using StageId = size_t;
using StageInfo = std::pair<StageId, AxisType>;
using StageInfoGroup = std::pair<std::set<StageInfo>, AxisType>;

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

inline int64_t GetAxisLen(const IterVar &axis) { return GetExprIntVal(axis->dom->extent); }

struct FuseInfo {
  std::vector<std::vector<size_t>> axis_index_groups;
  std::vector<std::vector<size_t>> reduce_axis_index_groups;
};

class FuseCheck {
 public:
  FuseCheck(const Schedule &sch, const std::vector<size_t> &split_config) : sch_(sch), split_config_(split_config) {}

  bool NeedToFuse() {
    if (HasExternOp()) {
      return false;
    }
    if (ReduceCheck() || !split_config_.empty()) {
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

  bool ReduceCheck() {
    auto reduce_need_fuse = false;
    auto max_inner_non_reduce_axis = 10240;
    for (const auto &s : sch_->stages) {
      auto op = s->op;
      CHECK(op.defined());
      auto tensor = op.output(0);
      auto compute_op = op.as<air::ComputeOpNode>();
      if (compute_op && !compute_op->reduce_axis.empty()) {
        // For the matmul, do not perform fuse
        if (IsMatmul(op)) {
          // This code will be deleted in the future:
          IterVar fused_axis;
          Array<IterVar> need_fused_axis;
          for (size_t i = 2; i < compute_op->axis.size(); ++i) {
            need_fused_axis.push_back(compute_op->axis[i - 2]);
          }
          sch_[tensor].fuse(need_fused_axis, &fused_axis);
          return false;
        }
        // Restrictions related to the Shared memory
        if (GetInnerNonReduceAxisLen(compute_op) > max_inner_non_reduce_axis) {
          return false;
        }
        reduce_need_fuse = true;
      }
    }
    return reduce_need_fuse;
  }

 private:
  Schedule sch_;
  std::vector<size_t> split_config_;

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
    auto left = akg::common::SplitCast(mul->a, compute_op->output_dtype(0)).as<Call>();
    auto right = akg::common::SplitCast(mul->b, compute_op->output_dtype(0)).as<Call>();
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

  int64_t GetInnerNonReduceAxisLen(const air::ComputeOpNode *reduce_op) {
    auto reduce = reduce_op->body[0].as<Reduce>();
    CHECK_NOTNULL(reduce);
    auto reduce_axis = reduce_op->reduce_axis;
    auto source = reduce->source[0];
    auto source_call = source.as<Call>();
    CHECK(source_call->func.defined());
    auto source_tensor = Downcast<Operation>(source_call->func).output(0);
    auto source_shape = source_tensor->shape;
    CHECK_EQ(source_call->args.size(), source_shape.size());
    std::unordered_set<std::string> reduce_axis_names;
    for (auto ax : reduce_axis) {
      reduce_axis_names.insert(ax->var->name_hint);
    }
    // If the last axis is a reduce axis, return 0
    if (auto var = source_call->args[source_call->args.size() - 1].as<Variable>()) {
      if (reduce_axis_names.count(var->name_hint)) {
        return 0;
      }
    }
    return GetExprIntVal(source_shape[source_shape.size() - 1]);
  }
};

class FuseVisit : public IRVisitor {
 public:
  explicit FuseVisit(const Schedule &sch, const std::unordered_map<Operation, Operation> &compute_at_pairs,
                     const std::vector<size_t> &split_config, std::vector<size_t> &split_index)
      : sch_(sch), compute_at_pairs_(compute_at_pairs), split_config_(split_config), split_index_(split_index) {}

  void Run() {
    for (size_t i = 0; i < sch_->stages.size(); ++i) {
      auto op = sch_->stages[i]->op;
      stage_id_ = i;
      if (auto compute_op = op.as<air::ComputeOpNode>()) {
        GetAxisInfo(compute_op->axis, compute_op->reduce_axis);
        VisitComputeOp(op);
      }
    }
    GetAxisStageInfoGroup();
    GetOpFuseInfo();
  }

  std::unordered_map<Operation, FuseInfo> OpFuseInfo() { return op_fuse_info_; }

 private:
  Schedule sch_;
  std::unordered_map<Operation, Operation> compute_at_pairs_;
  std::vector<size_t> split_config_;
  std::vector<size_t> &split_index_;
  size_t stage_id_{0};
  std::unordered_set<const Variable *> reduce_axis_var_;
  std::unordered_set<const Variable *> axis_var_;
  std::unordered_map<const Variable *, IterVar> var2axis_;
  std::unordered_map<IterVar, StageInfo> axis_stage_info_;
  Map<Var, Range> simplify_info_;
  std::unordered_map<IterVar, std::vector<FuncIndex>> axis_func_indexs_;
  std::vector<IterVar> axis_ordered_;
  std::unordered_map<IterVar, StageInfoGroup> axis_stage_info_group_;
  std::unordered_map<Operation, FuseInfo> op_fuse_info_;

  void VisitComputeOp(const Operation &op) {
    auto compute_op = op.as<air::ComputeOpNode>();
    for (size_t i = 0; i < compute_op->axis.size(); ++i) {
      auto axis = compute_op->axis[i];
      // The axis with length 1 is more special, so skip it. On the one hand, it may be the broadcast axis;
      // on the other hand, its subscript in func may be a constant 0 instead of a variable.
      if (GetAxisLen(axis) == 1) {
        continue;
      }
      auto func_index = FuncIndex{op, i};
      if (!axis_func_indexs_.count(axis)) {
        axis_ordered_.push_back(axis);
      }
      axis_func_indexs_[axis].push_back(func_index);
    }
    GetSimplifyInfo(compute_op->axis, compute_op->reduce_axis);
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
      // get var from arg of call
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
      // get axis from the var and build relations between the axis and the func_index
      for (auto var : arg_vars) {
        if (reduce_axis_var_.count(var) || axis_var_.count(var)) {
          CHECK(var2axis_.count(var));
          auto ax = var2axis_.at(var);
          if (!axis_func_indexs_.count(ax)) {
            axis_ordered_.push_back(ax);
          }
          axis_func_indexs_[ax].push_back(func_index);
        }
      }
    }
  }

  void GetAxisInfo(const Array<IterVar> &axis, const Array<IterVar> &reduce_axis) {
    // get relations between axis and var
    axis_var_.clear();
    reduce_axis_var_.clear();
    var2axis_.clear();
    for (auto ax : axis) {
      if (GetAxisLen(ax) == 1) {
        continue;
      }
      auto var = ax->var.get();
      axis_var_.insert(var);
      CHECK_EQ(var2axis_.count(var), 0);
      var2axis_[var] = ax;
    }
    for (auto ax : reduce_axis) {
      if (GetAxisLen(ax) == 1) {
        continue;
      }
      auto var = ax->var.get();
      reduce_axis_var_.insert(var);
      CHECK_EQ(var2axis_.count(var), 0);
      var2axis_[var] = ax;
    }
    // get relations between axis and stage
    auto stage_info = StageInfo{stage_id_, ELEWISE};
    for (auto ax : axis) {
      CHECK_EQ(axis_stage_info_.count(ax), 0);
      axis_stage_info_[ax] = stage_info;
    }
    stage_info = StageInfo{stage_id_, REDUCE};
    for (auto ax : reduce_axis) {
      axis_stage_info_[ax] = stage_info;
    }
  }

  FuncIndex GetRoot(const FuncIndex &func_index, std::unordered_map<FuncIndex, FuncIndex> &func_index_map) {
    auto root = func_index;
    std::unordered_set<FuncIndex> visited;
    while (func_index_map.count(root)) {
      visited.insert(root);
      root = func_index_map.at(root);
      if (visited.count(root)) {
        LOG(FATAL) << "There is a directed circle for node: " << root.GetStr();
      }
    }
    if (!(root == func_index)) {
      func_index_map[func_index] = root;
    }
    return root;
  }

  void MapNodeToRoot(const FuncIndex &node, const FuncIndex &root,
                     std::unordered_map<FuncIndex, FuncIndex> &func_index_map,
                     std::unordered_map<FuncIndex, std::unordered_set<IterVar>> &func_index_map_axes) {
    auto node_root = GetRoot(node, func_index_map);
    if (node_root == root) {
      return;
    }
    func_index_map[node_root] = root;
    if (func_index_map_axes.count(node_root)) {
      auto node_axes = func_index_map_axes.at(node_root);
      func_index_map_axes[root].insert(node_axes.begin(), node_axes.end());
      func_index_map_axes.erase(node_root);
    }
  }

  void GetAxisGroups(std::vector<std::unordered_set<IterVar>> &axis_groups) {
    std::unordered_map<FuncIndex, FuncIndex> func_index_map;
    std::unordered_map<FuncIndex, std::unordered_set<IterVar>> func_index_map_axes;
    for (const auto &ax : axis_ordered_) {
      CHECK(axis_func_indexs_.count(ax));
      auto func_indexs = axis_func_indexs_.at(ax);
      if (func_indexs.empty()) {
        continue;
      }
      auto func_index_first = func_indexs[0];
      auto func_index_first_root = GetRoot(func_index_first, func_index_map);
      func_index_map_axes[func_index_first_root].insert(ax);
      for (size_t i = 1; i < func_indexs.size(); ++i) {
        auto func_index = func_indexs[i];
        auto func_index_root = GetRoot(func_index, func_index_map);
        MapNodeToRoot(func_index_root, func_index_first_root, func_index_map, func_index_map_axes);
      }
    }
    for (const auto &kv : func_index_map_axes) {
      CHECK_EQ(func_index_map.count(kv.first), 0);
      axis_groups.emplace_back(kv.second);
    }
    // debug info
    for (size_t i = 0; i < axis_groups.size(); ++i) {
      std::stringstream info;
      info << "axis_group " << i << ": [";
      for (auto ax : axis_groups[i]) {
        CHECK(axis_stage_info_.count(ax));
        auto ax_stage_id = axis_stage_info_.at(ax).first;
        info << ax << "(" << sch_->stages[ax_stage_id]->op->func_name() << ")"
             << ", ";
      }
      info << "]";
      LOG(DEBUG) << info.str();
    }
  }

  std::unordered_set<StageId> GetStageComputeAt() {
    std::unordered_set<StageId> stage_compute_at;
    for (size_t i = 0; i < sch_->stages.size(); ++i) {
      if (compute_at_pairs_.count(sch_->stages[i]->op)) {
        stage_compute_at.insert(i);
      }
    }
    return stage_compute_at;
  }

  void GetAxisStageInfoGroup() {
    std::vector<std::unordered_set<IterVar>> axis_groups;
    GetAxisGroups(axis_groups);
    std::unordered_set<StageId> stage_compute_at = GetStageComputeAt();
    for (const auto &axis_group : axis_groups) {
      // get StageInfoGroup for axis_group
      int64_t ax_len = -1;
      StageInfoGroup stage_info_group{std::set<StageInfo>(), ELEWISE};
      for (const auto &ax : axis_group) {
        CHECK(axis_stage_info_.count(ax));
        auto stage_info = axis_stage_info_.at(ax);
        auto stage_id = stage_info.first;
        auto axis_type = stage_info.second;
        CHECK_LT(stage_id, sch_->stages.size());
        // skip stage that has been inlined
        if (sch_->stages[stage_id]->attach_type == air::kInline || stage_compute_at.count(stage_id)) {
          continue;
        }

        // length of axis shoule be equal
        auto cur_ax_len = GetAxisLen(ax);
        if (stage_info_group.first.empty()) {
          ax_len = cur_ax_len;
        }
        // -1 indicates that the length of axis is invalid
        if (cur_ax_len == -1) {
          LOG(WARNING) << "length of axis " << ax << "(" << sch_->stages[stage_id]->op->func_name() << ") is invalid";
        }
        if (cur_ax_len != ax_len || cur_ax_len == -1) {
          stage_info_group.first.clear();
          break;
        }

        stage_info_group.first.emplace(stage_info);
        if (axis_type > stage_info_group.second) {
          stage_info_group.second = axis_type;
        }
      }
      // set StageInfoGroup for axis
      if (!stage_info_group.first.empty()) {
        for (const auto &ax : axis_group) {
          axis_stage_info_group_[ax] = stage_info_group;
        }
      }
    }
  }

  void GetSimplifyInfo(const Array<IterVar> &axis, const Array<IterVar> &axis2) {
    Map<Var, Range> simplify_info;
    for (const auto &iter_var : axis) {
      simplify_info.Set(iter_var->var, iter_var->dom);
    }
    for (const auto &iter_var : axis2) {
      simplify_info.Set(iter_var->var, iter_var->dom);
    }
    simplify_info_ = simplify_info;
  }

  std::vector<std::vector<size_t>> InsertAxisOne(const Array<IterVar> &axis,
                                                 const std::vector<std::vector<size_t>> &groups) {
    std::vector<std::vector<size_t>> groups_new;
    std::unordered_set<size_t> fuse_groups_set(groups[0].begin(), groups[0].end());
    groups_new.push_back(std::vector<size_t>());
    for (size_t i = 0; i < axis.size(); ++i) {
      if (fuse_groups_set.count(i) || GetAxisLen(axis[i]) == 1) {
        groups_new[0].push_back(i);
      }
    }
    return groups_new;
  }

  // After operations such as rfactor, the axis, etc. may be regenerated, so save the groups with index
  std::vector<std::vector<size_t>> GetAxisIndexFuseGroups(const Array<IterVar> &axis, bool is_reduce) {
    std::vector<std::vector<size_t>> groups;
    if (!split_config_.empty()) {
      split_index_.clear();
      split_index_ = split_config_;
      groups.push_back(split_index_);
      return groups;
    }

    for (size_t i = 0; i < axis.size(); ++i) {
      auto ax = axis[i];
      auto group_id = groups.size();
      if (!axis_stage_info_group_.count(ax)) {
        continue;
      }
      for (size_t j = 0; j < groups.size(); ++j) {
        auto other_ax = axis[groups[j][0]];
        CHECK(axis_stage_info_group_.count(other_ax));
        if (axis_stage_info_group_.at(ax) == axis_stage_info_group_.at(other_ax)) {
          group_id = j;
          break;
        }
      }
      if (group_id == groups.size()) {
        std::vector<size_t> cur_axis_index_group = {i};
        groups.push_back(cur_axis_index_group);
      } else {
        groups[group_id].push_back(i);
      }
    }

    if (is_reduce && groups.size() == 1 && groups[0].size() < axis.size()) {
      // The axis with length 1 is added to the fuse group
      groups = InsertAxisOne(axis, groups);
    }

    std::vector<std::vector<size_t>> fuse_groups;
    for (const auto &group : groups) {
      if (group.size() > 1) {
        fuse_groups.emplace_back(group);
      }
    }

    if (!fuse_groups.empty()) {
      split_index_ = fuse_groups[0];
    }
    return fuse_groups;
  }

  void GetOpFuseInfo() {
    for (size_t i = 0; i < sch_->stages.size(); ++i) {
      auto stage = sch_->stages[i];
      if (stage->attach_type == air::kInline) {
        continue;
      }
      auto op = stage->op;
      if (auto compute_op = op.as<air::ComputeOpNode>()) {
        std::vector<std::vector<size_t>> reduce_axis_index_fuse_groups;
        if (compute_op->reduce_axis.size() > 1) {
          reduce_axis_index_fuse_groups = GetAxisIndexFuseGroups(compute_op->reduce_axis, true);
          if (!(reduce_axis_index_fuse_groups.size() == 1 &&
                reduce_axis_index_fuse_groups[0].size() == compute_op->reduce_axis.size())) {
            LOG(WARNING) << "The scenes where the reduce_axis cannot be fused into one axis currently not supported."
                         << std::endl;
            op_fuse_info_.clear();
            return;
          }
        }
        std::vector<std::vector<size_t>> axis_index_fuse_groups;
        if (compute_op->axis.size() > 1) {
          axis_index_fuse_groups = GetAxisIndexFuseGroups(compute_op->axis, false);
        }
        if (!axis_index_fuse_groups.empty() || !reduce_axis_index_fuse_groups.empty()) {
          op_fuse_info_[op] = FuseInfo{axis_index_fuse_groups, reduce_axis_index_fuse_groups};
        }
      }
    }
  }
};

class FuseMutator {
 public:
  FuseMutator(const Schedule &sch, const std::unordered_map<Operation, FuseInfo> &op_fuse_info)
      : sch_(sch), op_fuse_info_(op_fuse_info) {}

  void Fuse() {
    if (op_fuse_info_.empty()) {
      return;
    }
    for (auto op : sch_->outputs) {
      TraverseFuse(op);
    }
  }

  void TraverseFuse(const Operation &op) {
    if (!op.defined() || op->IsInstance<PlaceholderOpNode>() || fuse_visited_.count(op)) {
      return;
    }
    RunFuse(op);
    for (auto t : op->InputTensors()) {
      TraverseFuse(t->op);
    }
    fuse_visited_.insert(op);
  }

 private:
  Schedule sch_;
  std::unordered_set<Operation> fuse_visited_;
  std::unordered_map<Operation, FuseInfo> op_fuse_info_;

  void RunFuse(const Operation &op) {
    if (!op_fuse_info_.count(op)) {
      return;
    }
    LOG(INFO) << "Run fuse for op: " << op;
    auto fuse_info = op_fuse_info_.at(op);
    auto tensor = op.output(0);
    auto compute_op = sch_[tensor]->op.as<air::ComputeOpNode>();
    // fuse reduce axis of op
    auto reduce_axis_index_groups = fuse_info.reduce_axis_index_groups;
    CHECK_NOTNULL(compute_op);
    if (!reduce_axis_index_groups.empty()) {
      LOG(INFO) << "Fuse ReduceAxis: " << compute_op->reduce_axis;
      CHECK_EQ(compute_op->reduce_axis.size(), reduce_axis_index_groups[0].size());
      IterVar fused_reduce_axis;
      sch_[tensor].fuse(compute_op->reduce_axis, &fused_reduce_axis);
      // reduce by the fused_reduce_axis
      auto data_rf = sch_.rfactor(tensor, fused_reduce_axis);
      if (data_rf.size() == 1) {
        sch_[data_rf[0]].compute_inline();
      }
    }
    // fuse axis of op
    auto axis_index_groups = fuse_info.axis_index_groups;
    compute_op = sch_[tensor]->op.as<air::ComputeOpNode>();
    CHECK_NOTNULL(compute_op);
    if (!axis_index_groups.empty()) {
      auto axis_fuse_groups = GetAxisGroups(compute_op->axis, axis_index_groups);
      // As the fuse can only fuse iterVar that are consecutive between each other,
      // the axis that needs the fuse must be consecutive.
      // To make the axis are consecutive by reorder.
      auto axis_order = GetAxisOrder(compute_op->axis, axis_fuse_groups);
      sch_[tensor].reorder(axis_order);
      for (const auto &axis_group : axis_fuse_groups) {
        LOG(INFO) << "Fuse Axis: " << axis_group;
        IterVar fused_axis;
        sch_[tensor].fuse(axis_group, &fused_axis);
      }
    }
  }

  std::vector<Array<IterVar>> GetAxisGroups(const Array<IterVar> &axis,
                                            std::vector<std::vector<size_t>> axis_index_groups) {
    std::vector<Array<IterVar>> axis_fuse_groups;
    for (const auto &axis_index_group : axis_index_groups) {
      Array<IterVar> axis_group;
      for (const auto &axis_index : axis_index_group) {
        axis_group.push_back(axis[axis_index]);
      }
      axis_fuse_groups.push_back(axis_group);
    }
    return axis_fuse_groups;
  }

  Array<IterVar> GetAxisOrder(const Array<IterVar> &axis, const std::vector<Array<IterVar>> &axis_fuse_groups) {
    std::unordered_map<IterVar, size_t> axis_map_group_index;
    for (size_t i = 0; i < axis_fuse_groups.size(); ++i) {
      const auto &axis_fuse_group = axis_fuse_groups[i];
      for (const auto &ax : axis_fuse_group) {
        axis_map_group_index[ax] = i;
      }
    }
    std::unordered_set<IterVar> visited;
    Array<IterVar> axis_order;
    for (const auto &ax : axis) {
      if (visited.count(ax)) {
        continue;
      }
      if (axis_map_group_index.count(ax)) {
        auto group_index = axis_map_group_index.at(ax);
        for (const auto &ax_f : axis_fuse_groups[group_index]) {
          if (visited.count(ax_f)) {
            continue;
          }
          axis_order.push_back(ax_f);
          visited.insert(ax_f);
        }
      } else {
        axis_order.push_back(ax);
        visited.insert(ax);
      }
    }
    CHECK_EQ(axis_order.size(), axis.size());
    return axis_order;
  }
};

class ComputeAtProcess {
 public:
  explicit ComputeAtProcess(Schedule &sch) : sch_(sch) {}

  std::unordered_map<Operation, Operation> GetComputeAtPairs() {
    if (!enable_compute_at_) {
      return std::unordered_map<Operation, Operation>();
    }
    GetOutputBroadcastPair();
    compute_at_pairs_ = output_broadcast_pairs_;
    return compute_at_pairs_;
  }

  void ComputeAt(const std::unordered_map<Operation, FuseInfo> &op_fuse_info, const bool &enable_stitch_fusion) {
    if (!enable_compute_at_) {
      return;
    }
    for (auto kv : compute_at_pairs_) {
      auto op1 = kv.first;
      auto op2 = kv.second;
      if (!enable_stitch_fusion && !op_fuse_info.count(op2)) {
        continue;
      }
      auto leaf_iter_vars_size = sch_[op2]->leaf_iter_vars.size();
      auto compute_at_itervar = sch_[op2]->leaf_iter_vars[leaf_iter_vars_size - 1];
      sch_[op1].compute_at(sch_[op2], compute_at_itervar);
      LOG(INFO) << "Run compute_at for op: " << op1 << " and " << op2;
      // For the output, its is_output attribute should be set to false after the compute_at.
      sch_[op1]->is_output = false;
    }
  }

  void SetDisable() { enable_compute_at_ = false; }

 private:
  Schedule sch_;
  bool enable_compute_at_{true};
  std::unordered_map<Operation, Operation> compute_at_pairs_;
  std::unordered_map<Operation, std::unordered_set<Operation>> op_input_ops_;
  std::unordered_map<Operation, Operation> output_broadcast_pairs_;

  void GetOutputBroadcastPair() {
    if (sch_->outputs.size() < 2) {
      return;
    }
    std::unordered_map<Operation, std::vector<Operation>> enable_output_broadcast_pairs;
    std::vector<const ComputeOpNode *> output_compute_ops;
    for (auto op : sch_->outputs) {
      auto compute_op = op.as<ComputeOpNode>();
      if (compute_op->reduce_axis.size() > 0) {
        continue;
      }
      output_compute_ops.push_back(compute_op);
    }
    for (auto compute_op : output_compute_ops) {
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
        CHECK(op_input_ops_.count(enable_broadcast_op));
        if (op_input_ops_[enable_broadcast_op].count(op)) {
          if (output_broadcast_pairs_.count(op)) {
            // otherwise there will be a problem of poly stuck
            LOG(DEBUG) << "As output " << op->func_name() << " needs broadcast to output "
                       << output_broadcast_pairs_.at(op)->func_name() << " and output "
                       << enable_broadcast_op->func_name() << ", this scenario does not run the compute_at."
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
          CHECK(op_input_ops_.count(input_op->op));
          input_ops.insert(op_input_ops_[input_op->op].begin(), op_input_ops_[input_op->op].end());
        } else {
          input_ops.insert(input_op->op);
        }
      }
      op_input_ops_[op] = input_ops;
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
    auto ax_1_extent = GetAxisLen(ax_1);
    auto ax_2_extent = GetAxisLen(ax_2);
    if (ax_1_extent < 0 || ax_2_extent < 0) {
      return false;
    }
    return ax_1_extent == ax_2_extent;
  }
};

void AutoFuse(Schedule sch, const std::string &split_str, std::vector<size_t> &split_index,
              const bool &enable_stitch_fusion) {
  // Notice: The stitch related function is only a temporary solution,
  // and the stitch-related function code will be deleted from this pass in the future.

  std::vector<size_t> split_config;
  if (enable_stitch_fusion) {
    auto split_vec = dmlc::Split(split_str, ' ');
    for (const auto &c : split_vec) {
      char *endptr = nullptr;
      const int radix = 10;
      size_t split = strtol(c.c_str(), &endptr, radix);
      split_config.emplace_back(split);
    }
  }

  auto fuse_check = FuseCheck(sch, split_config);
  auto need_fuse = fuse_check.NeedToFuse();

  // For fuse, if there is a broadcast relationship between outputs,
  // the current solution requires the compute_at for outputs.
  std::unordered_map<Operation, Operation> compute_at_pairs;
  auto compute_at_process = ComputeAtProcess(sch);
  if (need_fuse || enable_stitch_fusion) {
    compute_at_pairs = compute_at_process.GetComputeAtPairs();
  }

  std::unordered_map<Operation, FuseInfo> op_fuse_info;
  if (need_fuse) {
    auto fuse_visit = FuseVisit(sch, compute_at_pairs, split_config, split_index);
    fuse_visit.Run();
    op_fuse_info = fuse_visit.OpFuseInfo();
    FuseMutator(sch, op_fuse_info).Fuse();
  }

  // The FuseMutator may change the axis, so the ComputeAt should run after it
  if (need_fuse || enable_stitch_fusion) {
    compute_at_process.ComputeAt(op_fuse_info, enable_stitch_fusion);
  }
}
}  // namespace schedule
}  // namespace akg
