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
#include "composite/lower_tree/stitch_fusion.h"
#include "composite/utils/util.h"
#include "composite/utils/dump.h"
#include "dmlc/logging.h"
#include <fstream>
#include "common/target_info.h"
#include "common/common_util.h"

namespace akg {
struct WorkspaceInfo {
  Array <Expr> name;
  Array <Expr> offset;
  Array <Expr> type;
  int64_t total_bytes{0};
};

int64_t GetTotalSize(const Array <Expr> &shape) {
  int64_t total_sz = 1;
  for (const auto &s : shape) {
    if (s.as<IntImm>()) {
      total_sz *= s.as<IntImm>()->value;
    } else if (s.as<UIntImm>()) {
      total_sz *= s.as<UIntImm>()->value;
    } else {
      LOG(FATAL) << "shape element should be of type IntImm or UIntImm";
    }
  }
  return total_sz;
}

int64_t GetTotalBytesAligned(const Array <Expr> &shape, const Type &dtype, int align = 1) {
  CHECK(align > 0);
  int64_t total_sz = GetTotalSize(shape);
  total_sz *= dtype.bytes();
  total_sz = (total_sz + align - 1) / align * align;
  return total_sz;
}

class AddCondition : public IRMutator {
 public:
  explicit AddCondition(Expr condition) : condition_(std::move(condition)) {}
  Stmt Run(Stmt &s) {
    s = this->Mutate(s);
    visit_ = false;
    if (last_alloc) {
      return this->Mutate(s);
    } else {
      return IfThenElse::make(condition_, s);
    }
  }

 private:
  Stmt Mutate_(const Allocate *op, const Stmt &s) final {
    if (visit_) {
      last_alloc = op;
    } else {
      if (op == last_alloc) {
        return Allocate::make(op->buffer_var, op->type, op->extents, op->condition,
                              IfThenElse::make(condition_, this->Mutate(op->body)));
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  Expr condition_;
  bool visit_{true};
  const Allocate *last_alloc{nullptr};
};

Stmt EmitUnifyIterVars(Stmt &stmt, std::unordered_map<NodeRef, Expr, air::NodeHash, air::NodeEqual> &itervars) {
  for (auto &kv : itervars) {
    stmt = AttrStmt::make(kv.first, air::ir::attr::thread_extent, kv.second, stmt);
  }
  return stmt;
}

int AvgType(std::vector<StitchOpType> &type_array) {
  if (type_array.empty()) return 0;
  int sum = 0;
  for (auto &i : type_array) {
    sum += static_cast<int>(i);
  }
  return sum / static_cast<int>(type_array.size());
}

IrAttrInfo GetIRAttr(StitchOpType type, BufferStitchAttr &stitch_attr_info, std::vector<StitchOpType> &type_array,
                     std::vector<GridBlockDims> &dim_array, const Map <std::string, NodeRef> &attrs) {
  // note: type_array dose NOT include current ir type.
  IrAttrInfo ir_attr_info;
  ir_attr_info.attrs = attrs;
  // In all stitch cases, grid_dims betweem irs are the same.
  auto grid_dims = dim_array[0].griddim_x * dim_array[0].griddim_y * dim_array[0].griddim_z;
  ir_attr_info.grid_dims = grid_dims;

  switch (type) {
    case StitchOpType::Broadcast:ir_attr_info.dims = dim_array[0];
      break;
    case StitchOpType::All_Reduce:
      // If another reduce exists before all_reduce.
      if (AvgType(type_array) > static_cast<int>(StitchOpType::Broadcast)) {
        ir_attr_info.block_dims = (stitch_attr_info.elemwise_size.as<IntImm>()->value - 1) / grid_dims + 1;
        ir_attr_info.dims.blockdim_x = ir_attr_info.block_dims;
        ir_attr_info.dims.griddim_x = grid_dims;
        if (grid_dims > 1) {
          // all reduce between different blocks should enable atomic add.
          ir_attr_info.attrs.Set("enable_atomic_add", Expr(1));
        }
      } else {
        ir_attr_info.dims = stitch_attr_info.dims;
        ir_attr_info.grid_dims = dim_array[0].griddim_x * dim_array[0].griddim_y * dim_array[0].griddim_z;
        ir_attr_info.block_dims = dim_array[0].blockdim_x * dim_array[0].blockdim_y * dim_array[0].blockdim_z;
      }
      break;
    case StitchOpType::Reduce2D_X:
      ir_attr_info.block_dims = dim_array[0].blockdim_x * dim_array[0].blockdim_y * dim_array[0].blockdim_z;
      ir_attr_info.dims = dim_array[0];
      break;
    case StitchOpType::Elem:
      ir_attr_info.block_dims = (stitch_attr_info.elemwise_size.as<IntImm>()->value - 1) / grid_dims + 1;
      ir_attr_info.dims = dim_array[0];
      break;
    default:auto dims = dim_array.back();
      ir_attr_info.dims = dims;
      ir_attr_info.block_dims = dims.blockdim_x * dims.blockdim_y * dims.blockdim_z;
  }
  if (AvgType(type_array) == static_cast<int>(StitchOpType::Reduce2D_X) &&
      static_cast<int>(type) <= static_cast<int>(StitchOpType::Broadcast)) {
    ir_attr_info.attrs.Set("use_shared_memory", Expr(0));
    ir_attr_info.attrs.Set("use_register_memory", Expr(0));
  }
  ir_attr_info.attrs.Set("enable_vectorization", Expr(0));
  return ir_attr_info;
}

class StitchMutateGPU : public IRMutator {
 public:
  explicit StitchMutateGPU(std::unordered_map<std::string, NodeRef> &stitch_buffer,
                           const std::unordered_map<std::string, NodeRef> &real_outputs,
                           Array <NodeRef> &workspace_args,
                           Map <Tensor, Buffer> &workspace_binds)
      : stitch_buffer_(stitch_buffer),
        real_outputs_(real_outputs),
        workspace_args_(workspace_args),
        workspace_binds_(workspace_binds) {}
  Stmt Run(Stmt &s) {
    StitchBufferScopeAnalyze();
    GetStitchBufferVars();
    s = Mutate(s);
    s = AddStitchBufRealize(s);
    return s;
  }
  void Get_GPU_Info(int total_block, std::unordered_map<std::string, Region> buf_region_map) {
    total_block_ = total_block;
    buf_region_map_ = buf_region_map;
  }

 private:
  void CollectCondition(const AttrStmt *op, const std::string &name) {
    if (IsThreadIdxX(name)) {
      if (is_one(Simplify((op->value - blockdim_x) < 0))) {
        // blockdim_x in elemwise is larger than blockdim_y in reduce.
        add_condition = true;
        condition = thread_idx_x < op->value;
      }
      if (is_one(Simplify((op->value - blockdim_x) > 0))) {
        Expr ele_red_loop_time = (op->value - 1) / blockdim_x + 1;
        if (is_one(Simplify((ele_red_loop_time - blockdim_y) < 0))) {
          add_condition = true;
          condition = thread_idx_y < ele_red_loop_time;
        }
      }
    }
  }

  bool CollectIdx(const std::string &name, const Var &var, const Expr &value) {
    bool tips = false;
    if (IsThreadIdxX(name)) {
      blockdim_x = value;
      thread_idx_x = var;
      tips = true;
    } else if (IsThreadIdxY(name)) {
      blockdim_y = value;
      thread_idx_y = var;
      tips = true;
    } else if (IsThreadIdxZ(name)) {
      blockdim_z = value;
      thread_idx_z = var;
      tips = true;
    } else if (IsBlockIdxX(name)) {
      block_idx_x = var;
      tips = true;
    } else if (IsBlockIdxY(name)) {
      block_idx_y = var;
      tips = true;
    } else if (IsBlockIdxZ(name)) {
      block_idx_z = var;
      tips = true;
    }
    return tips;
  }
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == air::ir::attr::thread_extent) {
      const auto iv = op->node.as<IterVarNode>();
      std::string name = iv->thread_tag;
      if (idx_names_.count(name)) {
        CollectCondition(op, name);
        return this->Mutate(op->body);
      }
      bool tips = CollectIdx(name, iv->var, op->value);
      if (tips) {
        idx_names_.insert(name);
        itervars_[op->node] = op->value;
      }
      return this->Mutate(op->body);
    }
    rm_attr_ = false;
    auto rm_realize_ = this->Mutate(op->body);
    if (rm_attr_) {
      rm_attr_ = false;
      return this->Mutate(op->body);
    }
    return IRMutator::Mutate_(op, s);
  }
  Stmt Mutate_(const Realize *op, const Stmt &s) final {
    auto ret = IRMutator::Mutate_(op, s);  // mutate for getting stitch_buffer_vars_
    if (op->func.defined() && stitch_buffer_vars_.count(op->func->func_name())) {
      // erase the old Realize and corresponding AttrStmt
      // and then a new Realize op will be recreated with same name and new lifetime in AddStitchBufRealize() finally.
      auto ret = this->Mutate(op->body);
      rm_attr_ = true;
      return ret;
    }
    return ret;
  }
  Expr Mutate_(const Variable *op, const Expr &e) final {
    auto name = op->name_hint;
    // substitute idx
    if (IsBlockIdxX(name)) {
      return rm_block_ ? Expr(0) : block_idx_x;
    }
    if (IsBlockIdxY(name)) {
      return rm_block_ ? Expr(0) : block_idx_y;
    }
    if (IsBlockIdxZ(name)) {
      return rm_block_ ? Expr(0) : block_idx_z;
    }
    if (IsThreadIdxX(name)) {
      return thread_idx_x;
    }
    if (IsThreadIdxY(name)) {
      return thread_idx_y;
    }
    if (IsThreadIdxZ(name)) {
      return thread_idx_z;
    }
    return IRMutator::Mutate_(op, e);
  }
  Stmt AddStitchBufRealize(Stmt &s) {
    auto stmt = s;
    for (auto &kv : stitch_buffer_vars_) {
      if (kv.second.second != SHARED) {
        continue;
      }
      auto t = kv.second.first;
      air::Region bounds;
      if (buf_region_map_.count(kv.first)) {
        bounds = buf_region_map_[kv.first];  // use old shared memory region without transforming.
      } else {
        size_t i = 0;
        int align = 1;
        for (auto j : t->shape) {
          if (i > 0 && !Equal(j, 1)) {
            j = air::ir::Simplify(floordiv(j + align - 1, align) * align);
          }
          ++i;
          bounds.push_back(air::Range::make_by_min_extent(Expr(0), j));
        }
      }
      stmt = air::ir::Realize::make(t->op, t->value_index, t->dtype, bounds, air::const_true(1), stmt);
      stmt = AttrStmt::make(t->op, air::ir::attr::realize_scope, Expr(SHARED), stmt);

      if (workspace_.total_bytes > 0) {
        Map <std::string, NodeRef> workspace;
        workspace.Set("name", workspace_.name);
        workspace.Set("offset", workspace_.offset);
        workspace.Set("type", workspace_.type);
        workspace.Set("total_bytes", IntImm::make(Int(64), workspace_.total_bytes));
        stmt = AttrStmt::make(workspace, "workspace", Expr(1), stmt);
      }
    }
    return stmt;
  }

  std::string GetStitchBufferScope(const NodeRef &stitch_buffer) {
    std::string scope(SHARED);
    if (scoped_stitch_buffer_.find(stitch_buffer) != scoped_stitch_buffer_.end()) {
      scope = scoped_stitch_buffer_[stitch_buffer];
    }
    return scope;
  }

  Expr Mutate_(const Call *op, const Expr &e) final {
    if (op->func.defined() && IsStitchBuffer(op->func->func_name())) {
      auto name = op->func->func_name();
      auto stitch_buffer = GetStitchBuffer(name);
      std::string scope = GetStitchBufferScope(stitch_buffer);
      auto stitch_name = stitch_buffer.as<BufferNode>()->name + "_" + scope;
      CHECK(stitch_buffer_vars_.count(stitch_name));
      auto stitch_var = stitch_buffer_vars_[stitch_name].first;
      rm_block_ = true;
      auto new_args = IRMutator::Mutate_(op, e).as<Call>()->args;
      auto ret = Call::make(op->type, stitch_var->op->name, new_args, op->call_type, stitch_var->op,
                            op->value_index);
      rm_block_ = false;
      return ret;
    }
    if (op->func.defined() && stitch_buffer_vars_.count(op->func->func_name())) {
      // Mutate the right-hand side of Provide Node, unify the call node func_name. we will unify the func var in lower
      // pass.
      auto name = op->func->func_name();
      auto stitch_var = stitch_buffer_vars_[name].first;
      rm_block_ = true;
      auto new_args = IRMutator::Mutate_(op, e).as<Call>()->args;
      auto ret = Call::make(op->type, stitch_var->op->name, new_args, op->call_type, stitch_var->op,
                            op->value_index);
      rm_block_ = false;
      return ret;
    }
    return IRMutator::Mutate_(op, e);
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    auto new_args = IRMutator::Mutate_(op, s).as<Provide>()->args;
    auto stmt = Provide::make(op->func, op->value_index, this->Mutate(op->value), new_args);
    auto name = op->func->func_name();
    if (IsStitchBuffer(name) && !IsOutput(name)) {
      auto stitch_buffer = GetStitchBuffer(name);
      auto scope = GetStitchBufferScope(stitch_buffer);
      if (scope == GLOBAL) return stmt;
      auto stitch_name = stitch_buffer.as<BufferNode>()->name + "_" + scope;
      CHECK(stitch_buffer_vars_.count(stitch_name));
      auto stitch_var = stitch_buffer_vars_[stitch_name].first;
      rm_block_ = true;
      new_args = IRMutator::Mutate_(op, s).as<Provide>()->args;
      stmt = Provide::make(stitch_var->op, op->value_index, this->Mutate(op->value), new_args);
      rm_block_ = false;
    }
    if (IsAllocatedShared(name)) {
      auto stitch_var = stitch_buffer_vars_[name].first;
      stmt = Provide::make(stitch_var->op, op->value_index, this->Mutate(op->value), new_args);
    }
    return stmt;
  }

  Tensor NewStitchTensor(const std::string &name, const Array <Expr> &shape, const Type &dtype,
                         const std::string &scope) {
    auto tensor = placeholder(shape, dtype, name);
    stitch_buffer_vars_[name] = std::make_pair(tensor, scope);
    // CUDA workspace is developing now.
    return tensor;
  }

  NodeRef GetStitchBuffer(const std::string &name) {
    CHECK(IsStitchBuffer(name));
    if (stitch_buffer_.count(name)) return stitch_buffer_[name];
    for (auto &kv : stitch_buffer_) {
      if (kv.second.as<BufferNode>()->name == name) {
        return kv.second;
      }
    }
    return {};
  }

  bool IsStitchBuffer(const std::string &name) {
    if (stitch_buffer_.count(name)) return true;
    for (auto &kv : stitch_buffer_) {
      if (kv.second.as<BufferNode>()->name == name) {
        return true;
      }
    }
    return false;
  }

  bool IsAllocatedShared(const std::string &name) {
    if (stitch_buffer_vars_.count(name)) return true;
    return false;
  }

  bool IsOutput(const std::string &name) {
    for (auto &kv : real_outputs_) {
      if (name == kv.second.as<BufferNode>()->name) {
        return true;
      }
    }
    return false;
  }
  void StitchBufferScopeAnalyze() {
    static int64_t MAX_MEMORY = 49152;
    for (const auto &it : stitch_buffer_) {
      auto buffer = it.second.as<air::BufferNode>();
      CHECK(buffer);
      auto total_bytes = GetTotalBytesAligned(buffer->shape, buffer->dtype) / total_block_;
      if (total_bytes <= MAX_MEMORY) {
        scoped_stitch_buffer_[it.second] = SHARED;
      } else {
        scoped_stitch_buffer_[it.second] = GLOBAL;
      }
    }
  }
  void GetStitchBufferVars() {
    for (auto &kv : stitch_buffer_) {
      std::string stitch_name = kv.second.as<BufferNode>()->name + "_" + SHARED;
      auto stitch_buffer = kv.second;
      auto shape = stitch_buffer.as<BufferNode>()
          ->shape;  // Note that this shape may not be correct, we will fix this in lower pass.
      auto scope = GetStitchBufferScope(stitch_buffer);
      auto stitch_var = NewStitchTensor(stitch_name, shape, stitch_buffer.as<BufferNode>()->dtype, scope);
    }
  }

 public:
  std::unordered_map<NodeRef, Expr, air::NodeHash, air::NodeEqual> itervars_;
  Expr condition;
  bool add_condition{false};

 private:
  int total_block_{1};
  std::unordered_map<std::string, Region> buf_region_map_;
  std::unordered_map<std::string, NodeRef> stitch_buffer_;
  std::unordered_map<std::string, NodeRef> real_outputs_;
  std::unordered_map<std::string, std::pair<Tensor, std::string>> stitch_buffer_vars_;
  std::unordered_map<NodeRef, std::string, air::NodeHash, air::NodeEqual> scoped_stitch_buffer_;
  Array <NodeRef> &workspace_args_;
  Map <Tensor, Buffer> &workspace_binds_;
  WorkspaceInfo workspace_;

  bool rm_attr_{false};
  bool rm_block_{false};

  std::unordered_map<std::string, Var> vars_;
  Expr blockdim_x{0};
  Expr blockdim_y{1};
  Expr blockdim_z{1};
  Var thread_idx_x;
  Var thread_idx_y;
  Var thread_idx_z;
  Var block_idx_x;
  Var block_idx_y;
  Var block_idx_z;
  std::unordered_set<std::string> idx_names_;
};
Stmt StitchFusionGPU(std::vector<Stmt> &stitch_irs, const std::string &kernel_name,
                     std::unordered_map<std::string, NodeRef> &stitch_buffer,
                     const std::unordered_map<std::string, NodeRef> &real_outputs, Array <NodeRef> &workspace_args,
                     Map <Tensor, Buffer> &workspace_binds) {
  CHECK(stitch_irs.size() > 1);
  for (const auto &kv : stitch_buffer) {
    LOG(INFO) << kv.first << "->" << kv.second.as<air::BufferNode>()->name;
  }
  DumpStmt2File("stitch_info/" + kernel_name + "_before_stitch.cc", Block::make(stitch_irs));
  auto func = StitchMutateGPU(stitch_buffer, real_outputs, workspace_args, workspace_binds);
  auto get_info_func = GetGpuMutateInfo(stitch_irs);
  func.Get_GPU_Info(get_info_func.get_total_block(), get_info_func.get_buffer_region_map());
  auto stmt = Block::make(stitch_irs);
  stmt = func.Run(stmt);
  stmt = EmitUnifyIterVars(stmt, func.itervars_);
  DumpStmt2File("stitch_info/" + kernel_name + "_after_stitch.cc", stmt);
  return stmt;
}
class StitchMutateAscend : public IRMutator {
 public:
  explicit StitchMutateAscend(std::unordered_map<std::string, NodeRef> &stitch_buffer,
                              const std::unordered_map<std::string, NodeRef> &real_outputs,
                              Array <NodeRef> &workspace_args, Map <Tensor, Buffer> &workspace_binds)
      : stitch_buffer_(stitch_buffer),
        real_outputs_(real_outputs),
        workspace_args_(workspace_args),
        workspace_binds_(workspace_binds) {}

  Stmt Run(Stmt &s) {
    StitchBufferScopeAnalyze();
    s = Mutate(s);
    s = AddStitchBufRealize(s);
    return s;
  }

 private:
  Stmt AddStitchBufRealize(Stmt &s) {
    auto stmt = s;
    for (auto &kv : stitch_buffer_vars_) {
      if (kv.second.second != LOCAL_L1) {
        continue;
      }
      auto t = kv.second.first;
      Region bounds;
      size_t i = 0;
      auto align = 32 / t->dtype.bytes();
      for (auto j : t->shape) {
        if (i > 0 && !Equal(j, 1)) {
          j = Simplify(floordiv(j + align - 1, align) * align);
        }
        ++i;
        bounds.push_back(Range::make_by_min_extent(Expr(0), j));
      }
      stmt = Realize::make(t->op, t->value_index, t->dtype, bounds, const_true(1), stmt);
      stmt = AttrStmt::make(t->op, air::ir::attr::realize_scope, Expr("local.L1"), stmt);
    }
    if (workspace_.total_bytes > 0) {
      Map <std::string, NodeRef> workspace;
      workspace.Set("name", workspace_.name);
      workspace.Set("offset", workspace_.offset);
      workspace.Set("type", workspace_.type);
      workspace.Set("total_bytes", IntImm::make(Int(64), workspace_.total_bytes));
      stmt = AttrStmt::make(workspace, "workspace", Expr(1), stmt);
    }
    return stmt;
  }

  std::string GetStitchBufferScope(const NodeRef &stitch_buffer) {
    std::string scope(LOCAL_L1);
    if (scoped_stitch_buffer_.find(stitch_buffer) != scoped_stitch_buffer_.end()) {
      scope = scoped_stitch_buffer_[stitch_buffer];
    }
    return scope;
  }

  Expr Mutate_(const Call *op, const Expr &e) final {
    if (op->func.defined() && IsStitchBuffer(op->func->func_name())) {
      auto name = op->func->func_name();
      auto stitch_buffer = GetStitchBuffer(name);
      auto stitch_name = stitch_buffer.as<BufferNode>()->name + "_stitch_" + GetStitchBufferScope(stitch_buffer);
      CHECK(stitch_buffer_vars_.count(stitch_name));
      auto stitch_var = stitch_buffer_vars_[stitch_name].first;
      return Call::make(op->type, stitch_var->op->name, op->args, op->call_type, stitch_var->op, op->value_index);
    }
    return IRMutator::Mutate_(op, e);
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    auto f = op;
    std::vector<const For *> fors;
    fors.emplace_back(op);
    while (f->body.as<For>()) {
      f = f->body.as<For>();
      fors.emplace_back(f);
    }
    if (f->body.as<Provide>()) {
      ub_to_gm_ = {};
      auto first = this->Mutate(f->body);
      if (ub_to_gm_.defined()) {
        Stmt rest = ub_to_gm_;
        size_t i = fors.size();
        std::unordered_map<const Variable *, Expr> value_map;
        for (auto it = fors.rbegin(); it < fors.rend(); ++it) {
          --i;
          const For *fs = *it;
          auto loopvar = Var("cc" + std::to_string(i), fs->loop_var->type);
          value_map[fs->loop_var.get()] = loopvar;
          first = For::make(fs->loop_var, fs->min, fs->extent, fs->for_type, fs->device_api, first);
          rest = For::make(loopvar, fs->min, fs->extent, fs->for_type, fs->device_api, rest);
        }
        rest = air::ir::Substitute(rest, value_map);
        ub_to_gm_ = {};
        return Block::make(first, rest);
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    auto stmt = Provide::make(op->func, op->value_index, this->Mutate(op->value), op->args);
    auto ub_to_gm = stmt;
    auto name = op->func->func_name();
    if (IsStitchBuffer(name)) {
      auto stitch_buffer = GetStitchBuffer(name);
      auto scope = GetStitchBufferScope(stitch_buffer);
      auto stitch_name = stitch_buffer.as<BufferNode>()->name + "_stitch_" + scope;
      bool new_buffer = !stitch_buffer_vars_.count(stitch_name);
      auto stitch_var = new_buffer ? NewStitchTensor(stitch_name, stitch_buffer.as<BufferNode>()->shape,
                                                     stitch_buffer.as<BufferNode>()->dtype, scope)
                                   : stitch_buffer_vars_[stitch_name].first;
      stmt = Provide::make(stitch_var->op, op->value_index, this->Mutate(op->value), op->args);
      if (IsOutput(name)) {
        ub_to_gm_ = ub_to_gm;
      }
    }
    return stmt;
  }

  Tensor NewStitchTensor(const std::string &name, const Array <Expr> &shape, const Type &dtype,
                         const std::string &scope) {
    auto tensor = placeholder(shape, dtype, name);
    stitch_buffer_vars_[name] = std::make_pair(tensor, scope);
    if (scope == GLOBAL) {
      // Add stitch buffer to args and binds if it uses global memory(workspace)
      auto buf = DeclBuffer(tensor, -1, 0);
      workspace_args_.push_back(buf);
      workspace_binds_.Set(tensor, buf);

      auto total_bytes = GetTotalBytesAligned(shape, dtype, 32);
      NodePtr <ExprNode> type_node = make_node<ExprNode>();
      type_node->type = dtype;
      workspace_.name.push_back(StringImm::make(name));
      workspace_.offset.push_back(IntImm::make(Int(64), workspace_.total_bytes));
      workspace_.type.push_back(Expr(type_node));
      workspace_.total_bytes += total_bytes;
    }
    return tensor;
  }

  NodeRef GetStitchBuffer(const std::string &name) {
    CHECK(IsStitchBuffer(name));
    if (stitch_buffer_.count(name)) return stitch_buffer_[name];
    for (auto &kv : stitch_buffer_) {
      if (kv.second.as<BufferNode>()->name == name) {
        return kv.second;
      }
    }
    return {};
  }

  bool IsStitchBuffer(const std::string &name) {
    if (stitch_buffer_.count(name)) return true;
    for (auto &kv : stitch_buffer_) {
      if (kv.second.as<BufferNode>()->name == name) {
        return true;
      }
    }
    return false;
  }

  bool IsOutput(const std::string &name) {
    for (auto &kv : real_outputs_) {
      if (name == kv.second.as<BufferNode>()->name) {
        return true;
      }
    }
    return false;
  }

  void StitchBufferScopeAnalyze() {
    static int64_t MAX_MEMORY = 1048576;
    for (const auto &it : stitch_buffer_) {
      auto buffer = it.second.as<BufferNode>();
      CHECK(buffer);
      auto total_bytes = GetTotalBytesAligned(buffer->shape, buffer->dtype, 32);
      if (total_bytes <= MAX_MEMORY) {
        scoped_stitch_buffer_[it.second] = LOCAL_L1;
      } else {
        scoped_stitch_buffer_[it.second] = GLOBAL;
      }
    }
  }

  Stmt ub_to_gm_;
  std::unordered_map<std::string, NodeRef> stitch_buffer_;
  std::unordered_map<std::string, NodeRef> real_outputs_;
  std::unordered_map<std::string, std::pair<Tensor, std::string>> stitch_buffer_vars_;
  std::unordered_map<NodeRef, std::string, air::NodeHash, air::NodeEqual> scoped_stitch_buffer_;
  WorkspaceInfo workspace_;
  Array <NodeRef> &workspace_args_;
  Map <Tensor, Buffer> &workspace_binds_;
};
Stmt StitchFusionAscend(std::vector<Stmt> &stitch_irs, const std::string &kernel_name,
                        std::unordered_map<std::string, NodeRef> &stitch_buffer,
                        const std::unordered_map<std::string, NodeRef> &real_outputs, Array <NodeRef> &workspace_args,
                        Map <Tensor, Buffer> &workspace_binds) {
  CHECK(stitch_irs.size() > 1);
  for (const auto &kv : stitch_buffer) {
    LOG(INFO) << kv.first << " -> " << kv.second.as<BufferNode>()->name;
  }
  DumpStmt2File("stitch_info/" + kernel_name + "_before_stitch.cc", Block::make(stitch_irs));
  auto stmt = Block::make(stitch_irs);
  stmt = StitchMutateAscend(stitch_buffer, real_outputs, workspace_args, workspace_binds).Run(stmt);
  DumpStmt2File("stitch_info/" + kernel_name + "_after_stitch.cc", stmt);
  return stmt;
}
}  // namespace akg
