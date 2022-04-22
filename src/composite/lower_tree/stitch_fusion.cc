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
#include <ir_pass.h>
#include <fstream>

namespace akg {
struct WorkspaceInfo {
  Array<Expr> name;
  Array<Expr> offset;
  Array<Expr> type;
  int64_t total_bytes{0};
};

int64_t GetTotalSize(const Array<Expr> &shape) {
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

int64_t GetTotalBytesAligned(const Array<Expr> &shape, const Type &dtype, int align = 1) {
  CHECK(align > 0);
  int64_t total_sz = GetTotalSize(shape);
  total_sz *= dtype.bytes();
  total_sz = (total_sz + align - 1) / align * align;
  return total_sz;
}

Var GetReplaceVar(const Var &var, std::unordered_map<std::string, Var> &vars, const std::string &name,
                  const StitchBufferInfo &info) {
  Var replace;
  if (info.type == StorageType::Shared) {
    CHECK(vars.count(info.buf_name));
    replace = vars[info.buf_name];
  } else if (info.type == StorageType::Global) {
    if (!vars.count(name)) return var;
    replace = vars[name];
  }
  return replace;
}

class StitchMutateGPU : public IRMutator {
 public:
  explicit StitchMutateGPU(std::unordered_map<std::string, StitchBufferInfo> &stitch_buffer_map,
                           std::unordered_map<std::string, StitchBufferInfo> &buf_within_op_map,
                           std::vector<std::string> &allocate_revoke, StitchAttrInfo &store_attr,
                           const std::unordered_map<std::string, NodeRef> &real_outputs)
      : stitch_buffer_map_(stitch_buffer_map),
        buf_within_op_map_(buf_within_op_map),
        allocate_revoke_(allocate_revoke),
        store_attr_(store_attr),
        real_outputs_(real_outputs) {}

  Stmt Run(Stmt &s) {
    stitch_type_ = store_attr_.type_array[phase];
    // save allocated shared buffer into vars_
    for (const auto &it : buf_within_op_map_) {
      if (!vars_.count(it.first)) {
        vars_[it.first] = Var();
      }
    }
    return Mutate(s);
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

  void CollectIdx(const std::string &name, const Var &var, const Expr &value) {
    if (IsThreadIdxX(name)) {
      blockdim_x = value;
      thread_idx_x = var;
    } else if (IsThreadIdxY(name)) {
      blockdim_y = value;
      thread_idx_y = var;
    } else if (IsThreadIdxZ(name)) {
      blockdim_z = value;
      thread_idx_z = var;
    } else if (IsBlockIdxX(name)) {
      block_idx_x = var;
    } else if (IsBlockIdxY(name)) {
      block_idx_y = var;
    } else if (IsBlockIdxZ(name)) {
      block_idx_z = var;
    }
  }
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == air::ir::attr::thread_extent) {
      const auto iv = op->node.as<IterVarNode>();
      std::string name = iv->thread_tag;
      if (idx_names_.count(name)) {
        CollectCondition(op, name);
        return this->Mutate(op->body);
      }
      if (phase == 0) {
        CollectIdx(name, iv->var, op->value);
        idx_names_.insert(name);
        itervars_[op->node] = op->value;
      }
      return this->Mutate(op->body);
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Allocate *op, const Stmt &s) final {
    const Variable *buf = op->buffer_var.get();
    std::string name = buf->name_hint;
    for (auto &var : vars_) {
      if (var.first == name && var.second.get()->name_hint != name) {
        vars_[name] = op->buffer_var;
      }
    }
    for (auto &var_revoke : allocate_revoke_) {
      if (name == var_revoke) {
        return this->Mutate(op->body);
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  bool IsOutput(const std::string &name) {
    for (auto &kv : real_outputs_) {
      if (name == kv.second.as<BufferNode>()->name) {
        return true;
      }
    }
    return false;
  }
  Stmt Mutate_(const Store *op, const Stmt &s) final {
    Var var = op->buffer_var;
    auto name = var->name_hint;
    if (stitch_buffer_map_.count(name) && !IsOutput(name)) {
      auto info = stitch_buffer_map_[name];
      if (info.type == StorageType::Shared) {
        auto shared_name = info.buf_name;
        bool new_buffer = !vars_.count(shared_name);
        Var shared = new_buffer ? Var(shared_name) : vars_[shared_name];
        vars_[shared_name] = shared;
        stitch_buffer_map_[shared_name] = info;
        rm_block_ = true;
        auto index = this->Mutate(op->index);
        rm_block_ = false;
        auto stmt = Store::make(shared, this->Mutate(op->value), index, op->predicate);
        if (new_buffer) new_allocate_.insert(stmt.as<Store>());
        return stmt;
      } else {
        vars_[name] = var;
      }
    }
    if (stitch_type_ == StitchOpType::Broadcast)
      return Store::make(op->buffer_var, this->Mutate(op->value), this->Mutate(op->index), op->predicate);
    return IRMutator::Mutate_(op, s);
  }

  Buffer GetReplaceBuffer(const Var &var, const StitchBufferInfo &info) {
    CHECK(var.defined());
    Buffer buf;
    if (var_buf_.find(var) == var_buf_.end()) {
      int64_t sh = info.alloc_size / info.dtype.bytes();
      Array<Expr> shape;
      shape.push_back(IntImm::make(Int(64), sh));
      buf = BufferNode::make(var, info.dtype, shape, {}, 0, var->name_hint, "", -1, 0, BufferType::kDefault);
      var_buf_[var] = buf;
    } else {
      buf = var_buf_[var];
    }
    return buf;
  }

  Expr Mutate_(const Load *op, const Expr &e) final {
    Var var = op->buffer_var;
    auto name = var->name_hint;
    auto index = op->index;
    for (auto &kv : stitch_buffer_map_) {
      if (kv.second.name == name || kv.first == name) {
        auto info = kv.second;
        Var replace = GetReplaceVar(var, vars_, kv.first, info);
        if (info.type == StorageType::Shared) {
          rm_block_ = true;
          index = this->Mutate(index);
          rm_block_ = false;
          return Load::make(op->type, replace, index, op->predicate);
        } else {  // use workspace
          auto buf = GetReplaceBuffer(replace, info);
          if (workspace_.find(buf) == workspace_.end()) {
            workspace_.insert(buf);
          }
          index = this->Mutate(index);
          return Load::make(op->type, replace, index, op->predicate);
        }
      }
    }
    if (stitch_type_ == StitchOpType::Broadcast)
      return Load::make(op->type, op->buffer_var, this->Mutate(index), op->predicate);
    return IRMutator::Mutate_(op, e);
  }

  Expr Mutate_(const Variable *op, const Expr &e) final {
    auto name = op->name_hint;
    // substitute idx
    if (IsBlockIdxX(name)) return rm_block_ ? Expr(0) : block_idx_x;
    if (IsBlockIdxY(name)) return rm_block_ ? Expr(0) : block_idx_y;
    if (IsBlockIdxZ(name)) return rm_block_ ? Expr(0) : block_idx_z;
    if (IsThreadIdxX(name)) return thread_idx_x;
    if (IsThreadIdxY(name)) return thread_idx_y;
    if (IsThreadIdxZ(name)) return thread_idx_z;
    return IRMutator::Mutate_(op, e);
  }

 public:
  std::unordered_set<NodeRef, air::ExprHash, air::NodeEqual> workspace_;
  std::unordered_set<const Store *> new_allocate_;
  std::unordered_map<NodeRef, Expr, air::NodeHash, air::NodeEqual> itervars_;
  Expr condition;
  bool add_condition{false};
  size_t phase{0};

 private:
  bool rm_block_{false};
  std::unordered_map<std::string, StitchBufferInfo> &stitch_buffer_map_;
  std::unordered_map<std::string, StitchBufferInfo> &buf_within_op_map_;
  std::unordered_map<Var, Buffer, air::NodeHash, air::NodeEqual> var_buf_;
  std::vector<std::string> &allocate_revoke_;
  StitchAttrInfo &store_attr_;
  std::unordered_map<std::string, Var> vars_;
  std::unordered_map<std::string, NodeRef> real_outputs_;
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
  StitchOpType stitch_type_{StitchOpType::Unknown};
};

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

Stmt EmitWorkspace(Stmt &stmt, const std::unordered_set<NodeRef, air::NodeHash, air::NodeEqual> &workspace,
                   Array<NodeRef> &workspace_args) {
  if (workspace.empty()) {
    return stmt;
  }
  WorkspaceInfo info{};
  for (const auto &it : workspace) {
    // Add workspace to args
    workspace_args.push_back(it);
    auto buf = it.as<BufferNode>();
    CHECK(buf);
    NodePtr<ExprNode> type_node = make_node<ExprNode>();
    type_node->type = buf->dtype;
    info.name.push_back(buf->name);
    info.offset.push_back(IntImm::make(Int(64), info.total_bytes));
    info.type.push_back(Expr(type_node));
    info.total_bytes += GetTotalBytesAligned(buf->shape, buf->dtype);
  }
  // Add workspace information to stmt, which will be used later when generate the cuda kernel function.
  Map<std::string, NodeRef> attr;
  attr.Set("name", info.name);
  attr.Set("offset", info.offset);
  attr.Set("type", info.type);
  attr.Set("total_bytes", IntImm::make(Int(64), info.total_bytes));
  stmt = AttrStmt::make(attr, "workspace", Expr(1), stmt);
  return stmt;
}

Stmt EmitNewAllocate(Stmt &stmt, const std::unordered_set<const Store *> &new_allocate,
                     std::unordered_map<std::string, StitchBufferInfo> &stitch_buffer_map) {
  for (auto &store : new_allocate) {
    CHECK(stitch_buffer_map.count(store->buffer_var->name_hint));
    auto info = stitch_buffer_map[store->buffer_var->name_hint];
    stmt = Allocate::make(store->buffer_var, info.dtype, {static_cast<int>(info.alloc_size) / info.dtype.bytes()},
                          const_true(), stmt);
    stmt = AttrStmt::make(store->buffer_var, air::ir::attr::storage_scope, StringImm::make("shared"), stmt);
  }
  return stmt;
}

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
                     std::vector<GridBlockDims> &dim_array, const Map<std::string, NodeRef> &attrs) {
  // note: type_array dose NOT include current ir type.
  IrAttrInfo ir_attr_info;
  ir_attr_info.attrs = attrs;
  // In all stitch cases, grid_dims betweem irs are the same.
  auto grid_dims = dim_array[0].griddim_x * dim_array[0].griddim_y * dim_array[0].griddim_z;
  ir_attr_info.grid_dims = grid_dims;

  switch (type) {
    case StitchOpType::Broadcast:
      ir_attr_info.dims = dim_array[0];
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
    default:
      auto dims = dim_array.back();
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

Stmt StitchFusionGpu(std::vector<Stmt> &stitch_irs, const std::string &kernel_name, StitchAttrInfo &store_attr,
                     std::unordered_map<std::string, StitchBufferInfo> &stitch_buffer_map,
                     std::unordered_map<std::string, StitchBufferInfo> &buf_within_op_map,
                     std::vector<std::string> &allocate_revoke,
                     const std::unordered_map<std::string, NodeRef> &real_outputs, Array<NodeRef> &workspace_args) {
  DumpStitchInfo(kernel_name, store_attr, stitch_buffer_map, buf_within_op_map, allocate_revoke);
  DumpStmt2File("stitch_info/" + kernel_name + "_before_stitch.cc", Block::make(stitch_irs));
  auto func = StitchMutateGPU(stitch_buffer_map, buf_within_op_map, allocate_revoke, store_attr, real_outputs);
  CHECK(stitch_irs.size() > 1);
  size_t i = 0;
  for (auto &ir : stitch_irs) {
    func.phase = i;
    ir = func.Run(ir);
    if (func.add_condition) {
      ir = AddCondition(func.condition).Run(ir);
      func.add_condition = false;
    }
    ++i;
    ir = Block::make(ir, Evaluate::make(Expr("=============split===============")));
  }
  auto stmt = Block::make(stitch_irs);
  stmt = EmitWorkspace(stmt, func.workspace_, workspace_args);
  stmt = EmitNewAllocate(stmt, func.new_allocate_, stitch_buffer_map);
  stmt = EmitUnifyIterVars(stmt, func.itervars_);
  DumpStmt2File("stitch_info/" + kernel_name + "_after_stitch.cc", stmt);
  stmt = Simplify(stmt);
  stmt = RemoveNoOp(stmt);
  DumpStmt2File("stitch_info/" + kernel_name + "_after_stitch_simplify.cc", stmt);
  return stmt;
}

class StitchMutateAscend : public IRMutator {
 public:
  explicit StitchMutateAscend(std::unordered_map<std::string, NodeRef> &stitch_buffer,
                              const std::unordered_map<std::string, NodeRef> &real_outputs,
                              Array<NodeRef> &workspace_args, Map<Tensor, Buffer> &workspace_binds)
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
      if (kv.second.second != "local_L1") {
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
      Map<std::string, NodeRef> workspace;
      workspace.Set("name", workspace_.name);
      workspace.Set("offset", workspace_.offset);
      workspace.Set("type", workspace_.type);
      workspace.Set("total_bytes", IntImm::make(Int(64), workspace_.total_bytes));
      stmt = AttrStmt::make(workspace, "workspace", Expr(1), stmt);
    }
    return stmt;
  }

  std::string GetStitchBufferScope(const NodeRef &stitch_buffer) {
    std::string scope("local_L1");
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

  Tensor NewStitchTensor(const std::string &name, const Array<Expr> &shape, const Type &dtype,
                         const std::string &scope) {
    auto tensor = placeholder(shape, dtype, name);
    stitch_buffer_vars_[name] = std::make_pair(tensor, scope);
    if (scope == "global") {
      // Add stitch buffer to args and binds if it uses global memory(workspace)
      auto buf = DeclBuffer(tensor, -1, 0);
      workspace_args_.push_back(buf);
      workspace_binds_.Set(tensor, buf);

      auto total_bytes = GetTotalBytesAligned(shape, dtype, 32);
      NodePtr<ExprNode> type_node = make_node<ExprNode>();
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
        scoped_stitch_buffer_[it.second] = "local_L1";
      } else {
        scoped_stitch_buffer_[it.second] = "global";
      }
    }
  }

  Stmt ub_to_gm_;
  std::unordered_map<std::string, NodeRef> stitch_buffer_;
  std::unordered_map<std::string, NodeRef> real_outputs_;
  std::unordered_map<std::string, std::pair<Tensor, std::string>> stitch_buffer_vars_;
  std::unordered_map<NodeRef, std::string, air::NodeHash, air::NodeEqual> scoped_stitch_buffer_;
  WorkspaceInfo workspace_;
  Array<NodeRef> &workspace_args_;
  Map<Tensor, Buffer> &workspace_binds_;
};
Stmt StitchFusionAscend(std::vector<Stmt> &stitch_irs, const std::string &kernel_name,
                        std::unordered_map<std::string, NodeRef> &stitch_buffer,
                        const std::unordered_map<std::string, NodeRef> &real_outputs, Array<NodeRef> &workspace_args,
                        Map<Tensor, Buffer> &workspace_binds) {
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
