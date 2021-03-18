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
#include "composite/stitch_fusion.h"
#include "composite/util.h"
#include "composite/dump.h"
#include <fstream>

namespace akg {
class GetLoopLen : public IRVisitor {
 public:
  explicit GetLoopLen(const For *f) : f_(f) {}
  void Visit_(const Mul *op) override {
    CHECK(f_);
    if (Equal(op->a, f_->loop_var)) {
      len_ = op->b * f_->extent;
    }
  }
  const For *f_{nullptr};
  Expr len_;
};

class GetBlockExpr : public IRVisitor {
 public:
  GetBlockExpr() = default;
  void Visit_(const Variable *op) override {
    if (op->name_hint.find(BLOCKIDX) != std::string::npos) {
      blockexpr_ = Var(op->name_hint);
    }
  }
  Expr blockexpr_{0};
};

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

struct StoreWithLoopVar {
  Expr old_index;
  std::vector<Var> loopvars;
};

class BroadcastSubstitute : public IRVisitor {
 public:
  BroadcastSubstitute(
    const std::vector<const For *> &loops, const std::unordered_map<std::string, Var> &vars,
    const std::unordered_map<std::string, StitchBufferInfo> &stitch_buffer_map,
    const std::unordered_map<Var, StoreWithLoopVar, air::NodeHash, air::NodeEqual> &store_with_loopvar)
      : loops_(loops), vars_(vars), stitch_buffer_map_(stitch_buffer_map), store_with_loopvar_(store_with_loopvar) {}

  void Visit_(const Load *op) final {
    Var var = op->buffer_var;
    auto name = var->name_hint;
    for (auto &kv : stitch_buffer_map_) {
      if (kv.second.name == name || kv.first == name) {
        auto info = kv.second;
        Var replace = GetReplaceVar(var, vars_, kv.first, info);
        for (auto &kv2 : store_with_loopvar_) {
          if (Equal(kv2.first, replace)) {
            auto old_loopvars = kv2.second.loopvars;
            std::unordered_map<const Variable *, Expr> varmap;
            size_t i = 0;
            for (auto &old_loopvar : old_loopvars) {
              CHECK(loops_.size() > i);
              varmap[old_loopvar.get()] = loops_[i]->loop_var;
              i++;
            }
            substitute_[op->index] = Substitute(kv2.second.old_index, varmap);
            return;
          }
        }
      }
    }
  }

 public:
  std::unordered_map<Expr, Expr, air::NodeHash, air::NodeEqual> substitute_;

 private:
  std::vector<const For *> loops_;
  std::unordered_map<std::string, Var> vars_;
  std::unordered_map<std::string, StitchBufferInfo> stitch_buffer_map_;
  std::unordered_map<Var, StoreWithLoopVar, air::NodeHash, air::NodeEqual> store_with_loopvar_;
};

class StitchMutate : public IRMutator {
 public:
  explicit StitchMutate(std::unordered_map<std::string, StitchBufferInfo> &stitch_buffer_map,
                        std::unordered_map<std::string, StitchBufferInfo> &buf_within_op_map,
                        std::vector<std::string> &allocate_revoke, StitchAttrInfo &store_attr)
      : stitch_buffer_map_(stitch_buffer_map),
        buf_within_op_map_(buf_within_op_map),
        allocate_revoke_(allocate_revoke),
        store_attr_(store_attr) {}

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

  void Mapping1Dto2D() {
    if (stitch_type_ <= StitchOpType::Broadcast && broadcast_substitute_.empty()) {
      substitute_[thread_idx_x.get()] = (thread_idx_y * blockdim_x) + thread_idx_x;
    }
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    loops_.emplace_back(op);
    if (!op->body.as<For>() && stitch_type_ == StitchOpType::Broadcast) {
      auto f = BroadcastSubstitute(loops_, vars_, stitch_buffer_map_, store_with_loopvar_);
      f.Visit(s);
      broadcast_substitute_ = f.substitute_;
    }
    auto stmt = IRMutator::Mutate_(op, s);
    loops_.pop_back();
    return stmt;
  }
  Expr GetIndex(const Expr &index) {
    if (stitch_type_ == StitchOpType::Broadcast && !broadcast_substitute_.empty()) {
      auto f = GetBlockExpr();
      f.Visit(index);
      for (auto &kv : broadcast_substitute_) {
        if (Equal(kv.first, index)) {
          auto f2 = GetLoopLen(*loops_.begin());
          f2.Visit(kv.second);
          return kv.second + f.blockexpr_ * f2.len_;
        }
      }
    }
    return index;
  }
  void CollectStoreWithLoopVar(Stmt &stmt) {
    StoreWithLoopVar swlv;
    swlv.old_index = stmt.as<Store>()->index;
    for (auto &l : loops_) {
      swlv.loopvars.emplace_back(l->loop_var);
    }
    store_with_loopvar_[stmt.as<Store>()->buffer_var] = swlv;
  }
  Stmt Mutate_(const Store *op, const Stmt &s) final {
    Var var = op->buffer_var;
    auto name = var->name_hint;
    auto index = GetIndex(op->index);
    if (stitch_buffer_map_.count(name)) {
      auto info = stitch_buffer_map_[name];
      if (info.type == StorageType::Shared) {
        auto shared_name = info.buf_name;
        bool new_buffer = !vars_.count(shared_name);
        Var shared = new_buffer ? Var(shared_name) : vars_[shared_name];
        vars_[shared_name] = shared;
        stitch_buffer_map_[shared_name] = info;
        fix_producer_ = true;
        auto stmt = Store::make(shared, this->Mutate(op->value), this->Mutate(index), op->predicate);
        fix_producer_ = false;
        if (!loops_.empty()) {
          CollectStoreWithLoopVar(stmt);
        }
        if (new_buffer) new_allocate_.insert(stmt.as<Store>());
        return stmt;
      } else {
        vars_[name] = var;
      }
    }
    if (stitch_type_ == StitchOpType::Broadcast)
      return Store::make(op->buffer_var, this->Mutate(op->value), this->Mutate(index), op->predicate);
    auto stmt = IRMutator::Mutate_(op, s);
    for (auto &kv : stitch_buffer_map_) {
      if (kv.second.buf_name == name) {
        if (!loops_.empty()) {
          CollectStoreWithLoopVar(stmt);
        }
      }
    }
    Mapping1Dto2D();
    return stmt;
  }

  Expr Mutate_(const Load *op, const Expr &e) final {
    Var var = op->buffer_var;
    auto name = var->name_hint;
    auto index = GetIndex(op->index);
    for (auto &kv : stitch_buffer_map_) {
      if (kv.second.name == name || kv.first == name) {
        auto info = kv.second;
        Var replace = GetReplaceVar(var, vars_, kv.first, info);
        if (info.type == StorageType::Shared || info.type == StorageType::Global) {
          fix_consumer_ = true;
          index = this->Mutate(index);
          fix_consumer_ = false;
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
    if (fix_producer_ || fix_consumer_) {
      if (IsBlockIdx(name)) return 0;
    }
    if (IsBlockIdxX(name)) {
      if (store_attr_.switch_x_2_y) {
        return block_idx_y;
      }
      return block_idx_x;
    }
    // substitute idx
    if (IsBlockIdxX(name)) return block_idx_x;
    if (IsBlockIdxY(name)) return block_idx_y;
    if (IsBlockIdxZ(name)) return block_idx_z;
    if (IsThreadIdxX(name)) return thread_idx_x;
    if (IsThreadIdxY(name)) return thread_idx_y;
    if (IsThreadIdxZ(name)) return thread_idx_z;
    return IRMutator::Mutate_(op, e);
  }

 public:
  std::unordered_set<const Store *> new_allocate_;
  std::unordered_map<NodeRef, Expr, air::NodeHash, air::NodeEqual> itervars_;
  std::unordered_map<const Variable *, Expr> substitute_;
  Expr condition;
  bool add_condition{false};
  size_t phase{0};

 private:
  bool fix_producer_{false};
  bool fix_consumer_{false};
  std::unordered_map<Expr, Expr, air::NodeHash, air::NodeEqual> broadcast_substitute_;
  std::unordered_map<std::string, StitchBufferInfo> &stitch_buffer_map_;
  std::unordered_map<std::string, StitchBufferInfo> &buf_within_op_map_;
  std::vector<std::string> &allocate_revoke_;
  std::vector<const For *> loops_;
  std::unordered_map<Var, StoreWithLoopVar, air::NodeHash, air::NodeEqual> store_with_loopvar_;
  StitchAttrInfo &store_attr_;
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

Stmt EmitNewAllocate(Stmt &stmt, const std::unordered_set<const Store *> &new_allocate,
                     std::unordered_map<std::string, StitchBufferInfo> &stitch_buffer_map) {
  for (auto &store : new_allocate) {
    CHECK(stitch_buffer_map.count(store->buffer_var->name_hint));
    stmt = Allocate::make(store->buffer_var, Float(32),
                          {static_cast<int>(stitch_buffer_map[store->buffer_var->name_hint].alloc_size)}, const_true(),
                          stmt);
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
      CHECK(stitch_attr_info.broadcast_size.defined());
      ir_attr_info.broadcast_size = stitch_attr_info.broadcast_size;
      ir_attr_info.dims.blockdim_x = dim_array[0].blockdim_x * dim_array[0].blockdim_y * dim_array[0].blockdim_z;
      ir_attr_info.dims.griddim_x = grid_dims;
      ir_attr_info.block_dims = ir_attr_info.dims.blockdim_x;
      if (dim_array[0].griddim_x == 1 && dim_array[0].griddim_y > 1) {
        ir_attr_info.switch_x_2_y = true;
      }
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
      ir_attr_info.dims.blockdim_x = ir_attr_info.block_dims;
      ir_attr_info.dims.griddim_x = grid_dims;
      ir_attr_info.elemwise_size = stitch_attr_info.elemwise_size;
      if (dim_array[0].griddim_x == 1 && dim_array[0].griddim_y > 1) {
        ir_attr_info.switch_x_2_y = true;
      }
      break;
    default:
      auto dims = dim_array.back();
      ir_attr_info.dims = dims;
      ir_attr_info.block_dims = dims.blockdim_x * dims.blockdim_y * dims.blockdim_z;
  }
  // special attr for softmax.
  if (AvgType(type_array) == static_cast<int>(StitchOpType::Reduce2D_X) &&
      static_cast<int>(type) <= static_cast<int>(StitchOpType::Broadcast)) {
    // softmax case.
    // compute dim attr.
    auto elemwise_size = stitch_attr_info.elemwise_size;
    auto l1_tile = elemwise_size.as<IntImm>()->value / grid_dims;
    CHECK(!stitch_attr_info.loop_extent.empty()) << "No Loop Exists in IR";
    CHECK_GT(stitch_attr_info.loop_extent[0].as<IntImm>()->value, 0) << "Loop Extent should be greater than zero!";
    auto l0_tile = l1_tile / stitch_attr_info.loop_extent[0].as<IntImm>()->value;
    std::string band_idx = "0";
    std::string axis_idx = "0";

    std::string dim_string = band_idx + " " + axis_idx + " " + std::to_string(l1_tile) + " " + std::to_string(l0_tile);

    ir_attr_info.attrs.Set("dim", StringImm::make(dim_string));
    ir_attr_info.attrs.Set("use_shared_memory", Expr(0));
    ir_attr_info.attrs.Set("use_register_memory", Expr(0));
  }
  return ir_attr_info;
}

Stmt StitchFusionGpu(std::vector<Stmt> &stitch_irs, const std::string &kernel_name, StitchAttrInfo &store_attr,
                     std::unordered_map<std::string, StitchBufferInfo> &stitch_buffer_map,
                     std::unordered_map<std::string, StitchBufferInfo> &buf_within_op_map,
                     std::vector<std::string> &allocate_revoke) {
  DumpStitchInfo(kernel_name, store_attr, stitch_buffer_map, buf_within_op_map, allocate_revoke);
  DumpStmt2File("stitch_info/" + kernel_name + "_before_stitch.cc", Block::make(stitch_irs));
  auto func = StitchMutate(stitch_buffer_map, buf_within_op_map, allocate_revoke, store_attr);
  CHECK(stitch_irs.size() > 1);
  size_t i = 0;
  for (auto &ir : stitch_irs) {
    auto f = GridBlockDimsAttr();
    f.Visit(ir);
    func.phase = i;
    ir = func.Run(ir);
    if (func.add_condition) {
      ir = AddCondition(func.condition).Run(ir);
      func.add_condition = false;
    }
    if (f.dims.blockdim_y == 1 && !func.substitute_.empty()) {
      ir = Substitute(ir, func.substitute_);
      func.substitute_.clear();
    }
    ++i;
    ir = Block::make(ir, Evaluate::make(Expr("=============split===============")));
  }
  auto stmt = Block::make(stitch_irs);
  stmt = EmitNewAllocate(stmt, func.new_allocate_, stitch_buffer_map);
  stmt = EmitUnifyIterVars(stmt, func.itervars_);
  DumpStmt2File("stitch_info/" + kernel_name + "_after_stitch.cc", stmt);
  stmt = Simplify(stmt);
  stmt = RemoveNoOp(stmt);
  DumpStmt2File("stitch_info/" + kernel_name + "_after_stitch_simplify.cc", stmt);
  return stmt;
}
}  // namespace akg
