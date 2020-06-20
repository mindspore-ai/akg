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
#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <ir_pass.h>
#include <algorithm>

namespace akg {
namespace ir {
class FindReduce : public IRVisitor {
 public:
  void Execute(const Stmt &stmt) {
    this->Visit(stmt);
    if (reduce_arr_.size() == 2) {
      this->SelectSameReduce();
    }
    if (output_arr_.size() == 2) {
      this->SelectSameOutput();
    }
  }

  std::unordered_set<const AttrStmt *> reduce_set_;
  std::unordered_set<const AttrStmt *> output_set_;

 private:
  bool IsSameShape(const AttrStmt *a, const AttrStmt *b) {
    auto it_a = a->body.as<For>();
    auto it_b = b->body.as<For>();
    while (it_a && it_b) {
      if (!Equal(it_a->min, it_b->min) || !Equal(it_a->extent, it_b->extent)) {
        return false;
      }
      it_a = it_a->body.as<For>();
      it_b = it_b->body.as<For>();
    }
    return !it_a && !it_b;
  }

  void SelectSameShape(std::vector<const AttrStmt *> &arr, std::unordered_set<const AttrStmt *> &set) {
    if (!arr.empty()) {
      auto target = arr.front();
      set.insert(target);
      for (auto it = arr.begin() + 1; it != arr.end(); it++) {
        if (IsSameShape(target, *it)) {
          set.insert(*it);
        }
      }
    }
  }
  void SelectSameReduce() { SelectSameShape(reduce_arr_, reduce_set_); }

  void SelectSameOutput() { SelectSameShape(output_arr_, output_set_); }

  const Store *ObtainStoreFromAttr(const AttrStmt *op) {
    if (op == nullptr) {
      return nullptr;
    }

    if (auto for_it = op->body.as<For>()) {
      while (for_it->body.as<For>()) {
        for_it = for_it->body.as<For>();
      }
      if (auto store = for_it->body.as<Store>()) {
        return store;
      }
      if (auto attr_stmt = for_it->body.as<AttrStmt>()) {
        if (auto store = attr_stmt->body.as<Store>()) {
          return store;
        }
      }
    }
    return nullptr;
  }

  void Visit_(const AttrStmt *op) final {
    if (auto attr_val = op->value.as<StringImm>()) {
      if (auto store = ObtainStoreFromAttr(op)) {
        // find reduce
        if (attr_val->value == "vec_binary_add") {
          if (auto add = store->value.as<Add>()) {
            if (auto load = add->a.as<Load>()) {
              if (Equal(load->index, store->index) && store->buffer_var.get() && load->buffer_var.get() &&
                  store->buffer_var->name_hint == load->buffer_var->name_hint) {
                if (reduce_scope_depth_ < 0) {
                  reduce_scope_depth_ = attr_depth_;
                }
                if (attr_depth_ == reduce_scope_depth_) {
                  reduce_arr_.push_back(op);
                }
              }
            }
          }
        } else if (attr_val->value == "dma_copy") {
          // find corresponding output
          if (auto load = store->value.as<Load>()) {
            if (store->buffer_var.get() && load->buffer_var.get() && !reduce_arr_.empty() &&
                store->buffer_var->name_hint + "_local_UB" == load->buffer_var->name_hint) {
              auto reduce_store = ObtainStoreFromAttr(reduce_arr_.back());
              if (reduce_store && reduce_store->buffer_var.get() &&
                  reduce_store->buffer_var->name_hint == load->buffer_var->name_hint) {
                if (output_scope_depth_ < 0) {
                  output_scope_depth_ = attr_depth_;
                }
                if (attr_depth_ == output_scope_depth_) {
                  output_arr_.push_back(op);
                }
              }
            }
          }
        } else if (attr_val->value == "dma_atomic_add") {
          if (auto add = store->value.as<Add>()) {
            if (auto load = add->b.as<Load>()) {
              if (store->buffer_var.get() && load->buffer_var.get() && !reduce_arr_.empty() &&
                  store->buffer_var->name_hint + "_local_UB" == load->buffer_var->name_hint) {
                auto reduce_store = ObtainStoreFromAttr(reduce_arr_.back());
                if (reduce_store && reduce_store->buffer_var.get() &&
                    reduce_store->buffer_var->name_hint == load->buffer_var->name_hint) {
                  if (output_scope_depth_ < 0) {
                    output_scope_depth_ = attr_depth_;
                  }
                  if (attr_depth_ == output_scope_depth_) {
                    output_arr_.push_back(op);
                  }
                }
              }
            }
          }
        }
      }
    }
    IRVisitor::Visit_(op);
  }

  void Visit_(const For *op) final {
    if (op->body.as<AttrStmt>()) {
      return;
    }
    attr_depth_++;
    IRVisitor::Visit_(op);
    attr_depth_--;
  }

  int reduce_scope_depth_{-1};
  int output_scope_depth_{-1};
  int attr_depth_{0};
  std::vector<const AttrStmt *> reduce_arr_;
  std::vector<const AttrStmt *> output_arr_;
};

class MoveReduce : public IRMutator {
 public:
  Stmt Execute(const Stmt &stmt, const std::unordered_set<const AttrStmt *> &reduce_set,
               const std::unordered_set<const AttrStmt *> &output_set) {
    this->reduce_set_ = reduce_set;
    this->output_set_ = output_set;
    return this->Mutate(stmt);
  }

 private:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (!reduce_set_.empty()) {
      auto found = reduce_set_.find(op);
      if (found != reduce_set_.end()) {
        reduce_arr_.push_back(AttrStmt::make(op->node, op->attr_key, op->value, op->body));
        if (reduce_count_ == reduce_set_.size() - 1) {
          auto block = Block::make(reduce_arr_);
          return AttrStmt::make(op->node, op->attr_key, Expr("reduce_reorder"), block);
        } else {
          reduce_count_++;
          return Evaluate::make(0);
        }
      }
    }
    if (!output_set_.empty()) {
      auto found = output_set_.find(op);
      if (found != output_set_.end()) {
        output_arr_.push_back(AttrStmt::make(op->node, op->attr_key, op->value, op->body));
        if (output_count_ == output_set_.size() - 1) {
          auto block = Block::make(output_arr_);
          return block;
        } else {
          output_count_++;
          return Evaluate::make(0);
        }
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  unsigned int reduce_count_{0};
  unsigned int output_count_{0};

  std::unordered_set<const AttrStmt *> reduce_set_;
  std::unordered_set<const AttrStmt *> output_set_;

  std::vector<Stmt> reduce_arr_;
  std::vector<Stmt> output_arr_;
};

class ExecuteInEachScope : public IRMutator {
 public:
  Stmt Mutate_(const For *op, const Stmt &s) final {
    auto stmt = IRMutator::Mutate_(op, s);
    if (op->body.as<AttrStmt>() || (op->body.as<Block>() && op->body.as<Block>()->first.as<AttrStmt>())) {
      FindReduce find;
      find.Execute(op->body);
      if (find.reduce_set_.size() == 2 && find.output_set_.size() == 2) {
        stmt = MoveReduce().Execute(stmt, find.reduce_set_, find.output_set_);
      }
    }
    return stmt;
  }
};

Stmt AutoReorder(Stmt stmt) {
  stmt = ExecuteInEachScope().Mutate(stmt);
  return stmt;
}
}  // namespace ir
}  // namespace akg
