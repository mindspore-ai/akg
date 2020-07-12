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
#include <tvm/ir_visitor.h>
#include <tvm/ir_pass.h>

#include <regex>

#include "ir_pass.h"
#include "pass/utils.h"
#include "pass/expr_alg_simplify.h"

/**
 * ir_before:
 *  // attr [output_local_UB] storage_scope = "local.UB"
 *  allocate output_local_UB[float32 * 1 * 1 * 1 * 16 * 16]
 *  // attr [output_local_UB_local_L0C] storage_scope = "local.L0C"
 *  allocate output_local_UB_local_L0C[float32 * 1 * 1 * 1 * 16 * 16]
 *    ...
 *    // attr [0] pragma_emit_insn = "dma_copy"
 *    for (ee9, 0, 256) {
 *      output_local_UB[ee9] = output_local_UB_local_L0C[ee9]
 *    }
 *    // attr [output_cast_local_UB] storage_scope = "local.UB
 *    allocate output_cast_local_UB[float16 * 1 * 1 * 1 * 16 * 16]
 *      // attr [0] pragma_emit_insn = "vec_single_cast"
 *      for (ee9, 0, 256) {
 *        output_cast_local_UB[ee9] = float16(output0_local_UB[ee9])
 *    }
 *    // attr [0] pragma_emit_insn = "dma_copy"
 *    for (ee9, 0, 256) {
 *      output[ee9] = f(output_cast_local_UB[ee9])
 *    }
 *
 * ir_after:
 *  // attr [output_local_UB] storage_scope = "local.UB"
 *  allocate output_local_UB[float16 * 1 * 1 * 1 * 16 * 16]
 *  // attr [output_local_UB_local_L0C] storage_scope = "local.L0C"
 *  allocate output_local_UB_local_L0C[float32 * 1 * 1 * 1 * 16 * 16]
 *    // attr [0] pragma_emit_insn = "dma_copy_F16"
 *    for (ee9, 0, 256) {
 *      output_local_UB[ee9] = float16(output_local_UB_local_L0C[ee9])
 *     }
 *    // attr [0] pragma_emit_insn = "dma_copy"
 *    for (ee9, 0, 256) {
 *      output[ee9] = f(output_local_UB[ee9])
 *    }
 **/

namespace akg {
namespace ir {
bool EndWith(std::string const &fullString, std::string const &ending) {
  if (fullString.length() >= ending.length()) {
    return (0 == fullString.compare(fullString.length() - ending.length(), ending.length(), ending));
  } else {
    return false;
  }
}

class CastFilterMutator : public IRMutator {
 public:
  class CountLoad : public IRVisitor {
   public:
    CountLoad() {}
    ~CountLoad() override = default;

    void Visit_(const Load *op) final {
      const auto &buffer_var = op->buffer_var.get();
      auto it = load_times_.find(buffer_var);
      if (it != load_times_.end()) {
        it->second += 1;
      } else {
        load_times_.emplace(buffer_var, 1);
      }
      IRVisitor::Visit_(op);
    }
    std::unordered_map<const Variable *, uint> load_times_;
  };

  class FindCast : public IRVisitor {
   public:
    explicit FindCast(const std::unordered_map<const Variable *, uint> &load_times) : load_times_(load_times) {}
    ~FindCast() override = default;

    void Visit_(const Store *op) final {
      if (auto load = op->value.as<Load>()) {
        if (store_load_.count(op->buffer_var.get()) == 0) {
          store_load_.emplace(op->buffer_var.get(), load->buffer_var.get());
        }
      }
      if (const auto cast = op->value.as<Cast>()) {
        if (const auto load = cast->value.as<Load>()) {
          auto it = load_times_.find(load->buffer_var.get());
          CHECK(it != load_times_.end());
          bool is_index_equal = ExprSimplifier().Equals(load->index, op->index);
          if (it->second == 1 && is_index_equal) {
            string load_name = load->buffer_var->name_hint;
            string store_name = op->buffer_var->name_hint;
            if (store_load_.count(load->buffer_var.get()) == 0) {
              return IRVisitor::Visit_(op);
            }
            string src_name = store_load_[load->buffer_var.get()]->name_hint;
            if (((EndWith(load_name, "UB") && EndWith(store_name, "UB")) ||
                 (EndWith(load_name, "L0C") && EndWith(store_name, "L0C"))) &&
                ((EndWith(load_name, "UB") && EndWith(src_name, "L0C")) ||
                 (EndWith(load_name, "L0C") && EndWith(src_name, "UB")))) {
              replace_and_src_.emplace(load->buffer_var.get(), op);
              remove_and_replace_.emplace(op->buffer_var.get(), load->buffer_var);
            }
          }
        }
      }
      IRVisitor::Visit_(op);
    }
    const std::unordered_map<const Variable *, uint> &load_times_;
    std::unordered_map<const Variable *, const Store *> replace_and_src_;
    std::unordered_map<const Variable *, Var> remove_and_replace_;
    std::unordered_map<const Variable *, const Variable *> store_load_;
  };

  Stmt Filter(Stmt s) {
    CountLoad counter;
    counter.Visit(s);
    FindCast finder(counter.load_times_);
    finder.Visit(s);
    replace_and_src_ = finder.replace_and_src_;
    remove_and_replace_ = finder.remove_and_replace_;

    if (replace_and_src_.empty()) {
      return s;
    }
    return Mutate(s);
  }

 private:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (air::ir::attr::IsPragmaKey(op->attr_key) && op->attr_key == "pragma_emit_insn") {
      auto pragma = op->value.as<StringImm>()->value;
      if (pragma == "vec_single_cast") {
        cast_can_remove_ = false;
        auto stmt = IRMutator::Mutate_(op, s);
        if (cast_can_remove_) {
          cast_can_remove_ = false;
          return Evaluate::make(Expr(0));
        }
        return stmt;
      }
    } else if (op->attr_key == "storage_scope") {
      buffer_can_replace_ = false;
      auto body = Mutate(op->body);
      if (buffer_can_replace_) {
        buffer_can_replace_ = false;
        return body;
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Store *op, const Stmt &s) final {
    auto it = replace_and_src_.find(op->buffer_var.get());
    if (it != replace_and_src_.end()) {
      return Store::make(op->buffer_var, Cast::make(it->second->value.type(), op->value), op->index, op->predicate);
    }
    return IRMutator::Mutate_(op, s);
  }

  Expr Mutate_(const Load *op, const Expr &e) final {
    if (replace_and_src_.count(op->buffer_var.get())) {
      cast_can_remove_ = true;
    } else {
      auto it = remove_and_replace_.find(op->buffer_var.get());
      if (it != remove_and_replace_.end()) {
        return Load::make(op->type, it->second, op->index, op->predicate);
      }
    }
    return IRMutator::Mutate_(op, e);
  }

  Stmt Mutate_(const Allocate *op, const Stmt &s) final {
    auto it = replace_and_src_.find(op->buffer_var.get());
    if (it != replace_and_src_.end()) {
      auto body = IRMutator::Mutate(op->body);
      return Allocate::make(op->buffer_var, it->second->value.type(), op->extents, op->condition, body, op->new_expr,
                            op->free_function);
    } else if (remove_and_replace_.count(op->buffer_var.get())) {
      buffer_can_replace_ = true;
      return IRMutator::Mutate(op->body);
    }
    return IRMutator::Mutate_(op, s);
  }

  std::unordered_map<const Variable *, uint> load_times_;
  std::unordered_map<const Variable *, const Store *> replace_and_src_;
  std::unordered_map<const Variable *, Var> remove_and_replace_;
  bool buffer_can_replace_ = false;
  bool cast_can_remove_ = false;
};

Stmt CastFilter(const Stmt &stmt) { return CastFilterMutator().Filter(stmt); }
}  // namespace ir
}  // namespace akg
