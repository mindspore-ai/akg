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
#include <pass/ir_util.h>
#include <emit_insn/insn_info.h>
#include <emit_insn/cce_params.h>
#include <algorithm>
#include "pass/expr_alg_simplify.h"
#include "pass/analyze_align.h"

namespace akg {
namespace ir {
class OptPragma : public IRMutator {
 public:
  explicit OptPragma(bool is_simple_addr) : is_simple_addr_(is_simple_addr) {}
  ~OptPragma() override = default;

  Stmt Run(Stmt s) {
    Stmt ret = this->Mutate(s);
    return ret;
  }

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (air::ir::attr::IsPragmaKey(op->attr_key) && op->value.as<StringImm>()) {
      is_candidate_ = true;
      loop_vars_ = {};
      loop_extends_ = {};
      is_broadcast_ = false;
      old_pragma_ = op->value.as<StringImm>()->value;

      static_cast<void>(this->Mutate(op->body));
      is_candidate_ = false;
      if (is_broadcast_) {
        std::string new_pragma = old_pragma_;
        if (old_pragma_ == "broadcast") {
          new_pragma = "mask_broadcast";
        } else if (old_pragma_ == "dma_copy" && is_simple_addr_) {
          new_pragma = "opt_broadcast";
        }
        return AttrStmt::make(op->node, op->attr_key, Expr(new_pragma), op->body);
      }
      return AttrStmt::make(op->node, op->attr_key, op->value, op->body);
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (is_candidate_) {
      loop_vars_.push_back(op->loop_var);
      loop_extends_.push_back(op->extent);
      Stmt body = this->Mutate(op->body);
      return For::make(op->loop_var, op->min, op->extent, op->for_type, op->device_api, body);
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Store *op, const Stmt &s) final {
    if (op->value.as<Load>() && old_pragma_ == "dma_copy") {
      is_broadcast_ = IsLastDimBroadcast(op);
    } else if (old_pragma_ == "broadcast") {
      is_broadcast_ = IsMaskBroadcast(op);
    }
    return s;
  }

  bool IsLastDimBroadcast(const Store *op) {
    bool flag = false;
    if (op->value.as<Load>() && op->buffer_var.get() && op->value.as<Load>()->buffer_var.get() &&
        GetBufScope(op->buffer_var->name_hint) == SCOPE_UBUF &&
        GetBufScope(op->value.as<Load>()->buffer_var->name_hint) == SCOPE_UBUF) {
      int block_size = GetUbBlkSize(op->value.type());
      int dst_pos = GetVectorizedVarPosition(op->index, loop_vars_);
      int src_pos = GetVectorizedVarPosition(op->value.as<Load>()->index, loop_vars_);
      if (dst_pos >= 0 && dst_pos != src_pos && src_pos != -1 && loop_vars_.size() >= 2 &&
          !HasVars(op->value.as<Load>()->index, loop_vars_[dst_pos]) && loop_extends_[dst_pos].as<IntImm>() &&
          loop_extends_[dst_pos].as<IntImm>()->value % block_size != 0) {
        flag = true;
      }
    }
    return flag;
  }

  bool IsMaskBroadcast(const Store *op) {
    bool flag = false;
    CHECK(op->buffer_var.get());
    CHECK(GetBufScope(op->buffer_var->name_hint) == SCOPE_UBUF);
    int dst_pos = GetVectorizedVarPosition(op->index, loop_vars_);
    if (dst_pos < 0) {
      return flag;
    }
    int block_size = GetUbBlkSize(op->value.type());
    auto index_mod = ExprSimplifier().Simplify(Mod::make(op->index - loop_vars_[dst_pos], block_size));
    if (!index_mod.as<IntImm>()) {
      return flag;
    }
    if (auto load = op->value.as<Load>()) {
      CHECK(load->buffer_var.get());
      CHECK(GetBufScope(load->buffer_var->name_hint) == SCOPE_UBUF);
      int src_pos = GetVectorizedVarPosition(load->index, loop_vars_);
      if (dst_pos >= 0 && dst_pos != src_pos && !HasVars(op->value.as<Load>()->index, loop_vars_[dst_pos]) &&
          loop_extends_[dst_pos].as<IntImm>() && loop_extends_[dst_pos].as<IntImm>()->value % block_size != 0) {
        flag = true;
      }
    } else if (dst_pos >= 0 && loop_extends_[dst_pos].as<IntImm>() &&
               loop_extends_[dst_pos].as<IntImm>()->value % block_size != 0) {
      flag = true;
    }
    return flag;
  }

  std::string old_pragma_;
  Array<Var> loop_vars_;
  Array<Expr> loop_extends_;
  bool is_broadcast_{false};
  bool is_candidate_{false};
  bool is_simple_addr_{true};
  int cnt_store_{0};
};

class EstimateAlign : public IRMutator {
 public:
  bool IsSimpleAddress(const Stmt &stmt) {
    Mutate(stmt);
    // Returns true only when the numbers of Store in IR that is not elementwise
    // is only 1 or less, in this case, we can consider optimizing broadcast by
    // using variable length mask in insn emitting pass safely because at most
    // 1 Store does not need to cosider block alignment.
    return (not_simple_addressing_cnt_ < 2);
  }

  Stmt Mutate_(const AttrStmt *op, const Stmt &stmt) final {
    if (air::ir::attr::IsPragmaKey(op->attr_key) && op->value.as<StringImm>()) {
      if (exclude_list.count(op->value.as<StringImm>()->value)) {
        return stmt;
      }

      StmtInfoList dst_info_list, src_info_list;
      StmtInfo if_info, for_info;
      GetCompactComputationInfo(op->body, dst_info_list, src_info_list, if_info, for_info, false);

      if (!src_info_list.empty() && !IsElementwise(dst_info_list, src_info_list)) {
        not_simple_addressing_cnt_++;
      }
    }
    return IRMutator::Mutate_(op, stmt);
  }

  int not_simple_addressing_cnt_{0};  // records the number of stores that are not elementwise
};

Stmt OptimizePragma(Stmt stmt) {
  bool is_simple_addr = EstimateAlign().IsSimpleAddress(stmt);
  return OptPragma(is_simple_addr).Run(stmt);
}
}  // namespace ir
}  // namespace akg
