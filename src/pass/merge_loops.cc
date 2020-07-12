/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "pass/analyze_align.h"

namespace akg {
namespace ir {
namespace {
int HasNode(const Array<NodeRef> &array, const NodeRef &node) {
  int index = 0;
  for (const auto &item : array) {
    if (item.same_as(node)) {
      return index;
    }
    index++;
  }
  return -1;
}

class LoopsCompacter : public IRMutator {
 public:
  explicit LoopsCompacter(bool is_dynamic) : is_dynamic_(is_dynamic) {}
  ~LoopsCompacter() override = default;

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == "pragma_ub_gm" || (op->attr_key == "pragma_emit_insn" && op->value->IsInstance<StringImm>() &&
                                           !exclude_list.count(op->value.as<StringImm>()->value))) {
      stores_ = Array<NodeRef>();
      loads_ = Array<NodeRef>();
      GetStoreAndLoads(op->body, stores_, loads_);

      StmtInfo if_info;
      StmtInfo for_info;
      GetIfForInfo(op->body, if_info, for_info);

      dst_info_list_ = GetComputationInfo(stores_, for_info);
      src_info_list_ = GetComputationInfo(loads_, for_info);

      CompactComputationInfoList(dst_info_list_, src_info_list_, if_info, for_info);

      for_vars_ = Map<Var, Expr>();

      in_insn_ = true;
      auto ret = IRMutator::Mutate_(op, s);
      in_insn_ = false;

      auto opn = ret.as<AttrStmt>();
      CHECK(opn);
      if (ForVarIsClean(opn->body, for_info)) {
        auto body = PackForLoop(opn->body, for_info);
        return AttrStmt::make(opn->node, opn->attr_key, opn->value, body);
      }
      return s;
    } else if (op->attr_key == "pragma_emit_insn") {
      return s;
    }
    return IRMutator::Mutate_(op, s);
  }

  Expr Mutate_(const Load *op, const Expr &e) final {
    auto load = IRMutator::Mutate_(op, e);
    if (in_insn_) {
      auto idx = HasNode(loads_, e);
      if (idx != -1) {
        CHECK_GT(src_info_list_.size(), idx);
        auto index = GenerateIndex(src_info_list_[idx]);
        auto opn = load.as<Load>();
        CHECK(opn);
        return Load::make(opn->type, opn->buffer_var, index, opn->predicate);
      }
    }
    return load;
  }

  Stmt Mutate_(const Store *op, const Stmt &s) final {
    auto store = IRMutator::Mutate_(op, s);
    if (in_insn_) {
      auto idx = HasNode(stores_, s);
      if (idx != -1) {
        CHECK_GT(dst_info_list_.size(), idx);
        auto index = GenerateIndex(dst_info_list_[idx]);
        auto opn = store.as<Store>();
        CHECK(opn);
        return Store::make(opn->buffer_var, opn->value, index, opn->predicate);
      }
    }
    return store;
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (in_insn_) {
      for_vars_.Set(op->loop_var, Expr(0));
      return this->Mutate(op->body);
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  Expr GenerateIndex(const StmtStoreInfo &info) {
    Expr expr = info->elem_offset_;
    for (size_t i = 0; i != info->var_.size(); ++i) {
      CHECK_GT(info->strides_.size(), i);
      expr += info->var_[i] * info->strides_[i];
    }
    return expr;
  }

  Stmt PackForLoop(Stmt s, const StmtInfo &for_info) {
    for (size_t i = 0; i != for_info.vars_.size(); ++i) {
      auto op = for_info.ops_[i].as<For>();
      CHECK(op);
      s = For::make(for_info.vars_[i], op->min, op->extent, op->for_type, op->device_api, s);
    }
    return s;
  }

  bool ForVarIsClean(const Stmt &s, const StmtInfo &for_info) {
    Map<Var, Expr> map;
    for (const auto &e : for_vars_) {
      if (!IsInArray(for_info.vars_, e.first)) {
        map.Set(e.first, Expr(0));
      }
    }
    for (const auto &e : map) {
      if (air::ir::StmtUseVar(s, e.first)) {
        return false;
      }
    }
    return true;
  }

  Map<Var, Expr> for_vars_;
  Array<NodeRef> stores_;
  Array<NodeRef> loads_;
  StmtInfoList dst_info_list_;
  StmtInfoList src_info_list_;
  bool in_insn_{false};
  bool is_dynamic_{false};
};
}  // namespace

Stmt MergeLoops(const Stmt &stmt, bool is_dynamic) { return LoopsCompacter(is_dynamic).Mutate(stmt); }
}  // namespace ir
}  // namespace akg
