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
#include <ir_pass.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm/arithmetic.h>
#include <string>

namespace akg {
namespace ir {
#define DEBUG_PASS 0
class MadInitRemover : public IRMutator {
 private:
  Stmt Mutate_(const For *op, const Stmt &s) final {
    in_for_ = true;
    in_block_ = false;
    if (op->body.as<Block>()) {
      in_block_ = true;
    }
    Stmt stmt = IRMutator::Mutate_(op, s);
    in_for_ = false;
    return stmt;
  }

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    if (op->attr_key == "pragma_is_reduce_k_outer") {
      bool found_mad = false;
      PostOrderVisit(stmt, [&found_mad](const NodeRef &node) {
        if (const auto v = node.as<Store>()) {
          if (v->buffer_var->name_hint.find(".local.UB.local.L0C") != std::string::npos) {
            found_mad = true;
          }
        }
      });
      if (!found_mad) {
        if (DEBUG_PASS) {
          LOG(INFO) << "removing all dead computation";
        }
        return Evaluate::make(0);
      }
    }
    return stmt;
  }

  Stmt Mutate_(const Store *op, const Stmt &s) final {
    if (const auto fv = op->value.as<FloatImm>()) {
      if (DEBUG_PASS) {
        LOG(INFO) << "Found a store op with FloatImm with buffer var: " << op->buffer_var->name_hint
                  << " and value: " << op->value << " with type: " << op->value->type_index();
      }
      if (in_for_ && !in_block_ && fv->value == 0.0 &&
          op->buffer_var->name_hint.find(".local.UB.local.L0C") != std::string::npos) {
        if (DEBUG_PASS) {
          LOG(INFO) << "Removing MAD init";
        }
        return Evaluate::make(0);
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  bool in_for_{false};
  bool in_block_{true};
};

class FixMadAttr : public IRMutator {
 private:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == "pragma_mad_pattern" && is_one(op->value)) {
      is_mad_pattern_one_ = true;
      Stmt new_stmt = IRMutator::Mutate_(op, s);
      is_mad_pattern_one_ = false;
      return new_stmt;
    } else if (op->attr_key == "pragma_emit_insn" && op->value.as<StringImm>() &&
               op->value.as<StringImm>()->value == "mad" && is_mad_pattern_one_) {
      in_mad_ = true;
      extents_.clear();
      Stmt new_stmt = IRMutator::Mutate_(op, s);
      in_mad_ = false;
      return new_stmt;
    } else if (op->attr_key == "pragma_mad_m" && in_mad_) {
      CHECK(op->body.as<AttrStmt>()) << "Unsupported body in the mad loop";
      Stmt new_body = this->Mutate(op->body);
      if (DEBUG_PASS) {
        LOG(INFO) << "changing the mad m value: " << extents_[0];
      }
      CHECK(new_body.as<AttrStmt>());
      return AttrStmt::make(op->node, op->attr_key, extents_[0], new_body);
    } else if (op->attr_key == "pragma_mad_k" && in_mad_) {
      Stmt new_body = this->Mutate(op->body);
      CHECK(new_body.as<AttrStmt>()) << "Unsupported body in the mad loop";
      Expr new_k;
      if (extents_.size() == 3) {
        new_k = extents_[2];
      } else if (extents_.size() == 4) {
        new_k = extents_[2] * extents_[3];
      }
      if (DEBUG_PASS) {
        LOG(INFO) << "changing the mad k value: " << new_k;
      }
      return AttrStmt::make(op->node, op->attr_key, new_k, new_body);
    } else if (op->attr_key == "pragma_mad_n" && in_mad_) {
      Stmt new_body = this->Mutate(op->body);
      CHECK(extents_.size() == 3 || extents_.size() == 4) << "Unsupported mad loop";
      if (DEBUG_PASS) {
        LOG(INFO) << "changing the mad n value : " << extents_[1];
      }
      return AttrStmt::make(op->node, op->attr_key, extents_[1], new_body);
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (in_mad_) {
      extents_.push_back(op->extent);
    }
    return IRMutator::Mutate_(op, s);
  }

  bool in_mad_{false};
  bool is_mad_pattern_one_{false};
  std::vector<Expr> extents_;
};

Stmt FixMadAttrs(Stmt stmt) {
  stmt = MadInitRemover().Mutate(stmt);
  stmt = RemoveNoOp(stmt);
  stmt = FixMadAttr().Mutate(stmt);
  return stmt;
}
}  // namespace ir
}  // namespace akg
