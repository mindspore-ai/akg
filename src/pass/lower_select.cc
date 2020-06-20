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

#include <ir_pass.h>
#include <tvm/ir_mutator.h>

#define BINARY128 128

namespace akg {
namespace ir {
class LowerSelect : public IRMutator {
 public:
  LowerSelect() = default;
  ~LowerSelect() override = default;

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == "pragma_emit_insn" && op->value.as<StringImm>() &&
        op->value.as<StringImm>()->value == "vec_select") {
      in_attr_stmt_ = true;
      Stmt stmt = IRMutator::Mutate_(op, s);
      const auto opn = stmt.as<AttrStmt>();
      CHECK(opn);
      in_attr_stmt_ = false;
      std::string newTag = std::string("vec_select_") + com_type_;
      com_type_ = "";
      return AttrStmt::make(opn->node, opn->attr_key, newTag, opn->body);
    } else {
      return IRMutator::Mutate_(op, s);
    }
  }

  Stmt Mutate_(const Store *op, const Stmt &s) final {
    if (in_attr_stmt_) {
      static_cast<void>(this->Mutate(op->value));
      if (com_type_ != "scalar") {
        Array<Expr> args(loads_);
        Expr val = Call::make(data_type_, "vselect", args, Call::PureExtern);
        data_type_ = Float(BINARY128);
        loads_.clear();
        return Store::make(op->buffer_var, val, op->index, op->predicate);
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  Expr Mutate_(const Select *op, const Expr &e) override {
    if (in_attr_stmt_) {
      // We normally don't see GE, GT, or LE because Simplify_cce transforms
      // these all to LT.
      if (op->condition->IsInstance<GE>()) {
        com_type_ = "ge";
      } else if (op->condition->IsInstance<GT>()) {
        com_type_ = "gt";
      } else if (op->condition->IsInstance<LT>()) {
        com_type_ = "lt";
      } else if (op->condition->IsInstance<LE>()) {
        com_type_ = "le";
      } else if (op->condition->IsInstance<EQ>()) {
        com_type_ = "eq";
      } else {
        com_type_ = "scalar";
        LOG(FATAL) << "Unexpected select condition";
      }
    }
    return IRMutator::Mutate_(op, e);
  }

  Expr Mutate_(const Load *op, const Expr &e) final {
    if (in_attr_stmt_) {
      if (data_type_ == Float(BINARY128)) {
        data_type_ = op->type;
      }
      loads_.push_back(e);

      // Need to disable in_attr_stmt_ here, because the address expression inside Load may contain
      // immediate numbers and these immediate numbers should NOT be considered as an attribute.
      in_attr_stmt_ = false;
      Expr after_mutate = IRMutator::Mutate_(op, e);
      in_attr_stmt_ = true;
      return after_mutate;
    } else {
      return IRMutator::Mutate_(op, e);
    }
  }

  Expr Mutate_(const FloatImm *op, const Expr &e) final {
    if (in_attr_stmt_) {
      LOG(WARNING) << "Warning: Float Immediate found in vselect operator. If the ISA does not "
                      "support, check ToThreeAddress "
                      "for errors.";
      if (data_type_ == Float(BINARY128)) {
        data_type_ = op->type;
      }
      loads_.push_back(e);
    }
    return IRMutator::Mutate_(op, e);
  }

 private:
  std::string com_type_{""};
  bool in_attr_stmt_{false};
  std::vector<Expr> loads_;
  Type data_type_{Float(BINARY128)};
};

Stmt SelectLower(Stmt stmt) {
  stmt = LowerSelect().Mutate(stmt);
  return stmt;
}
}  // namespace ir
}  // namespace akg
