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
#include <emit_insn/cce_params.h>
#include "pass/utils.h"

namespace akg {
namespace ir {
enum TYPE { NONE, SINGLE, BINARY };
// recognize Op of Store
class OpRecog : public IRVisitor {
 public:
  std::string Tolower(std::string str) {
    std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) { return std::tolower(c); });
    return str;
  }

  std::string Run(const Stmt &stmt) {
    op_.clear();
    store_ = NONE;
    this->Visit(stmt);
    if (store_ > SINGLE) {
      op_ = "scalar";
    }
    return Tolower(op_);
  }

  bool indexLastDimIsLoad(const Load *load) {
    if (load) {
      return (load->index.as<Load>() || (load->index.as<Add>() && load->index.as<Add>()->b.as<Load>()));
    }
    return false;
  }

  void Visit_(const Store *op) final {
    CHECK(op);
    auto type_index = op->value->type_index();
    op_ = Node::TypeIndex2Key(type_index);
    if (op_ == "Call") {
      const auto c = op->value.as<Call>();
      CHECK(c);
      op_ = c->name;
      is_call_ = true;
    }

    store_++;
    ++index_depth_;
    this->Visit(op->index);
    --index_depth_;
    this->Visit(op->predicate);
    in_store_ = true;
    this->Visit(op->value);
    in_store_ = false;

    if (load_ > SINGLE && op_ == "Cast") {
      const auto cast = op->value.as<Cast>();
      CHECK(cast);
      type_index = cast->value->type_index();
      op_ = Node::TypeIndex2Key(type_index);
    }

    bool imm_lhs = is_const(op->index);
    if (index_depth_ > NONE) {
      op_ = "scatter";
    } else if (op_ == "Select") {
      const auto select = op->value.as<Select>();
      CHECK(select);
      select_store_ = NONE;
      in_select_ = true;
      this->Visit(select->true_value);
      this->Visit(select->false_value);
      in_select_ = false;
      op_ = "vec_select_scalar";
    } else if (load_ == SINGLE && (is_call_ || op_ == "Cast")) {
      op_ = "vec_single_" + op_;
    } else if ((load_ > SINGLE && op_ == "Load") || (op_ == "Load" && indexLastDimIsLoad(op->value.as<Load>()))) {
      // lastDimIsLoad: A[cc0] = B[cc1*10 + reg0_local_REG[0]]
      op_ = "scatter";
    } else if (imm_lhs && load_ == BINARY && immNum_ == NONE && scalar_loads_ == SINGLE && op_ == "Add") {
      // A[0] = (B[0] + C[cc1])
      op_ = "vec_binary_" + op_;
    } else if (((load_ == BINARY && immNum_ == NONE && scalar_loads_ == SINGLE)       // A[cc1] = B[cc1] + C[0]
                || (load_ == SINGLE && immNum_ == SINGLE && scalar_loads_ == NONE)    // A[cc1] = B[cc1] + 0.1
                || (load_ == SINGLE && immNum_ == SINGLE && scalar_loads_ == SINGLE)  // A[0] = (B[0]*-1.0h)
                || (load_ == SINGLE && varNum_ == SINGLE))                            // A[cc1] = B[cc1] *  var
               && (op_ == "Add" || op_ == "Mul")) {
      if (GetBufScope(op->buffer_var->name_hint) == SCOPE_REG) {
        // value_local_REG[0] = cc3*1016 + input_2_local_UB[cc4]
        op_ = "scatter";
      } else {
        op_ = "vec_single_" + op_ + "s";
      }
    } else if (((load_ == SINGLE && immNum_ == SINGLE && scalar_loads_ == NONE) ||
                (load_ == SINGLE && immNum_ == SINGLE && scalar_loads_ == SINGLE)) &&
               (op_ == "Max") && (op->value.type().is_float())) {
      op_ = "vec_single_relu";
    } else if (load_ == SINGLE && immNum_ == NONE && varNum_ == NONE && !is_call_) {
      if (is_const(op->value) || (is_const(op->index) && GetBufScope(op->buffer_var->name_hint) == SCOPE_REG)) {
        // value_local_UB[0] = xxx
        op_ = "scatter";
      } else {
        auto load = op->value.as<Load>();
        if (load && IsConstExpr(load->index) && GetBufScope(op->buffer_var->name_hint) == SCOPE_UBUF &&
            GetBufScope(load->buffer_var->name_hint) == SCOPE_UBUF) {
          // value_local_UB[cc1] = other_local_UB[0]
          // value_local_UB[cc1] = other_local_UB[ccx] can also pragma as broadcast
          auto vars = GetVarsInExpr(op->index);
          auto strides = air::arith::DetectLinearEquation(op->index, vars);
          strides = RemoveItemAtIndex(strides, -1);
          if (!strides.empty() && (GetInt32Const(GetItem(strides, -1)) % GetUbBlkSize(op->value.type())) == 0) {
            op_ = "broadcast";
          } else {
            // Below case if pragma as broadcast, will cause ub inflation, so currently keep using dma_copy
            // for (cc3, 0, 2183) {
            //   for (c6, 0, 6) {
            //     one_hot_hybrid_2_local_UB[((cc3*6) + c6)] = input_3_local_UB[0]
            //   }
            // }
            op_ = "dma_copy";
          }
        } else {
          // value[xx] = value_local_UB[xx]
          // value_local_UB[cc1] = value[cc1]
          op_ = "dma_copy";
        }
      }
    } else if ((immNum_ == SINGLE || is_const(op->value))) {
      if (op->value.as<Call>() && op->value.as<Call>()->name == "nms") {
        op_ = "vec_binary_nms";
      } else if (op->value.as<Call>() && op->value.as<Call>()->name == "vaxpy") {
        op_ = "vaxpy";
      } else if (load_ == NONE) {
        op_ = "broadcast";
      } else {
        op_ = "scatter";
      }
    } else if (load_ >= BINARY && immNum_ == NONE) {
      if (GetBufScope(op->buffer_var->name_hint) == DMA_COPY_GLOBAL && op_ == "Add") {
        op_ = "dma_atomic_add";
      } else {
        op_ = "vec_binary_" + op_;
      }
    } else if ((load_ + immNum_ == NONE) || (load_ == SINGLE && varNum_ == SINGLE)) {
      // A = var.
      op_ = "scalar_dma";
    }
  }

  void Visit_(const Call *op) final {
    if (index_depth_ > NONE &&
        (op->call_type != Call::CallType::PureIntrinsic || (op->name != "shift_right" && op->name != "shift_left"))) {
      ++index_depth_;
    }
    IRVisitor::Visit_(op);
  }

  void Visit_(const Load *op) final {
    if (index_depth_ > NONE && GetBufScope(op->buffer_var->name_hint) == SCOPE_REG) {
      return;
    }
    if (index_depth_ > NONE && (!is_constant(op->index) || !in_store_)) {
      ++index_depth_;
    }
    if (in_select_) {
      select_store_++;
    } else if (in_store_) {
      // B[0] --> treat as imm
      if (is_const(op->index)) {
        scalar_loads_++;
      }
      load_++;
      load_depth_++;
    }

    ++index_depth_;
    this->Visit(op->index);
    --index_depth_;
    this->Visit(op->predicate);

    if (!in_select_ && in_store_) {
      load_depth_--;
    }
  }

  void Visit_(const FloatImm *op) final {
    if (in_store_) {
      immNum_++;
    }
    IRVisitor::Visit_(op);
  }

  void Visit_(const IntImm *op) final {
    if (in_store_ && !is_call_ && load_depth_ == NONE && index_depth_ == NONE) {
      immNum_++;
    }
    IRVisitor::Visit_(op);
  }

  void Visit_(const UIntImm *op) final {
    if (in_store_ && !is_call_ && load_depth_ == NONE && index_depth_ == NONE) {
      immNum_++;
    }
    IRVisitor::Visit_(op);
  }

  void Visit_(const Variable *op) final {
    if (in_store_ && !is_call_ && load_depth_ == NONE && index_depth_ == NONE) {
      varNum_++;
    }
    IRVisitor::Visit_(op);
  }

 private:
  bool in_select_{false};
  bool in_store_{false};
  bool is_call_{false};
  int load_{0};
  int load_depth_{0};
  int index_depth_{0};
  int immNum_{0};
  int varNum_{0};
  std::string op_;
  int store_{0};
  int select_store_{0};
  int scalar_loads_{0};
};

class InjectPragma : public IRMutator {
  // for those have been marked, we just ignore it
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) override {
    if (op->attr_key == "pragma_emit_insn" || op->attr_key == "pragma_im2col" || op->attr_key == "pragma_fractal" ||
        op->attr_key == "pragma_filter" || op->attr_key == "pragma_ub_gm") {
      return s;
    } else if (op->attr_key == "pragma_load3d" && op->body.as<For>()) {
      return AttrStmt::make(op->node, op->attr_key, op->value, op->body);
    }
    return IRMutator::Mutate_(op, s);
  }

  // for store not in loops
  Stmt Mutate_(const Store *op, const Stmt &s) override {
    auto insn = OpRecog().Run(s);
    Stmt stmt = AttrStmt::make(make_zero(Int(32)), "pragma_emit_insn", Expr(insn), s);
    return stmt;
  }

  // inject pragma for outermost Vectorized loop
  Stmt Mutate_(const For *op, const Stmt &s) override {
    if (op->for_type == ForType::Vectorized) {
      auto insn = OpRecog().Run(op->body);
      Stmt stmt = AttrStmt::make(make_zero(Int(32)), "pragma_emit_insn", Expr(insn), s);
      return stmt;
    }
    return IRMutator::Mutate_(op, s);
  }
};

// res[cc1] = int8(uint1(res_local_UB[cc1]))
// if res.type = res_local_UB.type, we can remove cast
class LowerCast : public IRMutator {
  Stmt Mutate_(const Store *op, const Stmt &s) override {
    loads_ = 0;
    Stmt stmt = IRMutator::Mutate_(op, s);
    const auto n = stmt.as<Store>();
    CHECK(n);
    const auto c = n->value.as<Cast>();
    if (loads_ == 1 && c && op->value.type() == src_type_) {
      const auto ns = stmt.as<Store>();
      CHECK(ns);
      return Store::make(ns->buffer_var, load_, ns->index, ns->predicate);
    }
    return stmt;
  }

  Expr Mutate_(const Load *op, const Expr &e) override {
    loads_++;
    src_type_ = op->type;
    load_ = e;
    return IRMutator::Mutate_(op, e);
  }

 private:
  int loads_{0};
  Type src_type_{Type()};
  Expr load_{Expr()};
};

Stmt AutoPragma(Stmt stmt) {
  stmt = VectorizeFor().Mutate(stmt);
  stmt = LowerCast().Mutate(stmt);
  stmt = InjectPragma().Mutate(stmt);
  stmt = RecoverFor().Mutate(stmt);
  return stmt;
}
}  // namespace ir
}  // namespace akg
