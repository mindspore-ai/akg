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
#include "emit_insn/insn_info.h"
#include "analyze_align.h"
#include "emit_insn/ir_transform.h"

namespace akg {
namespace ir {

class ReduceRecover : public IRMutator {
 public:
  ReduceRecover() = default;
  ~ReduceRecover() override = default;

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == "pragma_emit_insn" && op->value->IsInstance<StringImm>() &&
        op->value.as<StringImm>()->value.find("reduce_") != std::string::npos) {
      old_pragma_ = op->value.as<StringImm>()->value;
      if (old_pragma_ == "reduce_add") {
        new_pragma_ = "vec_binary_add";
      } else if (old_pragma_ == "reduce_max") {
        new_pragma_ = "vec_binary_max";
      } else if (old_pragma_ == "reduce_min") {
        new_pragma_ = "vec_binary_min";
      } else if (old_pragma_ == "reduce_fargmax") {
        new_pragma_ = "vec_binary_fargmax";
      } else if (old_pragma_ == "reduce_fargmin") {
        new_pragma_ = "vec_binary_fargmin";
      }
      in_reduce_ = true;
      auto body = this->Mutate(op->body);
      in_reduce_ = false;
      return AttrStmt::make(make_zero(Int(32)), "pragma_emit_insn", Expr(new_pragma_), body);
    } else if (op->attr_key == "pragma_emit_insn" && op->value->IsInstance<StringImm>() &&
               op->value.as<StringImm>()->value == "dma_copy_transpose") {
      return AttrStmt::make(make_zero(Int(32)), "pragma_emit_insn", Expr("vtranspose"), op->body);
    } else if (op->attr_key == "align_info") {
      return this->Mutate(op->body);
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Store *op, const Stmt &s) final {
    if (in_reduce_) {
      if (old_pragma_ == "reduce_fargmax") {
        auto load_load = op->value.as<Call>()->args[0];
        auto src_load = Load::make(op->value.type(), op->buffer_var, op->index, op->predicate);
        auto new_value = Call::make(load_load.type(), "fargmax", {src_load, load_load}, Call::CallType::PureIntrinsic);
        auto new_store = Store::make(op->buffer_var, new_value, op->index, op->predicate);
        return new_store;
      } else if (old_pragma_ == "reduce_fargmin") {
        auto load_load = op->value.as<Call>()->args[0];
        auto src_load = Load::make(op->value.type(), op->buffer_var, op->index, op->predicate);
        auto new_value = Call::make(load_load.type(), "fargmin", {src_load, load_load}, Call::CallType::PureIntrinsic);
        auto new_store = Store::make(op->buffer_var, new_value, op->index, op->predicate);
        return new_store;
      } else if (old_pragma_ == "reduce_add") {
        auto src_load = Load::make(op->value.type(), op->buffer_var, op->index, op->predicate);
        auto new_value = Add::make(src_load, op->value.as<Call>()->args[0]);
        auto new_store = Store::make(op->buffer_var, new_value, op->index, op->predicate);
        return new_store;
      } else if (old_pragma_ == "reduce_max") {
        auto src_load = Load::make(op->value.type(), op->buffer_var, op->index, op->predicate);
        auto new_value = Max::make(src_load, op->value.as<Call>()->args[0]);
        auto new_store = Store::make(op->buffer_var, new_value, op->index, op->predicate);
        return new_store;
      } else if (old_pragma_ == "reduce_min") {
        auto src_load = Load::make(op->value.type(), op->buffer_var, op->index, op->predicate);
        auto new_value = Min::make(src_load, op->value.as<Call>()->args[0]);
        auto new_store = Store::make(op->buffer_var, new_value, op->index, op->predicate);
        return new_store;
      } else {
        return s;
      }
    } else {
      return IRMutator::Mutate_(op, s);
    }
  }

 private:
  std::string old_pragma_;
  std::string new_pragma_;
  bool in_reduce_;
};

std::string GetOpCode(const std::string &op_type) {
  std::string op_code{};
  if (op_type == "Add") {
    op_code = "vadds";
  } else if (op_type == "Mul") {
    op_code = "vmuls";
  } else if (op_type == "vaxpy") {
    op_code = "vaxpy";
  } else if (op_type == "DMACopy") {
    op_code = "vector_dup";
  }
  return op_code;
}

class FinetunePragma : public IRMutator {
 public:
  FinetunePragma() = default;
  ~FinetunePragma() override = default;

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if ((op->attr_key == "pragma_emit_insn" && op->value->IsInstance<StringImm>() &&
         !exclude_align_analyze_list.count(op->value.as<StringImm>()->value))) {
      IRInfo info;
      ParserVisitor(info, true).Run(s);
      std::string op_code = GetOpCode(info.arith_info.op_type);
      if (!info.arith_info.dst_info.IsUB() || op_code.empty() ||
          (!info.arith_info.src_info.empty() && !info.arith_info.src_info[0].IsUB())) {
        return s;
      }
      if (info.arith_info.insn_type == "simd" && info.arith_info.scalar_imm_num == 1 &&
          (op_code == "vmuls" || op_code == "vadds") && !info.arith_info.dst_info.p_store->value.type().is_float()) {
        return AttrStmt::make(make_zero(Int(32)), "pragma_emit_insn", Expr("scalar_calc"), op->body);
      }
      if (info.arith_info.insn_type == "vector_scalar" || info.arith_info.insn_type == "vector_dump") {
        return GenStore(info, op_code, 0);
      } else if (info.arith_info.insn_type == "simd" && info.arith_info.scalar_imm_num > 0) {
        CHECK_EQ(info.arith_info.scalar_imm_num, 1);
        return GenStore(info, op_code, 1);
      } else if (info.arith_info.insn_type == "simd" && info.arith_info.scalar_imm_num == 0 &&
                 info.arith_info.op_type == "DMACopy" && info.arith_info.dst_info.IsUB() &&
                 info.arith_info.src_info.size() == 1 && info.arith_info.src_info[0].IsUB() &&
                 info.arith_info.dst_info.p_store->value.type().is_float()) {
        /// change copy_ub_to_ub (fp16 or fp32) to adds (scalar = 0)
        op_code = "vadds";
        info.arith_info.scalar_imm_num = 1;
        info.arith_info.scalar_imm = FloatImm::make(info.arith_info.dst_info.p_store->value.type(), 0);
        return GenStore(info, op_code, 1);
      } else if (info.arith_info.op_type == "DMACopy" &&
                 (info.arith_info.insn_type == "scalar" || info.arith_info.insn_type == "discrete") &&
                 info.arith_info.dst_info.IsUB() &&
                 (info.arith_info.src_info.size() == 1 && info.arith_info.src_info[0].IsUB())) {
        return AttrStmt::make(make_zero(Int(32)), "pragma_emit_insn", Expr("scalar_dma"), op->body);
      } else if (info.arith_info.op_type == "DMACopy" &&
                 (info.arith_info.insn_type == "scalar" || info.arith_info.insn_type == "discrete") &&
                 info.arith_info.dst_info.IsUB() && info.arith_info.scalar_imm_num == 1) {
        return GenStore(info, op_code, 1);
      } else if (op->value.as<StringImm>()->value == "vec_single_muls" ||
                 op->value.as<StringImm>()->value == "vec_single_adds") {
        if (op->value.as<StringImm>()->value == "vec_single_muls") {
          op_code = "vmuls";
        } else if (op->value.as<StringImm>()->value == "vec_single_adds") {
          op_code = "vadds";
        }
        return GenStore(info, op_code, 1);
      }
      return s;
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt GenStore(IRInfo &info, const std::string &intrin_name, const int scalar_type = 0) {
    CHECK(intrin_name == "vector_dup" || intrin_name == "vadds" || intrin_name == "vmuls" || intrin_name == "vaxpy");

    /// scalar value
    Expr scalar_value =
      (scalar_type == 0) ? GetRef<Expr>(info.arith_info.scalar_load.p_load) : info.arith_info.scalar_imm;
    Array<Expr> call_args{};
    if (intrin_name == "vector_dup") {
      call_args = {scalar_value};
    } else {
      Expr tensor_value = GetRef<Expr>(info.arith_info.src_info[0].p_load);
      call_args = {tensor_value, scalar_value};
    }
    /// set store
    auto old_ptr = info.arith_info.dst_info.p_store;
    Expr new_value = Call::make(old_ptr->value.type(), intrin_name, call_args, Call::PureIntrinsic);
    Stmt ret = Store::make(old_ptr->buffer_var, new_value, old_ptr->index, old_ptr->predicate);
    if (scalar_type == 0) {
      auto scalar_vars = info.arith_info.scalar_load.vars;
      ///  set inner for loop
      for (int i = static_cast<int>(info.for_info.vars.size()) - 1; i >= 0; --i) {
        if (!IsInArray(scalar_vars, info.for_info.vars[i])) {
          ret = For::make(info.for_info.vars[i], 0, info.for_info.exts[i], ForType::Serial, DeviceAPI::None, ret);
        }
      }
      /// set attribute
      ret = AttrStmt::make(make_zero(Int(32)), "pragma_emit_insn", Expr(intrin_name), ret);
      ///  set outer for loop
      for (int i = static_cast<int>(info.for_info.vars.size()) - 1; i >= 0; --i) {
        if (IsInArray(scalar_vars, info.for_info.vars[i])) {
          ret = For::make(info.for_info.vars[i], 0, info.for_info.exts[i], ForType::Serial, DeviceAPI::None, ret);
        }
      }
      return ret;
    } else {
      for (int i = static_cast<int>(info.for_info.vars.size()) - 1; i >= 0; --i) {
        ret = For::make(info.for_info.vars[i], 0, info.for_info.exts[i], ForType::Serial, DeviceAPI::None, ret);
      }
      ret = AttrStmt::make(make_zero(Int(32)), "pragma_emit_insn", Expr(intrin_name), ret);
      return ret;
    }
  }
};

Stmt RecoverStore(Stmt stmt) {
  stmt = IfReorder().Mutate(stmt);
  stmt = FinetunePragma().Mutate(stmt);
  stmt = ReduceRecover().Mutate(stmt);
  return stmt;
}
}  // namespace ir
}  // namespace akg
