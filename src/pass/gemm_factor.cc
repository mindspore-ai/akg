/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include <dmlc/common.h>
#include <tvm/ir.h>
#include <tvm/expr.h>
#include <tvm/operation.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_mutator.h>
#include <tvm/expr_operator.h>
#include <tvm/ir_pass.h>
#include <tvm/buffer.h>
#include <tvm/target_info.h>
#include <tvm/build_module.h>
#include <tvm/runtime/device_api.h>

#include <unordered_map>

#include "common/common_util.h"
#include "pass/utils.h"
#include "ir_pass.h"

namespace akg {
namespace ir {

constexpr auto REDUCE_PROVIDE = "reduce_provide";
constexpr auto PACKA = "pack_a";
constexpr auto TENSOR_C = "tensor_c";
constexpr auto MAJOR_MATRIX = "major_matrix";
constexpr auto MATRIX_A = "matrix_a";
constexpr auto MATRIX_B = "matrix_b";
constexpr auto COL_MAJOR = "col_major";
constexpr auto ROW_MAJOR = "row_major";
constexpr auto LOCAL = "local";

enum GemmMNK { M = 0, N, K };

class ExprUsedVarsVisitor : public IRVisitor {
 public:
  explicit ExprUsedVarsVisitor() {}

  std::vector<const Variable *> Run(Expr e) {
    this->Visit(e);
    return vars_;
  }

 private:
  void Visit_(const Variable *op) { vars_.push_back(op); }
  std::vector<const Variable *> vars_;
};

class ModifyTheLocalOffset : public IRMutator {
 private:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == REDUCE_PROVIDE) {
      sync_compute_flag_ = true;
      Stmt body = IRMutator::Mutate(op->body);
      sync_compute_flag_ = false;
      return AttrStmt::make(op->node, op->attr_key, op->value, body);
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) {
    if (sync_compute_flag_) {
      Stmt stmt = ModifyTheOpIndexOfSync(op, GetFragmentIndex(op->args));
      return stmt;
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const For *op, const Stmt &s) {
    vec_for_vars_.push_back(op);
    Stmt stmt = IRMutator::Mutate_(op, s);
    vec_for_vars_.pop_back();
    return stmt;
  }

  Array<Expr> GetFragmentIndex(const Array<Expr> args) {
    Array<Expr> new_index;
    for (auto &arg : args) {
      auto used_vars = ExprUsedVarsVisitor().Run(arg);
      Expr e = make_const(Int(32), 0);
      int size = used_vars.size();

      if (size == 0) {
        e = arg;
      } else if (size == 1 && arg.as<Add>() != nullptr) {
        Expr add_a = arg.as<Add>()->a;
        Expr add_b = arg.as<Add>()->b;
        Expr last_var = Expr(GetObjPtr(used_vars[size - 1]));
        e = last_var.same_as(add_a) ? add_b : add_a;
      }

      for (int i = 0; i < size - 1; i++) {
        auto u = used_vars[i];
        Expr temp = Expr(GetObjPtr(u));
        for (int j = i + 1; j < size; j++) {
          temp = Mul::make(temp, FindExtentOfForVar(used_vars[j]));
        }
        e = Add::make(e, temp);
      }
      new_index.push_back(e);
    }
    return new_index;
  }

  Expr Mutate_(const Call *op, const Expr &e) {
    if (sync_value_mod_) {
      Array<Expr> real_index;
      real_index = GetFragmentIndex(op->args);
      return Call::make(op->type, op->name, real_index, op->call_type, op->func, op->value_index);
    }

    return IRMutator::Mutate_(op, e);
  }

  Stmt ModifyTheOpIndexOfSync(const Provide *op, Array<Expr> real_index) {
    auto value = op->value;
    sync_value_mod_ = true;
    value = this->Mutate(value);
    sync_value_mod_ = false;
    return Provide::make(op->func, op->value_index, value, real_index);
  }

  Expr FindExtentOfForVar(const Variable *var) {
    for (auto &v : vec_for_vars_) {
      if (v->loop_var.get() == var) {
        return v->extent;
      }
    }
    return Expr();
  }

 private:
  std::vector<const For *> vec_for_vars_;
  std::string major_matrix_;
  bool sync_value_mod_{false};
  bool sync_compute_flag_{false};
};

class GemmCompute : public IRMutator {
 private:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == TENSOR_C) {
      sync_compute_flag_ = true;
      Stmt body = IRMutator::Mutate(op->body);
      sync_compute_flag_ = false;
      Stmt stmt = Evaluate::make(Call::make(Handle(), air::ir::intrinsic::sgemm_kernel_avx,
                                            {input_2_, input_1_, compute_, mnk_size_[GemmMNK::M], mnk_size_[GemmMNK::N],
                                             mnk_size_[GemmMNK::K], ldc_, Cast::make(Float(32), 1)},
                                            Call::Intrinsic));
      return stmt;
    } else if (op->attr_key == REDUCE_PROVIDE) {
      AnalyzeComputeInfo(op->body.as<Provide>());
    } else if (op->attr_key.find(MAJOR_MATRIX) != std::string::npos) {
      std::string major_matrix = op->attr_key.find(MAJOR_MATRIX) != std::string::npos ? MATRIX_A : MATRIX_B;
      SetMNKSize(op->attr_key);
      vec_for_matrixs_.clear();
      return IRMutator::Mutate(op->body);
    }
    return IRMutator::Mutate_(op, s);
  }

  void SetMNKSize(const std::string &attr_key) {
    bool is_matrix_a = attr_key.find(MATRIX_A) != std::string::npos;
    bool is_matrix_b = attr_key.find(MATRIX_B) != std::string::npos;
    bool is_col_major = attr_key.find(COL_MAJOR) != std::string::npos;
    bool is_row_major = attr_key.find(ROW_MAJOR) != std::string::npos;

    int last_pos = static_cast<int>(vec_for_matrixs_.size()) - 1;
    CHECK(last_pos >= 1) << "The gemm operation contains at least two axes";

    if (is_matrix_a && is_row_major) {
      mnk_size_[GemmMNK::N] = vec_for_matrixs_[last_pos - 1];
      mnk_size_[GemmMNK::K] = vec_for_matrixs_[last_pos];
    } else if (is_matrix_a && is_col_major) {
      mnk_size_[GemmMNK::N] = vec_for_matrixs_[last_pos];
      mnk_size_[GemmMNK::K] = vec_for_matrixs_[last_pos - 1];
    } else if (is_matrix_b && is_col_major) {
      mnk_size_[GemmMNK::M] = vec_for_matrixs_[last_pos - 1];
      mnk_size_[GemmMNK::K] = vec_for_matrixs_[last_pos];
    } else if (is_matrix_b && is_row_major) {
      mnk_size_[GemmMNK::M] = vec_for_matrixs_[last_pos];
      mnk_size_[GemmMNK::K] = vec_for_matrixs_[last_pos - 1];
    }
  }

  void AnalyzeComputeInfo(const Provide *op) {
    Expr c_expr = op->value.as<Add>()->a;
    Expr a_expr = op->value.as<Add>()->b.as<Mul>()->a;
    Expr b_expr = op->value.as<Add>()->b.as<Mul>()->b;

    compute_ = Call::make(Handle(), air::ir::intrinsic::tvm_address_of, {c_expr}, Call::PureIntrinsic);
    input_1_ = Call::make(Handle(), air::ir::intrinsic::tvm_address_of, {a_expr}, Call::PureIntrinsic);
    input_2_ = Call::make(Handle(), air::ir::intrinsic::tvm_address_of, {b_expr}, Call::PureIntrinsic);
  }

  Stmt Mutate_(const Realize *op, const Stmt &s) final {
    for (auto bound : op->bounds) {
      auto extent = bound->extent.as<IntImm>();
      if (!extent) {
        vec_for_matrixs_.push_back(1);
        continue;
      }
      vec_for_matrixs_.push_back(extent->value);
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    if (sync_compute_flag_) {
      auto operator_c = op->func.as<OperationNode>();
      auto operator_c_shape = operator_c->output_shape(0);
      ldc_ = operator_c_shape[operator_c_shape.size() - 1].as<IntImm>()->value;
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (sync_compute_flag_) {
      return IRMutator::Mutate(op->body);
    }
    return IRMutator::Mutate_(op, s);
  }

  std::unordered_map<GemmMNK, int64_t> mnk_size_;
  std::vector<int64_t> vec_for_matrixs_;

  bool sync_compute_flag_{false};
  int64_t ldc_{0};
  Expr compute_;
  Expr input_1_;
  Expr input_2_;
};

class GemmCheck : public IRVisitor {
 public:
  bool IsGemm() { return is_gemm_; }

 private:
  void Visit_(const AttrStmt *op) final {
    if (op->attr_key == PACKA) {
      is_gemm_ = true;
    }
    IRVisitor::Visit_(op);
  }

 private:
  bool is_gemm_{false};
};

Stmt GemmFactor(const Stmt &stmt) {
  GemmCheck checker;
  checker.Visit(stmt);
  if (!checker.IsGemm()) {
    return stmt;
  }

  ModifyTheLocalOffset modify;
  Stmt s = modify.Mutate(stmt);

  GemmCompute gemm_compute;
  return gemm_compute.Mutate(s);
}

}  // namespace ir
}  // namespace akg
