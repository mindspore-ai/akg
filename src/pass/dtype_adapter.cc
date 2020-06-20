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

#define OPERATOR_MAX_PARAMETERS_TVM_ACCESS_PTR 5
#define OPERATOR_MAX_PARAMETERS_VECTOR_DUP 7
#define OPERATOR_MAX_PARAMETERS_VTRANSPOSE_BEFORE_CHANGEDTYPE 2
#define ONE_REGISTER_MAX_PARAMETERS_VTRANSPOSE 5
#define OPERATOR_MAX_PARAMETERS_VAND_VOR 10
#define OPERATOR_MAX_PARAMETERS_TYPE_ANNOTATION 0
#define OPERATOR_MAX_PARAMETERS_VCMPV 10

namespace akg {
namespace ir {
/*
 * vector_dup(tvm_access_ptr(type_annotation(), input_local_UB, 0, 128, 3), (int16)1, 1, 1, 1, 8, 8)
 *                           int16_t
 * -->
 * vector_dup(tvm_access_ptr(type_annotation(), input_local_UB, 0, 128, 3), (int16)1, 1, 1, 1, 8, 8)
 *                           uint16_t
 *
 */

class VectorDupAdapter : public IRMutator {
 public:
  VectorDupAdapter() {}
  ~VectorDupAdapter() override = default;

  Expr Mutate_(const Call *op, const Expr &e) final {
    if (op->name == "vector_dup") {
      CHECK_EQ(op->args.size(), OPERATOR_MAX_PARAMETERS_VECTOR_DUP);
      vector_dup_ = true;
      Expr dst = this->Mutate(op->args[0]);
      vector_dup_ = false;

      return Call::make(op->type, op->name,
                        {dst, op->args[1], op->args[2], op->args[3], op->args[4], op->args[5], op->args[6]},
                        op->call_type, op->func, op->value_index);
    }

    if (vector_dup_ && op->name == "tvm_access_ptr") {
      CHECK_EQ(op->args.size(), OPERATOR_MAX_PARAMETERS_TVM_ACCESS_PTR);
      type_annotation_ = true;
      Expr type_annotation = this->Mutate(op->args[0]);
      type_annotation_ = false;

      return Call::make(op->type, op->name, {type_annotation, op->args[1], op->args[2], op->args[3], op->args[4]},
                        op->call_type, op->func, op->value_index);
    }

    if (vector_dup_ && type_annotation_ && op->name == "type_annotation") {
      CHECK_EQ(op->args.size(), OPERATOR_MAX_PARAMETERS_TYPE_ANNOTATION);
      if (op->type.bits() == 16 && op->type.lanes() == 1 && op->type.is_int()) {
        // int16_t -> uint16_t
        return Call::make(UInt(16, 1), op->name, {}, op->call_type, op->func, op->value_index);
      }

      return e;
    }

    return IRMutator::Mutate_(op, e);
  }

 private:
  bool vector_dup_{false};
  bool type_annotation_{false};
};

class VTransposeAdapter : public IRMutator {
 public:
  VTransposeAdapter() {}
  ~VTransposeAdapter() override = default;

  Stmt Mutate_(const Evaluate *op, const Stmt &s) final {
    const Call *call = op->value.as<Call>();
    if (call && call->name == "vtranspose") {
      CHECK_EQ(call->args.size(), OPERATOR_MAX_PARAMETERS_VTRANSPOSE_BEFORE_CHANGEDTYPE);
      Expr exprBody = Call::make(call->type, call->name, {ChangeDtype(call->args[0]), ChangeDtype(call->args[1])},
                                 call->call_type, call->func, call->value_index);

      return Evaluate::make(exprBody);
    }
    return s;
  }

  Expr ChangeDtype(const Expr &buffer) {
    auto call = buffer.as<Call>();
    CHECK(call);
    auto annt_call = call->args[0].as<Call>();
    CHECK(annt_call);
    Expr new_annotation =
      Call::make(UInt(16, 1), annt_call->name, {}, annt_call->call_type, annt_call->func, annt_call->value_index);

    CHECK(call->args.size() >= ONE_REGISTER_MAX_PARAMETERS_VTRANSPOSE);
    Expr arg =
      Call::make(call->type, call->name, {new_annotation, call->args[1], call->args[2], call->args[3], call->args[4]},
                 call->call_type, call->func, call->value_index);

    return arg;
  }
};

/*
 * For binary boolean operations storage type is int8, while the bitwise intrinsics require 16 bit type
 * We change the type and addresses in access pointers.
 *
 * vand or vor(tvm_access_ptr(type_annotation(), C_local_UB, 256, 154, 2),
 *      tvm_access_ptr(type_annotation(), A_local_UB, 256, 154, 1),
 *      tvm_access_ptr(type_annotation(), B_local_UB, 256, 154, 1), 1, 1, 1, 1, 0, 0, 0)
 *                     int8_t
 * -->
 * vand or vor(tvm_access_ptr(type_annotation(), C_local_UB, 128, 77, 2),
 *      tvm_access_ptr(type_annotation(), A_local_UB, 128, 77, 1),
 *      tvm_access_ptr(type_annotation(), B_local_UB, 128, 77, 1), 1, 1, 1, 1, 0, 0, 0)
 *                     uint16_t
 *
 */
class LogicalBinaryOpAdapter : public IRMutator {
 public:
  LogicalBinaryOpAdapter() {}
  ~LogicalBinaryOpAdapter() override = default;

  Expr Mutate_(const Call *op, const Expr &e) final {
    if (op->name == "vand" || op->name == "vor") {
      CHECK_EQ(op->args.size(), OPERATOR_MAX_PARAMETERS_VAND_VOR);
      logical_op_ = true;
      Expr dst = this->Mutate(op->args[0]);
      Expr src1 = this->Mutate(op->args[1]);
      Expr src2 = this->Mutate(op->args[2]);
      logical_op_ = false;
      return Call::make(
        op->type, op->name,
        {dst, src1, src2, op->args[3], op->args[4], op->args[5], op->args[6], op->args[7], op->args[8], op->args[9]},
        op->call_type, op->func, op->value_index);
    }

    if (logical_op_ && op->name == "tvm_access_ptr") {
      CHECK_EQ(op->args.size(), OPERATOR_MAX_PARAMETERS_TVM_ACCESS_PTR);
      if (op->args[0].type().bits() == 8 && op->args[0].type().lanes() == 1) {
        // int8_t -> uint16_t
        type_annotation_ = true;
        Expr type_annotation = this->Mutate(op->args[0]);
        type_annotation_ = false;
        return Call::make(
          op->type, op->name,
          {type_annotation, op->args[1], truncdiv(op->args[2], 2), truncdiv(op->args[3], 2), op->args[4]},
          op->call_type, op->func, op->value_index);
      }

      return e;
    }

    if (logical_op_ && type_annotation_ && op->name == "type_annotation") {
      CHECK_EQ(op->args.size(), OPERATOR_MAX_PARAMETERS_TYPE_ANNOTATION);
      // vand and vor only accept 16 bit types
      return Call::make(UInt(16, 1), op->name, {}, op->call_type, op->func, op->value_index);
    }

    return IRMutator::Mutate_(op, e);
  }

 private:
  bool logical_op_{false};
  bool type_annotation_{false};
};

class VcmpvOpAdapter : public IRMutator {
 public:
  VcmpvOpAdapter() {}
  ~VcmpvOpAdapter() override = default;

  Expr Mutate_(const Call *op, const Expr &e) final {
    if (op->name == "vcmpv_ge" || op->name == "vcmpv_gt" || op->name == "vcmpv_lt" || op->name == "vcmpv_le" ||
        op->name == "vcmpv_eq" || op->name == "vcmpv_ne") {
      CHECK_EQ(op->args.size(), OPERATOR_MAX_PARAMETERS_VCMPV);
      vcmpv_op_ = true;
      Expr dst = this->Mutate(op->args[0]);
      vcmpv_op_ = false;
      return Call::make(op->type, op->name,
                        {dst, op->args[1], op->args[2], op->args[3], op->args[4], op->args[5], op->args[6], op->args[7],
                         op->args[8], op->args[9]},
                        op->call_type, op->func, op->value_index);
    }

    if (vcmpv_op_ && op->name == "tvm_access_ptr") {
      CHECK_EQ(op->args.size(), OPERATOR_MAX_PARAMETERS_TVM_ACCESS_PTR);
      type_annotation_ = true;
      Expr type_annotation = this->Mutate(op->args[0]);
      type_annotation_ = false;
      return Call::make(op->type, op->name, {type_annotation, op->args[1], op->args[2], op->args[3], op->args[4]},
                        op->call_type, op->func, op->value_index);
    }

    if (vcmpv_op_ && type_annotation_ && op->name == "type_annotation") {
      CHECK_EQ(op->args.size(), OPERATOR_MAX_PARAMETERS_TYPE_ANNOTATION);
      return Call::make(UInt(8, 1), op->name, {}, op->call_type, op->func, op->value_index);
    }

    return IRMutator::Mutate_(op, e);
  }

 private:
  bool vcmpv_op_{false};
  bool type_annotation_{false};
};

Stmt DTypeAdapter(Stmt stmt) {
  stmt = VectorDupAdapter().Mutate(stmt);
  stmt = VTransposeAdapter().Mutate(stmt);
  stmt = LogicalBinaryOpAdapter().Mutate(stmt);
  stmt = VcmpvOpAdapter().Mutate(stmt);
  return stmt;
}
}  // namespace ir
}  // namespace akg
