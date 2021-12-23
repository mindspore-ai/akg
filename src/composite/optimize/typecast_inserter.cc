/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "composite/optimize/pass.h"
#include "src/pass/ir_util.h"
/**
 * input0 = xxx : Int32
 * input1 = xxx : Int32
 *
 * output(1024) = Equal(input0(1024), input1(1024))
 * output1(1024) = Cast(ouput(1024), 'int32')
 *
 * ->
 *
 * cast_0 = Cast(input0(1024), 'fp32')
 * cast_1 = Cast(input1(1024), 'fp32')
 * output(1024) = Equal(cast0, cast1)
 * tmp_output(1024) = Cast(output(1024), 'fp16')
 * output1(1024) = Cast(tmp_output(1024), 'int32')
 */
namespace akg {

Stmt CastStmtMaker(const Expr &input_tensor, const Expr &output_tensor, std::string cast_type) {
  auto cast_call = Call::make(output_tensor.type(), "Cast", {input_tensor}, Call::CallType::Intrinsic);
  auto output_call = output_tensor.as<Call>();
  CHECK(output_call);
  auto provide = Provide::make(output_call->func, 0, cast_call, output_call->args);
  Map<std::string, NodeRef> attr_node;
  attr_node.Set("dst_type", StringImm::make(cast_type));
  auto attr_pack = AttrStmt::make(attr_node, "attrs", Expr(1), provide);
  return attr_pack;
}

class EqualCastInserterMutator : public IRMutator {
 public:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) override {
    if (EqualCast(op)) {
      auto provide = op->body.as<Provide>();
      auto call = provide->value.as<Call>();
      auto shape = provide->args;
      Expr input_a = call->args[0];
      Expr input_b = call->args[1];
      
      Tensor cast_output_a_node = placeholder(shape, Float(32), "cmp_input1");
      Tensor cast_output_b_node = placeholder(shape, Float(32), "cmp_input2");
      Expr cast_output_a = Call::make(cast_output_a_node->dtype, cast_output_a_node->op->func_name(), shape, Call::CallType::Halide, cast_output_a_node->op);
      Expr cast_output_b = Call::make(cast_output_b_node->dtype, cast_output_b_node->op->func_name(), shape, Call::CallType::Halide, cast_output_b_node->op);
      Stmt cast_a = CastStmtMaker(input_a, cast_output_a, "float32");
      Stmt cast_b = CastStmtMaker(input_b, cast_output_b, "float32");
      auto cmp_call = Call::make(call->type, call->name, {cast_output_a, cast_output_b}, Call::CallType::Intrinsic);
      auto cmp_provide = Provide::make(provide->func, 0, cmp_call, shape);
      Stmt cmp_stmt = AttrStmt::make(op->node, op->attr_key, op->value, cmp_provide);
      return air::ir::MergeSeq({cast_a, cast_b, cmp_stmt});
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  bool EqualCast(const AttrStmt *op) {
    auto provide = op->body.as<Provide>();
    if (op->attr_key == "attrs" && provide) {
      auto call = provide->value.as<Call>();
      if (call && typecast_ops_.find(call->name) != typecast_ops_.end() && call->args.size()) {
        auto arg0 = call->args[0].as<Call>();
        auto arg1 = call->args[1].as<Call>();
        if (arg0 && arg1 && arg0->type == Int(32) && arg1->type == Int(32)) {
          return true;
        }
      }
    }
    return false;
  }

  const std::unordered_set<std::string> typecast_ops_ = {"Equal", "LessEqual", "Less", "Greater", "GreaterEqual"};
};

// change bool -> int32/fp32 to bool -> fp16 -> int32/fp32
class BoolCastInserterMutator : public IRMutator {
 public:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) override {
    auto bool_cast_type = BoolCastType(op);
    if (!bool_cast_type.empty()) {
      auto dst_type = ((bool_cast_type == "float32") ? Float(32) : Int(32));
      auto provide = op->body.as<Provide>();
      auto cast_call = provide->value.as<Call>();
      auto shape = provide->args;
      Expr input_tensor = cast_call->args[0];
      Tensor cast_tmp_tensor_node = placeholder(shape, Float(16), "fp16_input1");
      Expr cast_tmp_tensor = Call::make(cast_tmp_tensor_node->dtype, cast_tmp_tensor_node->op->func_name(), shape, Call::CallType::Halide, cast_tmp_tensor_node->op);
      Expr output_tensor = Call::make(dst_type, provide->func->func_name(), shape, Call::CallType::Halide, provide->func);
      Stmt cast_float = CastStmtMaker(input_tensor, cast_tmp_tensor, "float16");
      Stmt cast_int = CastStmtMaker(cast_tmp_tensor, output_tensor, bool_cast_type);
      return Block::make(cast_float, cast_int);
    }
    return s;
  }

 private:
  std::string BoolCastType(const AttrStmt *op) {
    auto provide = op->body.as<Provide>();
    if (op->attr_key == "attrs" && provide) {
      auto attrs = Downcast<Map<std::string, NodeRef>>(op->node);
      if (attrs.find("dst_type") != attrs.end()) {
        auto dst_type = attrs["dst_type"].as<StringImm>();
        if (dst_type && (dst_type->value == "int32" || dst_type->value == "float32")) {
          auto call = provide->value.as<Call>();
          if (call && call->name == "Cast" && call->args[0].as<Call>() && call->args[0].as<Call>()->type == Bool(1)) {
            return dst_type->value;
          }
        }
      }
    }
    return "";
  }
};

/**
 * output(1024) = 1000 - input(1024)
 *
 * ->
 * tmp(1024) = cast_fp32(input(1024))
 * output_tmp(1024) = 1000 - tmp(1024)
 * output(1024) = cast_int32(output_tmp(1024))
 *
 *
 * to avoid int32 vmuls in ascend backend
 **/

class ScalarSubCastInserterMutator : public IRMutator {
 public:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) override {
    if (ScalarSub(op)) {
      auto provide = op->body.as<Provide>();
      auto sub_call = provide->value.as<Call>();
      auto input_1 = sub_call->args[1].as<Call>();
      auto shape = provide->args;
      Expr input_1_tensor = sub_call->args[1];
      Tensor cast_tmp_op = placeholder(shape, Float(32), "fp32_" + input_1->func->func_name());
      Expr cast_tmp_tensor = Call::make(cast_tmp_op->dtype, cast_tmp_op->op->func_name(), shape, Call::CallType::Halide, cast_tmp_op->op);
      Stmt cast_float = CastStmtMaker(input_1_tensor, cast_tmp_tensor, "float32");
      auto new_sub_call =
        Call::make(Float(32), sub_call->name, {sub_call->args[0], cast_tmp_tensor}, Call::CallType::Intrinsic);
      Tensor new_sub_op = placeholder(shape, Float(32), "sub_cast_" + provide->func->func_name());
      Expr new_sub = Call::make(new_sub_op->dtype, new_sub_op->op->func_name(), shape, Call::CallType::Halide, new_sub_op->op);
      auto sub_provide = Provide::make(new_sub_op->op, 0, new_sub_call, shape);
      Stmt sub_stmt = AttrStmt::make(op->node, op->attr_key, op->value, sub_provide);
      Expr output_tensor = Call::make(Int(32), provide->func->func_name(), shape, Call::CallType::Halide, provide->func);
      Stmt cast_int = CastStmtMaker(new_sub, output_tensor, "int32");
      return air::ir::MergeSeq({cast_float, sub_stmt, cast_int});
    }
    return s;
  }

 private:
  bool ScalarSub(const AttrStmt *op) {
    auto provide = op->body.as<Provide>();
    if (op->attr_key == "attrs" && provide) {
      auto call = provide->value.as<Call>();
      if (call && call->name == "Sub" && call->args.size() > 1) {
        auto arg1 = call->args[1].as<Call>();
        if ((call->args[0].as<IntImm>() || call->args[0].as<FloatImm>()) && arg1->type == Int(32)) {
          return true;
        }
      }
    }
    return false;
  }
};

Stmt TypeCastInserter(const Stmt &s, BuildInfo*) {
  Stmt stmt = EqualCastInserterMutator().Mutate(s);
  stmt = BoolCastInserterMutator().Mutate(stmt);
  stmt = ScalarSubCastInserterMutator().Mutate(stmt);
  return stmt;
}
}  // namespace akg
