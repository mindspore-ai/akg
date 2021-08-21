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

#include "composite/optimize/typecast_inserter.h"
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

Expr TensorToCall(const Tensor &tensor) {
  return Call::make(tensor->dtype, tensor->op->name, tensor->shape, Call::CallType::Halide, tensor->op);
}

Stmt CastStmtMaker(const Tensor &input_tensor, const Tensor &output_tensor, std::string cast_type) {
  auto input_call = TensorToCall(input_tensor);
  auto cast_call = Call::make(output_tensor->dtype, "Cast", {input_call}, Call::CallType::Intrinsic);
  auto provide = Provide::make(output_tensor->op, 0, cast_call, output_tensor->shape);
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
      Tensor input_a = Downcast<Operation>(call->args[0].as<Call>()->func).output(0);
      Tensor input_b = Downcast<Operation>(call->args[1].as<Call>()->func).output(0);
      Tensor cast_output_a = placeholder(input_a->shape, Float(32), "cmp_input1");
      Tensor cast_output_b = placeholder(input_b->shape, Float(32), "cmp_input2");
      Stmt cast_a = CastStmtMaker(input_a, cast_output_a, "float32");
      Stmt cast_b = CastStmtMaker(input_b, cast_output_b, "float32");
      auto cmp_call = Call::make(call->type, call->name, {TensorToCall(cast_output_a), TensorToCall(cast_output_b)},
                                 Call::CallType::Intrinsic);
      auto cmp_provide = Provide::make(provide->func, 0, cmp_call, provide->args);
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

// change bool -> int32 to bool -> fp16 -> int32
class BoolCastInserterMutator : public IRMutator {
 public:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) override {
    if (CastToInt(op)) {
      auto provide = op->body.as<Provide>();
      auto cast_call = provide->value.as<Call>();
      auto input_call = cast_call->args[0].as<Call>();
      Tensor input_tensor = Downcast<Operation>(input_call->func).output(0);
      Tensor cast_tmp_tensor = placeholder(input_call->args, Float(16), "fp16_input1");
      Tensor output_tensor = Downcast<Operation>(provide->func).output(0);
      Stmt cast_float = CastStmtMaker(input_tensor, cast_tmp_tensor, "float16");
      Stmt cast_int = CastStmtMaker(cast_tmp_tensor, output_tensor, "int32");
      return Block::make(cast_float, cast_int);
    }
    return s;
  }

 private:
  bool CastToInt(const AttrStmt *op) {
    auto provide = op->body.as<Provide>();
    if (op->attr_key == "attrs" && provide) {
      auto attrs = Downcast<Map<std::string, NodeRef>>(op->node);
      if (attrs.find("dst_type") != attrs.end()) {
        auto dst_type = attrs["dst_type"].as<StringImm>();
        if (dst_type && dst_type->value == "int32") {
          auto call = provide->value.as<Call>();
          if (call && call->name == "Cast" && call->args[0].as<Call>() && call->args[0].as<Call>()->type == Bool(1)) {
            return true;
          }
        }
      }
    }
    return false;
  }
};

Stmt TypeCastInserter::Run(const Stmt &s) {
  Stmt stmt = EqualCastInserterMutator().Mutate(s);
  stmt = BoolCastInserterMutator().Mutate(stmt);
  return stmt;
}
}  // namespace akg
