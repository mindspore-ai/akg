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

namespace akg {
class EqualCastInserterMutator : public IRMutator {
 public:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) override {
    if (op->attr_key == "attrs" && op->body.as<Provide>()) {
      const auto *provide = op->body.as<Provide>();
      CHECK(provide);
      auto call = provide->value.as<Call>();
      CHECK(call);
      auto it = typecast_ops_.find(call->name);
      if (it != typecast_ops_.end() && call->type == Int(32)) {
        CHECK_EQ(call->args.size(), 2);
        if (call->args[0].as<Call>() && call->args[1].as<Call>() && call->args[0].as<Call>()->type == Int(32) &&
            call->args[1].as<Call>()->type == Int(32)) {
          auto input0 = call->args[0];
          auto input1 = call->args[1];
          auto in0_shape = call->args[0].as<Call>()->args;
          auto in1_shape = call->args[1].as<Call>()->args;
          Tensor t0 = placeholder(in0_shape, Float(32), "cmp_input1");
          Tensor t1 = placeholder(in1_shape, Float(32), "cmp_input2");
          Tensor t2 = placeholder(provide->args, Float(32), "cmp_output");
          Map<std::string, NodeRef> attrs0, attrs1, attrs2, attrs3;
          attrs0.Set("dst_type", StringImm::make("float32"));
          attrs1.Set("dst_type", StringImm::make("float32"));
          attrs3.Set("dst_type", StringImm::make("float32"));

          auto arg0 = Call::make(t0->dtype, t0->op->name, t0->shape, Call::CallType::Halide, t0->op);
          auto arg1 = Call::make(t1->dtype, t1->op->name, t1->shape, Call::CallType::Halide, t1->op);
          auto arg2 = Call::make(t2->dtype, t2->op->name, t2->shape, Call::CallType::Halide, t2->op);
          auto cast0 = Call::make(t0->dtype, "Cast", {input0}, Call::CallType::Intrinsic);
          auto cast1 = Call::make(t1->dtype, "Cast", {input1}, Call::CallType::Intrinsic);
          auto cmp_op = Call::make(Float(32), call->name, {arg0, arg1}, Call::CallType::Intrinsic);
          auto assign_cast0 = Provide::make(t0->op, 0, cast0, in0_shape);
          auto assign_cast1 = Provide::make(t1->op, 0, cast1, in1_shape);
          auto assign_cmp = Provide::make(t2->op, 0, cmp_op, provide->args);
          auto value_int32 = Call::make(Float(32), "Cast", {arg2}, Call::CallType::Intrinsic);
          auto new_provide = Provide::make(provide->func, provide->value_index, value_int32, provide->args);
          auto new_attr0 = AttrStmt::make(attrs0, "attrs", Expr(1), assign_cast0);
          auto new_attr1 = AttrStmt::make(attrs1, "attrs", Expr(1), assign_cast1);
          auto new_attr2 = AttrStmt::make(attrs2, "attrs", Expr(1), assign_cmp);
          auto new_attr3 = AttrStmt::make(attrs3, "attrs", Expr(1), new_provide);
          auto new_body = Block::make(Block::make(new_attr0, new_attr1), Block::make(new_attr2, new_attr3));
          auto new_attr = AttrStmt::make(op->node, op->attr_key, op->value, new_body);
          return new_attr;
        }
      }
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  std::unordered_map<std::string, unsigned> typecast_ops_ = {
    {"Equal", -1}, {"LessEqual", -1}, {"Less", -1}, {"Greater", -1}, {"GreaterEqual", -1},
  };
};

// change bool -> int32 to bool -> fp16 -> int32
class BoolCastInserterMutator : public IRMutator {
 public:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) override {
    if (op->attr_key == "attrs" && op->body.as<Provide>()) {
      auto attrs = Downcast<Map<std::string, NodeRef>>(op->node);
      if (attrs.find("dst_type") == attrs.end() || !attrs["dst_type"].as<StringImm>() ||
          attrs["dst_type"].as<StringImm>()->value != "int32") {
        return s;
      }
      const auto *provide = op->body.as<Provide>();
      auto call = provide->value.as<Call>();
      CHECK(call);
      if (call->name == "Cast" && call->args[0].as<Call>() && call->args[0].as<Call>()->type == Bool(1)) {
        auto input0 = call->args[0];
        auto in0_shape = call->args[0].as<Call>()->args;
        Tensor t0 = placeholder(in0_shape, Float(16), "fp16_input1");
        Map<std::string, NodeRef> attrs0, attrs1;
        attrs0.Set("dst_type", StringImm::make("float16"));
        attrs1.Set("dst_type", StringImm::make("int32"));
        auto arg0 = Call::make(t0->dtype, t0->op->name, t0->shape, Call::CallType::Halide, t0->op);
        auto cast0 = Call::make(Float(16), "Cast", {input0}, Call::CallType::Intrinsic);
        auto cast1 = Call::make(Int(32), "Cast", {arg0}, Call::CallType::Intrinsic);
        auto assign_cast0 = Provide::make(t0->op, 0, cast0, in0_shape);
        auto assign_cast1 = Provide::make(provide->func, 0, cast1, in0_shape);
        auto new_attr0 = AttrStmt::make(attrs0, "attrs", Expr(1), assign_cast0);
        auto new_attr1 = AttrStmt::make(attrs1, "attrs", Expr(1), assign_cast1);
        auto new_block = Block::make(new_attr0, new_attr1);
        return new_block;
      }
    }
    return s;
  }
};

Stmt TypeCastInserter::Run(const Stmt &s) {
  auto s1 = EqualCastInserterMutator().Mutate(s);
  auto s2 = BoolCastInserterMutator().Mutate(s1);
  return s2;
}
}  // namespace akg
