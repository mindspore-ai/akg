/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include <tvm/ir_pass.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_visitor.h>
#include <tvm.h>

namespace akg {
namespace ir {
class DivScalarFinder : public IRVisitor {
 public:
  DivScalarFinder() {}
  ~DivScalarFinder() override = default;

  void Visit_(const AttrStmt *op) final {
    auto func = Downcast<FunctionRef>(op->node);
    attr_.push_back(func);
    IRVisitor::Visit(op->body);
    attr_.pop_back();
  }

  void Visit_(const Realize *op) final {
    realize_.push_back(op->func);
    IRVisitor::Visit(op->body);
    realize_.pop_back();
  }

  void Visit_(const ProducerConsumer *op) final {
    produce_.push_back(op->func);
    IRVisitor::Visit(op->body);
    produce_.pop_back();
  }

  void Visit_(const Provide *op) final {
    auto func = op->func;
    if (attr_.back() != func || realize_.back() != func || produce_.back() != func) {
      return;
    }

    auto div = op->value.as<Div>();
    if (!div) {
      return;
    }

    auto call_a = div->a.as<Call>();
    auto call_b = div->b.as<Call>();
    if (call_a && !IsShapeOne(call_a->args) && call_b && IsShapeOne(call_b->args) && div->b.type().is_float()) {
      target_provide_[func] = div;
    }
  }

  std::unordered_map<FunctionRef, const Div *, NodeHash, NodeEqual> target_provide_;

 private:
  bool IsShapeOne(const Array<Expr> &args) {
    bool is_shape_one = true;
    for (size_t i = 0; i < args.size(); ++i) {
      auto var = args[i].as<IntImm>();
      if (!var || var->value != 0) {
        is_shape_one = false;
        break;
      }
    }
    return is_shape_one;
  }

  std::vector<FunctionRef> attr_;
  std::vector<FunctionRef> realize_;
  std::vector<FunctionRef> produce_;
};

class DivScalarRewriter : public IRMutator {
 public:
  explicit DivScalarRewriter(const std::unordered_map<FunctionRef, const Div *, NodeHash, NodeEqual> &target_provide)
      : target_provide_(target_provide) {}
  ~DivScalarRewriter() override = default;

  Stmt DoRewrite(const Stmt &s) {
    Array<Expr> shape;
    Array<Expr> args;
    Region region;
    shape.push_back(make_const(Int(32), 1));
    args.push_back(make_const(Int(32), 0));
    region.push_back(Range::make_by_min_extent(Expr(0), Expr(1)));

    for (const auto &it : target_provide_) {
      auto b = it.second->b.as<Call>();
      CHECK(b);
      auto invert = PlaceholderOpNode::make(b->name + "_invert", shape, b->type);
      div_invert_[it.second->b] = invert;
    }

    return Mutate(s);
  }

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    auto func = Downcast<FunctionRef>(op->node);
    if (target_provide_.count(func) == 0) {
      return IRMutator::Mutate_(op, s);
    }

    auto b = target_provide_[func]->b.as<Call>();
    CHECK(b);

    Array<Expr> shape;
    Array<Expr> args;
    Region region;
    shape.push_back(make_const(Int(32), 1));
    args.push_back(make_const(Int(32), 0));
    region.push_back(Range::make_by_min_extent(Expr(0), Expr(1)));

    auto div = target_provide_[func];
    auto invert = div_invert_[div->b];
    auto invert_provide = Provide::make(invert, 0, Div::make(make_const(b->type, 1), div->b), args);
    auto invert_produce = ProducerConsumer::make(invert, true, invert_provide);
    auto body = Mutate(op->body);
    auto orig_attr = AttrStmt::make(op->node, op->attr_key, op->value, body);
    auto invert_block = Block::make(invert_produce, orig_attr);
    auto invert_realize = Realize::make(invert, 0, b->type, region, const_true(1), invert_block);
    auto invert_attr = AttrStmt::make(invert, op->attr_key, op->value, invert_realize);

    return invert_attr;
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    auto func = op->func;
    auto div = op->value.as<Div>();
    if (target_provide_.count(func) != 0 && div) {
      auto b = div->b.as<Call>();
      if (b && div_invert_.count(div->b) != 0) {
        Array<Expr> args;
        args.push_back(make_const(Int(32), 0));
        auto invert = div_invert_[div->b];
        auto new_b = Call::make(b->type, invert->name, args, b->call_type, invert, 0);
        return Provide::make(func, op->value_index, Mul::make(div->a, new_b), op->args);
      }
    }

    return IRMutator::Mutate_(op, s);
  }

 private:
  std::unordered_map<FunctionRef, const Div *, NodeHash, NodeEqual> target_provide_;
  std::unordered_map<Expr, Operation, NodeHash, NodeEqual> div_invert_;
};

Stmt ScalarComputeRewrite(const Stmt &stmt) {
  DivScalarFinder finder;
  finder.Visit(stmt);
  DivScalarRewriter rewriter(finder.target_provide_);
  return rewriter.DoRewrite(stmt);
}
}  // namespace ir
}  // namespace akg
