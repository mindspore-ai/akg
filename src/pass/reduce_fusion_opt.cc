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
#include "pass/post_fusion_utils.h"

namespace akg {
namespace ir {
Stmt ReduceUbInit(const Map<std::string, NodeRef> &attrs, size_t index) {
  CHECK(attrs[ATTR_CONV_KERNEL_N].as<IntImm>());
  int kernel_n = GET_INTIMM_ATTR_DEFAULT(attrs, ATTR_CONV_KERNEL_N, 0);
  CHECK(kernel_n);
  const int block_size = 16;
  int c1_extent = kernel_n / block_size;
  int c0_extent = block_size;
  int w_extent = 4;
  Var c1("c1");
  Var c0("c0");
  Var w("c0_out");
  Array<Expr> args;
  args.push_back(Expr(0));
  args.push_back(c1);
  args.push_back(Expr(0));
  args.push_back(w);
  args.push_back(c0);
  Region bounds;
  bounds.push_back(Range::make_by_min_extent(0, 1));
  bounds.push_back(Range::make_by_min_extent(0, c1_extent));
  bounds.push_back(Range::make_by_min_extent(0, 1));
  bounds.push_back(Range::make_by_min_extent(0, w_extent));
  bounds.push_back(Range::make_by_min_extent(0, c0_extent));
  Array<Expr> shape;
  std::string reduce_name = WHOLE_REDUCE_UB + std::to_string(index);
  auto t = placeholder(shape, Float(32), reduce_name);
  // attr realize scope = local.UB
  // realize whole_red_local_UB
  // for (c1)
  //   for (w, 4)
  //     for (c0)
  //       whole_red_local_UB(0, c1, 0, w, c0) = 0
  auto first = Provide::make(t->op, t->value_index, FloatImm::make(Float(32), 0), args);
  first = For::make(c0, Expr(0), c0_extent, ForType::Serial, DeviceAPI::None, first);
  first = For::make(w, Expr(0), w_extent, ForType::Serial, DeviceAPI::None, first);
  first = For::make(c1, Expr(0), c1_extent, ForType::Serial, DeviceAPI::None, first);

  // for (c1)
  //   for (w, 1, 3)
  //     for (c0)
  //       whole_red_local_UB(0, c1, 0, 0, c0) = whole_red_local_UB(0, c1, 0, 0, c0)
  //       + whole_red_local_UB(0, c1, 0, w, c0)
  auto call = Call::make(Float(32), reduce_name, args, Call::CallType::Halide, t->op, t->value_index);
  Array<Expr> reduce_args;
  reduce_args.push_back(Expr(0));
  reduce_args.push_back(c1);
  reduce_args.push_back(Expr(0));
  reduce_args.push_back(Expr(0));
  reduce_args.push_back(c0);
  auto reduce_call = Call::make(Float(32), reduce_name, reduce_args, Call::CallType::Halide, t->op, t->value_index);
  auto add = Add::make(reduce_call, call);
  auto rest = Provide::make(t->op, t->value_index, add, reduce_args);
  rest = For::make(c0, Expr(0), c0_extent, ForType::Serial, DeviceAPI::None, rest);
  rest = For::make(w, Expr(1), Simplify_cce(w_extent - 1), ForType::Serial, DeviceAPI::None, rest);
  rest = For::make(c1, Expr(0), c1_extent, ForType::Serial, DeviceAPI::None, rest);

  auto stmt = Block::make(first, rest);
  stmt = Realize::make(t->op, t->value_index, Float(32), bounds, const_true(), stmt);
  stmt = AttrStmt::make(t->op, air::ir::attr::realize_scope, Expr("local.UB"), stmt);

  return stmt;
}

class FixOpAfterWholeReduce : public IRMutator {
 public:
  explicit FixOpAfterWholeReduce(int c1_extent) : c1_extent_(c1_extent) {}
  ~FixOpAfterWholeReduce() override = default;

  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (!op->body.as<For>()) {
      Var c1("c1");
      c1_var_ = c1;
      auto body = this->Mutate(op->body);
      body = For::make(op->loop_var, op->min, op->extent, op->for_type, op->device_api, body);
      return For::make(c1_var_, op->min, Expr(c1_extent_), op->for_type, op->device_api, body);
    } else {
      return this->Mutate(op->body);
    }
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    auto args = op->args;
    CHECK_EQ(args.size(), 5);
    Array<Expr> new_args;
    new_args.push_back(args[NN]);
    new_args.push_back(c1_var_);
    new_args.push_back(args[HH]);
    new_args.push_back(args[WW]);
    new_args.push_back(args[C0]);
    auto value = this->Mutate(op->value);
    return Provide::make(op->func, op->value_index, value, new_args);
  }

  Expr Mutate_(const Call *op, const Expr &e) final {
    auto args = op->args;
    CHECK_EQ(args.size(), 5);
    Array<Expr> new_args;
    new_args.push_back(args[NN]);
    new_args.push_back(c1_var_);
    new_args.push_back(args[HH]);
    new_args.push_back(args[WW]);
    new_args.push_back(args[C0]);
    return Call::make(op->type, op->name, new_args, op->call_type, op->func, op->value_index);
  }

 private:
  int c1_extent_{0};
  Var c1_var_;
};

Stmt GetOpAfterReduce(Stmt &s, const Map<std::string, NodeRef> &attrs, const std::string &name) {
  auto f = GatherOpAfterReduce(name);
  f.Visit(s);
  Stmt stmt = Evaluate::make(0);
  const int block_size = 16;
  int kernel_n = GET_INTIMM_ATTR_DEFAULT(attrs, ATTR_CONV_KERNEL_N, 0);
  int c1_extent = kernel_n / block_size;
  int c0_extent = block_size;
  for (auto p : f.op_after_reduce_) {
    p = FixOpAfterWholeReduce(c1_extent).Mutate(p);
    stmt = Block::make(stmt, p);
  }
  Region bounds;
  bounds.push_back(Range::make_by_min_extent(0, 1));
  bounds.push_back(Range::make_by_min_extent(0, c1_extent));
  bounds.push_back(Range::make_by_min_extent(0, 1));
  bounds.push_back(Range::make_by_min_extent(0, 1));
  bounds.push_back(Range::make_by_min_extent(0, c0_extent));
  for (auto r : f.miss_realize_) {
    stmt = Realize::make(r->func, r->value_index, Float(32), bounds, const_true(), stmt);
    stmt = AttrStmt::make(r->func, air::ir::attr::realize_scope, Expr("local.UB"), stmt);
  }
  return stmt;
}

class RmOpAfterReduce : public IRMutator {
 public:
  RmOpAfterReduce() = default;
  ~RmOpAfterReduce() override = default;

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == "pragma_op_after_reduce") {
      return Evaluate::make(0);
    }
    return IRMutator::Mutate_(op, s);
  }
};

Stmt FuseWholeReduceUB(const Stmt &stmt, const Map<std::string, NodeRef> &attrs) {
  Stmt s = stmt;
  auto f = GatherReduceUB();
  f.Visit(s);
  size_t reduce_buffer_size = f.reduce_ubs_.size();
  for (size_t i = 0; i < reduce_buffer_size; ++i) {
    auto reduce_ub_stmt = ReduceUbInit(attrs, i);
    auto attr = reduce_ub_stmt.as<AttrStmt>();
    CHECK(attr);
    auto realize = attr->body.as<Realize>();
    CHECK(realize);
    auto block = realize->body.as<Block>();
    CHECK(block);
    s = TensorSubstitute2(s, f.reduce_ubs_[i], realize->func, realize->value_index);
    auto op_after_reduce = GetOpAfterReduce(s, attrs, realize->func->func_name());
    auto body = Block::make(Block::make(block->first, s), Block::make(block->rest, op_after_reduce));
    s = Realize::make(realize->func, realize->value_index, realize->type, realize->bounds, realize->condition, body);
    s = AttrStmt::make(attr->node, attr->attr_key, attr->value, s);
  }
  return s;
}

class FuseWholeReduceUBInBatch : public IRMutator {
 public:
  FuseWholeReduceUBInBatch(const Map<std::string, NodeRef> &attrs, const Variable *batch_axis)
      : attrs_(attrs), batch_axis_(batch_axis) {}
  ~FuseWholeReduceUBInBatch() override = default;

  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (op->loop_var.get() == batch_axis_) {
      auto attr = op->body.as<AttrStmt>();
      if (attr && attr->attr_key == "pragma_multi_core_depth") {
        auto body = FuseWholeReduceUB(attr->body, attrs_);
        body = AttrStmt::make(attr->node, attr->attr_key, attr->value, body);
        return For::make(op->loop_var, op->min, op->extent, op->for_type, op->device_api, body);
      } else {
        auto body = FuseWholeReduceUB(op->body, attrs_);
        return For::make(op->loop_var, op->min, op->extent, op->for_type, op->device_api, body);
      }
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  Map<std::string, NodeRef> attrs_;
  const Variable *batch_axis_{nullptr};
};

Stmt NewReduceUB(Stmt &s) {
  Convolution collector;
  collector.Visit(s);
  if (IS_ATTR_EXIST(collector.attrs_, ATTR_CONV_KERNEL_N) && IS_ATTR_EXIST(collector.attrs_, ATTR_CONV_FEATURE_NAME)) {
    CHECK(collector.attrs_[ATTR_CONV_FEATURE_NAME].as<StringImm>());
    std::string feature = GET_STRINGIMM_ATTR(collector.attrs_, ATTR_CONV_FEATURE_NAME);
    CHECK(!feature.empty());

    auto f = GetBatchAxis(feature);
    f.Visit(s);
    if (f.batch_axis_) {
      // make block inside of batch axis
      s = FuseWholeReduceUBInBatch(collector.attrs_, f.batch_axis_).Mutate(s);
    } else {
      s = FuseWholeReduceUB(s, collector.attrs_);
    }
    s = RmOpAfterReduce().Mutate(s);
    s = RemoveNullRealize().Mutate(s);
  }
  return s;
}

class RmReduceInit : public IRMutator {
 public:
  RmReduceInit() = default;
  ~RmReduceInit() override = default;

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == "pragma_reduce_init") {
      return Evaluate::make(0);
    }
    return IRMutator::Mutate_(op, s);
  }
};

class ReduceAxisOpt : public IRMutator {
 public:
  ReduceAxisOpt() = default;
  ~ReduceAxisOpt() override = default;

  Stmt Mutate_(const For *op, const Stmt &s) final {
    fors_.clear();
    auto f = op;
    fors_.emplace_back(f);
    while (f->body.as<For>()) {
      f = f->body.as<For>();
      fors_.emplace_back(f);
    }
    auto attr = f->body.as<AttrStmt>();
    if (attr && attr->attr_key == "pragma_reduce_partial_dma_condition") {
      CHECK_GE(fors_.size(), 2);
      std::reverse(fors_.begin(), fors_.end());
      mutate_axis_ = true;
      Var c0_out("c0_out");
      c0_out_ = c0_out;
      auto stmt = this->Mutate(f->body);
      mutate_axis_ = false;
      for (size_t i = 0; i < fors_.size(); i++) {
        auto ff = fors_[i];
        auto extent = ff->extent;
        if (i == 1) {
          extent = Simplify_cce(floordiv(extent, 4));
          stmt = For::make(ff->loop_var, ff->min, extent, ff->for_type, ff->device_api, stmt);
          extent = Expr(4);
          stmt = For::make(c0_out_, ff->min, extent, ff->for_type, ff->device_api, stmt);
          continue;
        }
        stmt = For::make(ff->loop_var, ff->min, extent, ff->for_type, ff->device_api, stmt);
      }
      fors_.clear();
      Var tmp;
      c0_out_ = tmp;
      return stmt;
    }

    fors_.clear();
    return IRMutator::Mutate_(op, s);
  }

  Expr Mutate_(const Call *op, const Expr &e) final {
    if (mutate_axis_) {
      auto args = op->args;
      CHECK_EQ(args.size(), 5);
      Array<Expr> new_args;
      new_args.push_back(args[NN]);
      new_args.push_back(args[C1]);
      new_args.push_back(args[HH]);
      if (op->name.find("red_local_UB") != std::string::npos) {
        CHECK_GE(fors_.size(), 2);
        new_args.push_back(fors_[1]->loop_var);
      } else {
        new_args.push_back(Simplify_cce(c0_out_ * 4 + args[WW]));
      }
      offset_ = Simplify_cce(new_args[WW] - args[WW]);
      new_args.push_back(args[C0]);
      return Call::make(op->type, op->name, new_args, op->call_type, op->func, op->value_index);
    }
    return IRMutator::Mutate_(op, e);
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    if (mutate_axis_) {
      auto args = op->args;
      CHECK_EQ(args.size(), 5);
      Array<Expr> new_args;
      new_args.push_back(args[NN]);
      new_args.push_back(args[C1]);
      new_args.push_back(args[HH]);
      CHECK_GE(fors_.size(), 2);
      new_args.push_back(fors_[1]->loop_var);
      new_args.push_back(args[C0]);
      auto value = this->Mutate(op->value);
      return Provide::make(op->func, op->value_index, value, new_args);
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const IfThenElse *op, const Stmt &s) final {
    if (mutate_axis_) {
      offset_ = Expr(0);
      auto body = this->Mutate(op->then_case);
      auto condition = op->condition.as<LT>();
      CHECK(condition);
      return IfThenElse::make(LT::make(Simplify_cce(condition->a + offset_), condition->b), body);
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  bool mutate_axis_{false};
  std::vector<const For *> fors_;
  Expr offset_{0};
  Var c0_out_;
};

class FixC1Axis : public IRMutator {
 public:
  explicit FixC1Axis(const std::vector<Expr> &c1_offset) : c1_offset_{c1_offset} {}
  ~FixC1Axis() override = default;

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == "pragma_fuse_vector") {
      ++count_;
    }
    if (op->attr_key == "pragma_reduce_partial_dma_condition") {
      fix_provide_ = true;
      Stmt stmt = IRMutator::Mutate_(op, s);
      fix_provide_ = false;
      return stmt;
    }
    return IRMutator::Mutate_(op, s);
  }

  Expr Mutate_(const Call *op, const Expr &e) final {
    if (fix_provide_ && isReduceUB(op->func->func_name())) {
      Array<Expr> args;
      args.push_back(op->args[NN]);
      CHECK_GE(count_, 1);
      CHECK_GE(c1_offset_.size(), count_);
      args.push_back(op->args[C1] + c1_offset_[count_ - 1]);  // non reduce axis
      args.push_back(op->args[HH]);
      args.push_back(op->args[WW]);
      args.push_back(op->args[C0]);  // non reduce axis
      return Call::make(op->type, op->name, args, Call::CallType::Halide, op->func, op->value_index);
    }
    return IRMutator::Mutate_(op, e);
  }
  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    if (fix_provide_ && isReduceUB(op->func->func_name())) {
      auto value = this->Mutate(op->value);
      Array<Expr> args;
      args.push_back(op->args[NN]);
      CHECK_GE(count_, 1);
      CHECK_GE(c1_offset_.size(), count_);
      args.push_back(op->args[C1] + c1_offset_[count_ - 1]);  // non reduce axis
      args.push_back(op->args[HH]);
      args.push_back(op->args[WW]);
      args.push_back(op->args[C0]);  // non reduce axis
      return Provide::make(op->func, op->value_index, value, args);
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  inline bool isReduceUB(const std::string &name) { return name.find("red_local_UB") != std::string::npos; }
  size_t count_{0};
  bool fix_provide_{false};
  std::vector<Expr> c1_offset_;
};

Stmt FixC1ForWholeReduceUB(Stmt &stmt, const Map<Tensor, Buffer> &extern_buffer) {
  auto f = GatherC1Offset(extern_buffer);
  f.Visit(stmt);
  if (!f.c1_offset_.empty()) {
    stmt = FixC1Axis(f.c1_offset_).Mutate(stmt);
  }
  return stmt;
}

Stmt ReduceFusionOpt(Stmt stmt, const Map<Tensor, Buffer> &extern_buffer) {
  ReduceFusionCheck checker;
  checker.Visit(stmt);
  if (!checker.is_reduce_fusion_) {
    return stmt;
  }
  // 1. remove tmp reduce ub init
  stmt = RmReduceInit().Mutate(stmt);
  stmt = RemoveNoOp(stmt);
  // 2. change reduce op from 16*16 to 4*4*16
  stmt = ReduceAxisOpt().Mutate(stmt);
  // 3. allocate whole reduce ub[0, c1, 0, 4, c0] & init ub & ub reduce & sink op after reduce
  stmt = NewReduceUB(stmt);
  // 4. add c1 offset to tmp reduce ub
  stmt = FixC1ForWholeReduceUB(stmt, extern_buffer);

  return stmt;
}
}  // namespace ir
}  // namespace akg
