/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file ir.cc
 */

/*
 * 2019.12.30 - Add new conditions for call types.
 * 2022.05.30 - Fix types of base and stride did not match in Ramp.
 * 2023.02.03 - Add bounds, stride and attrs of node.
 * 2023.03.25 - Add TVM 0.8 attributes to the node and conversion pass for exporting TVM 0.8 IR.
 */

#include <tvm/base.h>
#include <tvm/expr.h>
#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <tvm/string.h>
#include <memory>
#include "../pass/ir_util.h"

namespace air {
namespace ir {

// constructors
Expr UIntImm::make(DataType t, uint64_t value) {
  CHECK(t.is_uint() && t.lanes() == 1)
      << "ValueError: UIntImm can only take scalar";
  NodePtr<UIntImm> node = make_node<UIntImm>();
  node->type = t;
  node->value = value;
  return Expr(node);
}

Expr FloatImm::make(DataType t, double value) {
  CHECK_EQ(t.lanes(), 1)
      << "ValueError: FloatImm can only take scalar";
  NodePtr<FloatImm> node = make_node<FloatImm>();
  node->type = t;
  node->value = value;
  return Expr(node);
}

Expr StringImm::make(std::string value) {
  NodePtr<StringImm> node = make_node<StringImm>();
  node->type = Handle();
  node->value = std::move(value);
  return Expr(node);
}

Expr Cast::make(DataType t, Expr value) {
  CHECK(value.defined());
  CHECK_EQ(t.lanes(), value.type().lanes());
  NodePtr<Cast> node = make_node<Cast>();
  node->type = t;
  node->value = std::move(value);
  return Expr(node);
}

Expr And::make(Expr a, Expr b) {
  CHECK(a.defined()) << "ValueError: a is undefined";
  CHECK(b.defined()) << "ValueError: b is undefined";
  CHECK(a.type().is_bool()||a.type().is_int());
  CHECK(b.type().is_bool()||b.type().is_int());
  CHECK(a.type() == b.type()) << "TypeError: mismatched types";

  NodePtr<And> node = make_node<And>();
  if (a.type().is_bool()) {
    node->type = Bool(a.type().lanes());
  } else {
    node->type = Int(32);
  }
  node->a = std::move(a);
  node->b = std::move(b);
  return Expr(node);
}

Expr Or::make(Expr a, Expr b) {
  CHECK(a.defined()) << "ValueError: a is undefined";
  CHECK(b.defined()) << "ValueError: b is undefined";
  CHECK(a.type().is_bool()||a.type().is_int());
  CHECK(b.type().is_bool()||b.type().is_int());
  CHECK(a.type() == b.type()) << "TypeError: mismatched types";

  NodePtr<Or> node = make_node<Or>();
  if (a.type().is_bool()) {
    node->type = Bool(a.type().lanes());
  } else {
    node->type = Int(32);
  }
  node->a = std::move(a);
  node->b = std::move(b);
  return Expr(node);
}

Expr Not::make(Expr a) {
  CHECK(a.defined()) << "ValueError: a is undefined";
  CHECK(a.type().is_bool());

  NodePtr<Not> node = make_node<Not>();
  node->type = Bool(a.type().lanes());
  node->a = std::move(a);
  return Expr(node);
}

Expr Select::make(Expr condition, Expr true_value, Expr false_value) {
  CHECK(condition.defined()) << "ValueError: condition is undefined";
  CHECK(true_value.defined()) << "ValueError: true_value is undefined";
  CHECK(false_value.defined()) << "ValueError: true_value is undefined";
  CHECK(condition.type().is_bool());
  CHECK_EQ(condition.type().lanes(), true_value.type().lanes());
  CHECK(false_value.type() == true_value.type()) << "TypeError: mismatched types";

  NodePtr<Select> node = make_node<Select>();
  node->type = true_value.type();
  node->condition = std::move(condition);
  node->true_value = std::move(true_value);
  node->false_value = std::move(false_value);
  return Expr(node);
}

Expr Load::make(DataType type, Var buffer_var, Expr index, Expr predicate) {
  CHECK(buffer_var.defined());
  CHECK(predicate.defined());
  CHECK(index.defined());
  CHECK_EQ(type.lanes(), index.type().lanes());
  CHECK_EQ(type.lanes(), predicate.type().lanes());

  NodePtr<Load> node = make_node<Load>();
  node->type = type;
  node->buffer_var = std::move(buffer_var);
  node->index = std::move(index);
  node->predicate = std::move(predicate);

  return Expr(node);
}

Expr Ramp::make(Expr base, Expr stride, int lanes) {
  CHECK(base.defined());
  CHECK(stride.defined());
  CHECK(base.type().is_scalar());
  CHECK(stride.type().is_scalar());
  CHECK_GT(lanes, 1);
  bool need_cast = ((base.type() != stride.type()) && ((base.type().is_int() || base.type().is_uint()) &&
                   (stride.type().is_int() || stride.type().is_uint())));
  if (need_cast) {
    bool cast_base = ((base.type().bits() < stride.type().bits()) ||
                     ((base.type().bits() == stride.type().bits()) && (stride.type().is_int())));
    if (cast_base) {
      base = Cast::make(stride.type(), base);
    } else {
      stride = Cast::make(base.type(), stride);
    }
  }
  CHECK_EQ(stride.type(), base.type());

  NodePtr<Ramp> node = make_node<Ramp>();
  node->type = base.type().with_lanes(lanes);
  node->base = base;
  node->stride = stride;
  node->lanes = lanes;
  return Expr(node);
}

Expr Broadcast::make(Expr value, int lanes) {
  CHECK(value.defined());
  CHECK(value.type().is_scalar());
  CHECK_GT(lanes, 1);

  NodePtr<Broadcast> node = make_node<Broadcast>();
  node->type = value.type().with_lanes(lanes);
  node->value = std::move(value);
  node->lanes = lanes;
  return Expr(node);
}

Expr Let::make(Var var, Expr value, Expr body) {
  CHECK(value.defined());
  CHECK(body.defined());
  CHECK_EQ(value.type(), var.type());

  NodePtr<Let> node = make_node<Let>();
  node->type = body.type();
  node->var = std::move(var);
  node->value = std::move(value);
  node->body = std::move(body);
  return Expr(node);
}

const char* Call::vectorizable_intrinsics[] = {
    "floor", "ceil", "sign", "trunc", "fabs", "round", "exp", "tanh", "sqrt",
    "log", "sin", "cos", "pow", ir::Call::shift_left, ir::Call::shift_right,
    ir::Call::likely, ir::Call::popcount
};

bool Call::is_vectorizable() const {
  size_t cnt = sizeof(Call::vectorizable_intrinsics) / sizeof(char*);
  for (size_t i = 0; i < cnt; ++i) {
    if (name == Call::vectorizable_intrinsics[i]) {
      return true;
    }
  }
  return false;
}

Expr Call::make(DataType type,
                std::string name,
                Array<Expr> args,
                CallType call_type,
                FunctionRef func,
                int value_index,
                Map<std::string, NodeRef> attrs) {
  for (size_t i = 0; i < args.size(); ++i) {
    CHECK(args[i].defined());
  }
  for (auto iv : attrs) {
    CHECK(iv.second.defined());
  }

  if (call_type == Halide) {
    for (size_t i = 0; i < args.size(); ++i) {
      CHECK(args[i].type().is_int());
    }
  }

  NodePtr<Call> node = make_node<Call>();
  node->type = type;
  node->name = std::move(name);
  node->args = std::move(args);
  node->call_type = call_type;
  node->func = std::move(func);
  node->value_index = value_index;
  node->attrs = std::move(attrs);
  return Expr(node);
}

Expr Shuffle::make(Array<Expr> vectors,
                   Array<Expr> indices) {
  CHECK_NE(vectors.size(), 0U);
  CHECK_NE(indices.size(), 0U);

  Type base_type = vectors[0].type().element_of();
  int total_lanes = 0;

  for (Expr val : vectors) {
    CHECK(val.type().element_of() == base_type);
    total_lanes += val.type().lanes();
  }
  CHECK_LE(indices.size(), static_cast<size_t>(total_lanes));

  NodePtr<Shuffle> node = make_node<Shuffle>();
  node->type = base_type.with_lanes(static_cast<int>(indices.size()));
  node->vectors = std::move(vectors);
  node->indices = std::move(indices);
  return Expr(node);
}

Expr Shuffle::make_concat(Array<Expr> vectors) {
  CHECK_NE(vectors.size(), 0);
  if (vectors.size() == 1) {
    return vectors[0];
  }
  Array<Expr> indices;
  int index = 0;
  for (const Expr& e : vectors) {
    for (int i = 0; i < e.type().lanes(); ++i) {
      indices.push_back(IntImm::make(Int(32), index++));
    }
  }
  return make(vectors, indices);
}

Expr Shuffle::make_extract_element(Expr vector, int index) {
  return make({vector}, {Integer(index)});
}

CommReducer CommReducerNode::make(Array<Var> lhs,
                                  Array<Var> rhs,
                                  Array<Expr> result,
                                  Array<Expr> identity_element) {
  auto node = make_node<CommReducerNode>();
  node->lhs = lhs;
  node->rhs = rhs;
  node->result = result;
  node->identity_element = identity_element;
  return CommReducer(node);
}

Array<Expr> CommReducerNode::operator()(Array<Expr> a, Array<Expr> b) const {
  CHECK_EQ(a.size(), b.size());
  CHECK_EQ(lhs.size(), a.size());
  CHECK_EQ(rhs.size(), b.size());
  Map<Var, Expr> value_map;
  for (size_t i = 0; i < a.size(); ++i) {
    value_map.Set(lhs[i], a[i]);
    value_map.Set(rhs[i], b[i]);
  }
  return UpdateArray(result, [&value_map] (const Expr& e) {
      return Substitute(e, value_map);
    });
}

Expr Reduce::make(CommReducer combiner, Array<Expr> source,
                  Array<IterVar> axis, Expr condition, int value_index) {
  for (size_t i = 0; i < axis.size(); ++i) {
    CHECK_EQ(axis[i]->iter_type, kCommReduce)
        << "Can only take axis created by reduce_axis";
  }
  if (!condition.defined()) {
    condition = const_true();
  }
  auto n = make_node<Reduce>();
  CHECK(source.defined());
  for (size_t i = 0; i < axis.size(); ++i) {
    CHECK(axis[i].defined());
  }
  n->type = source[value_index].type();
  n->combiner = std::move(combiner);
  n->source = std::move(source);
  n->axis = std::move(axis);
  n->condition = condition;
  n->value_index = value_index;
  return Expr(n);
}

Expr Any::make() {
  auto n = make_node<Any>();
  return Expr(n);
}

Stmt LetStmt::make(Var var, Expr value, Stmt body) {
  CHECK(value.defined());
  CHECK(body.defined());
  CHECK_EQ(value.type(), var.type());

  NodePtr<LetStmt> node = make_node<LetStmt>();
  node->var = std::move(var);
  node->value = std::move(value);
  node->body = std::move(body);
  return Stmt(node);
}

Stmt AttrStmt::make(NodeRef node,
                    std::string attr_key,
                    Expr value,
                    Stmt body,
                    Region bounds) {
  auto n = make_node<AttrStmt>();
  n->node = node;
  n->attr_key = std::move(attr_key);
  n->value = std::move(value);
  n->body = std::move(body);
  n->bounds = std::move(bounds);
  return Stmt(n);
}

Stmt AssertStmt::make(Expr condition, Expr message, Stmt body) {
  CHECK(condition.defined());
  CHECK(message.type() == Int(32) ||
        message.as<StringImm>())
      << "TypeError: AssertStmt message must be an int or string:"
      << message << "\n";

  NodePtr<AssertStmt> node = make_node<AssertStmt>();
  node->condition = std::move(condition);
  node->message = std::move(message);
  node->body = std::move(body);
  return Stmt(node);
}

Stmt ProducerConsumer::make(FunctionRef func, bool is_producer, Stmt body) {
  CHECK(body.defined());

  NodePtr<ProducerConsumer> node = make_node<ProducerConsumer>();
  node->func = std::move(func);
  node->is_producer = is_producer;
  node->body = std::move(body);
  return Stmt(node);
}

Stmt For::make(Var loop_var,
               Expr min,
               Expr extent,
               ForType for_type,
               DeviceAPI device_api,
               Stmt body,
               Expr stride) {
  CHECK(min.defined());
  CHECK(extent.defined());
  CHECK(min.type().is_scalar());
  CHECK(extent.type().is_scalar());
  CHECK(loop_var.type().is_scalar());
  CHECK(body.defined());

  NodePtr<For> node = make_node<For>();
  node->loop_var = std::move(loop_var);
  node->min = std::move(min);
  node->extent = std::move(extent);
  node->for_type = for_type;
  node->device_api = device_api;
  node->body = std::move(body);
  node->stride = std::move(stride);
  return Stmt(node);
}

Stmt Store::make(Var buffer_var, Expr value, Expr index, Expr predicate) {
  CHECK(value.defined());
  CHECK(index.defined());
  CHECK(predicate.defined());
  CHECK_EQ(value.type().lanes(), index.type().lanes());
  CHECK_EQ(value.type().lanes(), predicate.type().lanes());

  NodePtr<Store> node = make_node<Store>();
  node->buffer_var = std::move(buffer_var);
  node->value = std::move(value);
  node->index = std::move(index);
  node->predicate = std::move(predicate);
  return Stmt(node);
}

Stmt Provide::make(FunctionRef func, int value_index, Expr value, Array<Expr> args) {
  CHECK(value_index >=0 && value_index < func->num_outputs())
      << "value index output function return value bound";
  CHECK(value.defined()) << "Provide of undefined value\n";

  for (size_t i = 0; i < args.size(); ++i) {
    CHECK(args[i].defined()) << "Provide to undefined location\n";
  }

  NodePtr<Provide> node = make_node<Provide>();
  node->func = std::move(func);
  node->value_index = value_index;
  node->value = std::move(value);
  node->args = std::move(args);
  return Stmt(node);
}

Stmt Allocate::make(Var buffer_var,
                    DataType type,
                    Array<Expr> extents,
                    Expr condition,
                    Stmt body,
                    Expr new_expr,
                    std::string free_function) {
    for (size_t i = 0; i < extents.size(); ++i) {
      CHECK(extents[i].defined());
      CHECK(extents[i].type().is_scalar());
    }
    CHECK(body.defined());
    CHECK(condition.defined());
    CHECK(condition.type().is_bool());

    NodePtr<Allocate> node = make_node<Allocate>();
    node->buffer_var = std::move(buffer_var);
    node->type = type;
    node->extents = std::move(extents);
    node->condition = std::move(condition);
    node->body = std::move(body);
    node->new_expr = std::move(new_expr);
    node->free_function = std::move(free_function);
    return Stmt(node);
}

int32_t Allocate::constant_allocation_size(const Array<Expr>& extents) {
  int64_t result = 1;
  for (size_t i = 0; i < extents.size(); ++i) {
    if (const IntImm *int_size = extents[i].as<IntImm>()) {
      result *= int_size->value;
      if (result > std::numeric_limits<int32_t>::max()) {
        return 0;
      }
    } else {
      return 0;
    }
  }
  return static_cast<int32_t>(result);
}

Stmt Free::make(Var buffer_var) {
  NodePtr<Free> node = make_node<Free>();
  node->buffer_var = buffer_var;
  return Stmt(node);
}

Stmt Realize::make(FunctionRef func,
                   int value_index,
                   DataType type,
                   Region bounds,
                   Expr condition,
                   Stmt body) {
  for (size_t i = 0; i < bounds.size(); ++i) {
    CHECK(bounds[i]->min.defined());
    CHECK(bounds[i]->extent.defined());
    CHECK(bounds[i]->min.type().is_scalar());
    CHECK(bounds[i]->extent.type().is_scalar());
  }
  CHECK(body.defined());
  CHECK(condition.defined());
  CHECK(condition.type().is_bool());

  NodePtr<Realize> node = make_node<Realize>();
  node->func = std::move(func);
  node->value_index = value_index;
  node->type = type;
  node->bounds = std::move(bounds);
  node->condition = std::move(condition);
  node->body = std::move(body);
  return Stmt(node);
}

Stmt Prefetch::make(FunctionRef func, int value_index, DataType type, Region bounds) {
  for (size_t i = 0; i < bounds.size(); ++i) {
    CHECK(bounds[i]->min.defined());
    CHECK(bounds[i]->extent.defined());
    CHECK(bounds[i]->min.type().is_scalar());
    CHECK(bounds[i]->extent.type().is_scalar());
  }

  NodePtr<Prefetch> node = make_node<Prefetch>();
  node->func = std::move(func);
  node->value_index = value_index;
  node->type = type;
  node->bounds = std::move(bounds);
  return Stmt(node);
}

Stmt Block::make(Stmt first, Stmt rest) {
  CHECK(first.defined());
  CHECK(rest.defined());
  NodePtr<Block> node = make_node<Block>();

  // canonicalize.
  if (const Block* b = first.as<Block>()) {
    node->first = b->first;
    node->rest = Block::make(b->rest, rest);
  } else {
    node->first = std::move(first);
    node->rest = std::move(rest);
  }
  return Stmt(node);
}

Stmt Block::make(const std::vector<Stmt>& stmts) {
  if (stmts.empty()) {
    return Stmt();
  }
  Stmt result = stmts.back();
  for (size_t i = stmts.size() - 1; i != 0; --i) {
    result = Block::make(stmts[i - 1], result);
  }
  return result;
}

Stmt IfThenElse::make(Expr condition, Stmt then_case, Stmt else_case) {
  CHECK(condition.defined());
  CHECK(then_case.defined());
  // else_case may be null.

  NodePtr<IfThenElse> node = make_node<IfThenElse>();
  node->condition = std::move(condition);
  node->then_case = std::move(then_case);
  node->else_case = std::move(else_case);
  return Stmt(node);
}

Stmt Evaluate::make(Expr value) {
  CHECK(value.defined());

  NodePtr<Evaluate> node = make_node<Evaluate>();
  node->value = std::move(value);
  return Stmt(node);
}

struct RefToObjectPtr : public ObjectRef {
  static ObjectPtr<Object> Get(const ObjectRef& ref) { return GetDataPtr<Object>(ref); }
};

TVM_REGISTER_REFLECTION_VTABLE(runtime::StringObj,
                               ::air::detail::ReflectionTrait<runtime::StringObj>)
    .set_creator([](const std::string& bytes) {
      return RefToObjectPtr::Get(runtime::String(bytes));
    })
    .set_repr_bytes([](const Object* n) -> std::string {
      return GetRef<runtime::String>(static_cast<const runtime::StringObj*>(n))
          .
          operator std::string();
    });

IntrinsicOp::IntrinsicOp(String name){
  ObjectPtr<IntrinsicOpNode> n = make_object<IntrinsicOpNode>();
  n->name = std::move(name);
  data_ = std::move(n);
}

// Printers
TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<UIntImm>([](const ObjectRef& node, IRPrinter* p) {
    auto* op = static_cast<const UIntImm*>(node.get());
    p->stream << "(" << op->type << ")" << op->value;
  });

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<FloatImm>([](const ObjectRef& node, IRPrinter* p) {
    auto* op = static_cast<const FloatImm*>(node.get());
    auto& stream = p->stream;
    switch (op->type.bits()) {
      case 64:
        stream << op->value;
        break;
      case 32:
        stream << op->value << 'f';
        break;
      case 16:
        stream << op->value << 'h';
        break;
      default:
        LOG(FATAL) << "Unknown float type bits=" << op->type.bits();
    }
  });

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<StringImm>([](const ObjectRef& node, IRPrinter* p) {
    auto* op = static_cast<const StringImm*>(node.get());
    auto& stream = p->stream;
    stream << '"';
    for (size_t i = 0; i < op->value.size(); ++i) {
      unsigned char c = op->value[i];
      if (c >= ' ' && c <= '~' && c != '\\' && c != '"') {
        stream << c;
      } else {
        stream << '\\';
        switch (c) {
          case '"':
            stream << '"';
            break;
          case '\\':
            stream << '\\';
            break;
          case '\t':
            stream << 't';
            break;
          case '\r':
            stream << 'r';
            break;
          case '\n':
            stream << 'n';
            break;
          default:
            const char* hex_digits = "0123456789ABCDEF";
            stream << 'x' << hex_digits[c >> 4] << hex_digits[c & 0xf];
        }
      }
    }
    stream << '"';
  });

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<Cast>([](const ObjectRef& node, IRPrinter* p) {
    auto* op = static_cast<const Cast*>(node.get());
    p->stream << op->type << '(';
    p->Print(op->value);
    p->stream << ')';
  })
.set_dispatch<Variable>([](const ObjectRef& node, IRPrinter* p) {
    auto* op = static_cast<const Variable*>(node.get());
    // omit the type
    // stream << op->name << "." << op->type;
    p->stream << op->name_hint;
  })
.set_dispatch<Add>([](const ObjectRef& node, IRPrinter* p) {
    auto* op = static_cast<const Add*>(node.get());
    p->stream << '(';
    p->Print(op->a);
    p->stream << " + ";
    p->Print(op->b);
    p->stream << ')';
  })
.set_dispatch<Sub>([](const ObjectRef& node, IRPrinter* p) {
    auto* op = static_cast<const Sub*>(node.get());
    p->stream << '(';
    p->Print(op->a);
    p->stream << " - ";
    p->Print(op->b);
    p->stream << ')';
  })
.set_dispatch<Mul>([](const ObjectRef& node, IRPrinter* p) {
    auto* op = static_cast<const Mul*>(node.get());
    p->stream << '(';
    p->Print(op->a);
    p->stream << "*";
    p->Print(op->b);
    p->stream << ')';
  })
.set_dispatch<Div>([](const ObjectRef& node, IRPrinter* p) {
    auto* op = static_cast<const Div*>(node.get());
    p->stream << '(';
    p->Print(op->a);
    p->stream << "/";
    p->Print(op->b);
    p->stream << ')';
  })
.set_dispatch<Mod>([](const ObjectRef& node, IRPrinter* p) {
    auto* op = static_cast<const Mod*>(node.get());
    p->stream << '(';
    p->Print(op->a);
    p->stream << " % ";
    p->Print(op->b);
    p->stream << ')';
})
.set_dispatch<Min>([](const ObjectRef& node, IRPrinter* p) {
    auto* op = static_cast<const Min*>(node.get());
    p->stream << "min(";
    p->Print(op->a);
    p->stream << ", ";
    p->Print(op->b);
    p->stream << ")";
})
.set_dispatch<Max>([](const ObjectRef& node, IRPrinter* p) {
    auto* op = static_cast<const Max*>(node.get());
    p->stream << "max(";
    p->Print(op->a);
    p->stream << ", ";
    p->Print(op->b);
    p->stream << ")";
})
.set_dispatch<EQ>([](const ObjectRef& node, IRPrinter* p) {
    auto* op = static_cast<const EQ*>(node.get());
    p->stream << '(';
    p->Print(op->a);
    p->stream << " == ";
    p->Print(op->b);
    p->stream << ')';
})
.set_dispatch<NE>([](const ObjectRef& node, IRPrinter* p) {
    auto* op = static_cast<const NE*>(node.get());
    p->stream << '(';
    p->Print(op->a);
    p->stream << " != ";
    p->Print(op->b);
    p->stream << ')';
})
.set_dispatch<LT>([](const ObjectRef& node, IRPrinter* p) {
    auto* op = static_cast<const LT*>(node.get());
    p->stream << '(';
    p->Print(op->a);
    p->stream << " < ";
    p->Print(op->b);
    p->stream << ')';
})
.set_dispatch<LE>([](const ObjectRef& node, IRPrinter* p) {
    auto* op = static_cast<const LE*>(node.get());
    p->stream << '(';
    p->Print(op->a);
    p->stream << " <= ";
    p->Print(op->b);
    p->stream << ')';
})
.set_dispatch<GT>([](const ObjectRef& node, IRPrinter* p) {
    auto* op = static_cast<const GT*>(node.get());
    p->stream << '(';
    p->Print(op->a);
    p->stream << " > ";
    p->Print(op->b);
    p->stream << ')';
})
.set_dispatch<GE>([](const ObjectRef& node, IRPrinter* p) {
    auto* op = static_cast<const GE*>(node.get());
    p->stream << '(';
    p->Print(op->a);
    p->stream << " >= ";
    p->Print(op->b);
    p->stream << ')';
});

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<FloorDiv>([](const ObjectRef& node, IRPrinter* p) {
    auto* op = static_cast<const FloorDiv*>(node.get());
  p->stream << "floordiv(" << op->a << ", " << op->b << ")";
});

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<FloorMod>([](const ObjectRef& node, IRPrinter* p) {
    auto* op = static_cast<const FloorMod*>(node.get());
  p->stream << "floormod(" << op->a << ", " << op->b << ")";
});

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<And>([](const ObjectRef& node, IRPrinter* p) {
    auto* op = static_cast<const And*>(node.get());
    p->stream << '(';
    p->Print(op->a);
    p->stream << " && ";
    p->Print(op->b);
    p->stream << ')';
});

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<Or>([](const ObjectRef& node, IRPrinter* p) {
    auto* op = static_cast<const Or*>(node.get());
    p->stream << '(';
    p->Print(op->a);
    p->stream << " || ";
    p->Print(op->b);
    p->stream << ')';
});

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<Not>([](const ObjectRef& node, IRPrinter* p) {
    auto* op = static_cast<const Not*>(node.get());
    p->stream << '!';
    p->Print(op->a);
});

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<Select>([](const ObjectRef& node, IRPrinter* p) {
    auto* op = static_cast<const Select*>(node.get());
    p->stream << "select(";
    p->Print(op->condition);
    p->stream << ", ";
    p->Print(op->true_value);
    p->stream << ", ";
    p->Print(op->false_value);
    p->stream << ")";
});

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<Load>([](const ObjectRef& node, IRPrinter* p) {
    auto* op = static_cast<const Load*>(node.get());
    p->stream << op->buffer_var << "[";
    p->Print(op->index);
    p->stream << "]";
    if (!is_one(op->predicate)) {
        p->stream << " if ";
        p->Print(op->predicate);
    }
});

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<Ramp>([](const ObjectRef& node, IRPrinter* p) {
    auto* op = static_cast<const Ramp*>(node.get());
    p->stream << "ramp(";
    p->Print(op->base);
    p->stream << ", ";
    p->Print(op->stride);
    p->stream << ", " << op->lanes << ")";
});

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<Broadcast>([](const ObjectRef& node, IRPrinter* p) {
    auto* op = static_cast<const Broadcast*>(node.get());
    p->stream << "x" << op->lanes << "(";
    p->Print(op->value);
    p->stream << ")";
});

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<Call>([](const ObjectRef& node, IRPrinter* p) {
    auto* op = static_cast<const Call*>(node.get());
    p->stream << op->name << "(";
    for (size_t i = 0; i < op->args.size(); ++i) {
      p->Print(op->args[i]);
      if (i < op->args.size() - 1) {
        p->stream << ", ";
      }
    }
    p->stream << ")";
    if (op->call_type != Call::Halide) {
        p->stream << ":" << op->type << ":";
        switch (op->call_type) {
            case Call::ExternCPlusPlus:
                p->stream << "ECPP"; break;
            case Call::PureExtern:
                p->stream << "PE"; break;
            case Call::Extern:
                p->stream << "EX"; break;
            case Call::Intrinsic:
                p->stream << "I"; break;
            case Call::PureIntrinsic:
                p->stream << "PI"; break;
            case Call::Halide:
            default:
                break;
        }
    }
});

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<Let>([](const ObjectRef& node, IRPrinter* p) {
    auto* op = static_cast<const Let*>(node.get());
    p->stream << "(let " << op->var << " = ";
    p->Print(op->value);
    p->stream << " in ";
    p->Print(op->body);
    p->stream << ")";
});

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<LetStmt>([](const ObjectRef& node, IRPrinter* p) {
    auto* op = static_cast<const LetStmt*>(node.get());
    p->PrintIndent();
    p->stream << "let " << op->var << " = ";
    p->Print(op->value);
    p->stream << '\n';
    p->Print(op->body);
  });

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<AttrStmt>([](const ObjectRef& node, IRPrinter* p) {
    auto* op = static_cast<const AttrStmt*>(node.get());
    p->PrintIndent();
    p->stream << "// attr [";
    p->Print(op->node);
    p->stream << "] "
              << op->attr_key << " = ";
    p->Print(op->value);
    p->stream << '\n';
    p->Print(op->body);
  });

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<AssertStmt>([](const ObjectRef& node, IRPrinter* p) {
    auto* op = static_cast<const AssertStmt*>(node.get());
    p->PrintIndent();
    p->stream << "assert(";
    p->Print(op->condition);
    p->stream << ", ";
    p->Print(op->message);
    p->stream << ")\n";
    p->Print(op->body);
  });

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<ProducerConsumer>([](const ObjectRef& node, IRPrinter* p) {
    auto* op = static_cast<const ProducerConsumer*>(node.get());
    if (op->is_producer) {
      p->PrintIndent();
      p->stream << "produce " << op->func->func_name() << " {\n";
      p->indent += 2;
      p->Print(op->body);
      p->indent -= 2;
      p->PrintIndent();
      p->stream << "}\n";
    } else {
      p->Print(op->body);
    }
  });

std::ostream &operator<<(std::ostream& out, ForType type) { // NOLINT(*)
  switch (type) {
    case ForType::Serial:
      out << "for";
      break;
    case ForType::Invariant:
      out << "invariant_for";
      break;
    case ForType::Reduce:
      out << "reduce";
      break;
    case ForType::Parallel:
      out << "parallel";
      break;
    case ForType::Unrolled:
      out << "unrolled";
      break;
    case ForType::Vectorized:
      out << "vectorized";
      break;
    case ForType::Swizzled:
      out << "swizzled";
      break;
  }
  return out;
}

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<For>([](const ObjectRef& node, IRPrinter* p) {
    auto* op = static_cast<const For*>(node.get());
    p->PrintIndent();
    p->stream << op->for_type << " (" << op->loop_var << ", ";
    p->Print(op->min);
    p->stream << ", ";
    p->Print(op->extent);
    p->stream << ") {\n";

    p->indent += 2;
    p->Print(op->body);
    p->indent -= 2;

    p->PrintIndent();
    p->stream << "}\n";
});

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<Store>([](const ObjectRef& node, IRPrinter* p) {
    auto* op = static_cast<const Store*>(node.get());
    p->PrintIndent();
    p->stream << op->buffer_var << "[";
    p->Print(op->index);
    p->stream << "] = ";
    p->Print(op->value);
    if (!is_one(op->predicate)) {
      p->stream << " if ";
      p->Print(op->predicate);
    }
    p->stream << '\n';
  });

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<Provide>([](const ObjectRef& node, IRPrinter* p) {
    auto* op = static_cast<const Provide*>(node.get());
    p->PrintIndent();
    p->stream << op->func->func_name() << "(";
    for (size_t i = 0; i < op->args.size(); ++i) {
      p->Print(op->args[i]);
      if (i < op->args.size() - 1) p->stream << ", ";
    }
    p->stream << ")";
    if (op->func->num_outputs() != 1) {
      p->stream << ".value[" << op->value_index << "]";
    }
    p->stream << " = ";
    p->Print(op->value);
    p->stream << '\n';
  });

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<Allocate>([](const ObjectRef& node, IRPrinter* p) {
    auto* op = static_cast<const Allocate*>(node.get());
    p->PrintIndent();
    p->stream << "allocate " << op->buffer_var << "[" << op->type;
    for (size_t i = 0; i < op->extents.size(); ++i) {
      p->stream << " * ";
      p->Print(op->extents[i]);
    }
    p->stream << "]";
    if (!is_one(op->condition)) {
      p->stream << " if ";
      p->Print(op->condition);
    }
    p->stream << "\n";
    p->Print(op->body);
  });

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<Free>([](const ObjectRef& node, IRPrinter* p) {
    auto* op = static_cast<const Free*>(node.get());
    p->PrintIndent();
    p->stream << "free " << op->buffer_var;
    p->stream << '\n';
  });

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<Realize>([](const ObjectRef& node, IRPrinter* p) {
    auto* op = static_cast<const Realize*>(node.get());
    p->PrintIndent();
    p->stream << "realize " << op->func->func_name()
              << "<" << op->type << ">(";
    for (size_t i = 0; i < op->bounds.size(); ++i) {
      p->stream << "[";
      p->Print(op->bounds[i]->min);
      p->stream << ", ";
      p->Print(op->bounds[i]->extent);
      p->stream << "]";
      if (i < op->bounds.size() - 1) p->stream << ", ";
    }
    p->stream << ")";
    if (op->func->num_outputs() != 1) {
      p->stream << ".value[" << op->value_index << "]";
    }
    if (!is_one(op->condition)) {
      p->stream << " if ";
      p->Print(op->condition);
    }
    p->stream << " {\n";

    p->indent += 2;
    p->Print(op->body);
    p->indent -= 2;

    p->PrintIndent();
    p->stream << "}\n";
  });

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<Prefetch>([](const ObjectRef& node, IRPrinter* p) {
    auto* op = static_cast<const Prefetch*>(node.get());
    p->PrintIndent();
    p->stream << "prefetch " << op->func->func_name() << "(";
    for (size_t i = 0; i < op->bounds.size(); ++i) {
      p->stream << "[";
      p->Print(op->bounds[i]->min);
      p->stream << ", ";
      p->Print(op->bounds[i]->extent);
      p->stream << "]";
      if (i < op->bounds.size() - 1) p->stream << ", ";
    }
    p->stream << ")";
    if (op->func->num_outputs() != 1) {
      p->stream << ".value[" << op->value_index << "]";
    }
  });

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<Block>([](const ObjectRef& node, IRPrinter* p) {
    auto* op = static_cast<const Block*>(node.get());
    p->Print(op->first);
    if (op->rest.defined()) p->Print(op->rest);
  });

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<IfThenElse>([](const ObjectRef& node, IRPrinter* p) {
    auto* op = static_cast<const IfThenElse*>(node.get());
    p->PrintIndent();
    while (true) {
      p->stream << "if (" << op->condition << ") {\n";
      p->indent += 2;
      p->Print(op->then_case);
      p->indent -= 2;

      if (!op->else_case.defined()) {
        break;
      }

      if (const IfThenElse *nested_if = op->else_case.as<IfThenElse>()) {
        p->PrintIndent();
        p->stream << "} else ";
        op = nested_if;
      } else {
        p->PrintIndent();
        p->stream << "} else {\n";
        p->indent += 2;
        p->Print(op->else_case);
        p->indent -= 2;
        break;
      }
    }
    p->PrintIndent();
    p->stream << "}\n";
});

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<Evaluate>([](const ObjectRef& node, IRPrinter* p) {
    auto* op = static_cast<const Evaluate*>(node.get());
    p->PrintIndent();
    p->Print(op->value);
    p->stream << "\n";
  });

template<typename T>
void PrintList(const Array<T> &exprs, IRPrinter* p) {
  for (size_t i = 0; i < exprs.size(); ++i) {
    p->Print(exprs[i]);
    if (i < exprs.size() - 1) {
      p->stream << ", ";
    }
  }
}

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<Shuffle>([](const ObjectRef& node, IRPrinter* p) {
    auto* op = static_cast<const Shuffle*>(node.get());
    p->stream << "shuffle(";
    PrintList(op->vectors, p);
    p->stream << ", ";
    PrintList(op->indices, p);
    p->stream << ")";
  });

// Container printer
TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<ArrayNode>([](const ObjectRef& node, IRPrinter* p) {
    auto* op = static_cast<const ArrayNode*>(node.get());
    p->stream << '[';
    for (size_t i = 0 ; i < op->data.size(); ++i) {
      if (i != 0) {
        p->stream << ", ";
      }
      p->Print(op->data[i]);
    }
    p->stream << ']';
});

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<MapNode>([](const ObjectRef& node, IRPrinter* p) {
    auto* op = static_cast<const MapNode*>(node.get());
    p->stream << '{';
    for (auto it = op->data.begin(); it != op->data.end(); ++it) {
      if (it != op->data.begin()) {
        p->stream << ", ";
      }
      p->Print(it->first);
      p->stream << ": ";
      p->Print(it->second);
    }
    p->stream << '}';
  });

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<StrMapNode>([](const ObjectRef& node, IRPrinter* p) {
    auto* op = static_cast<const StrMapNode*>(node.get());
    p->stream << '{';
    for (auto it = op->data.begin(); it != op->data.end(); ++it) {
      if (it != op->data.begin()) {
        p->stream << ", ";
      }
      p->stream << '\"' << it->first << "\": ";
      p->Print(it->second);
    }
    p->stream << '}';
  });

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<Reduce>([](const ObjectRef& node, IRPrinter* p) {
    auto* op = static_cast<const Reduce*>(node.get());
    p->stream << "reduce(combiner="
              << op->combiner;
    p->stream << ", source=" << op->source;
    p->stream << ", axis=" << op->axis;
    p->stream << ", where=" << op->condition;
    p->stream << ", value_index=" << op->value_index;
    p->stream << ")";
  });

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<CommReducerNode>([](const ObjectRef& node, IRPrinter* p) {
    auto* op = static_cast<const CommReducerNode*>(node.get());
    p->stream << "comm_reducer(result=" << op->result
              << ", lhs=" << op->lhs
              << ", rhs=" << op->rhs
              << ", identity_element=" << op->identity_element
              << ")";
  });

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<Any>([](const ObjectRef& node, IRPrinter* p) {
    p->stream << "?";
});

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<IntrinsicOpNode>([](const ObjectRef& node, IRPrinter* p) {
    auto* op = static_cast<const IntrinsicOpNode*>(node.get());
    p->stream << "IntrinsicOp(" << op->name << ")";
  });


TVM_REGISTER_NODE_TYPE(CommReducerNode).set_rename_type_key("tir.CommReducerNode");
TVM_REGISTER_NODE_TYPE(Reduce).set_rename_type_key("tir.Reduce");
TVM_REGISTER_NODE_TYPE(Any).set_rename_type_key("tir.Any");
TVM_REGISTER_NODE_TYPE(AttrStmt).set_rename_type_key("tir.AttrStmt");
TVM_REGISTER_NODE_TYPE(FloatImm).set_rename_type_key("FloatImm");
TVM_REGISTER_NODE_TYPE(IntImm);
TVM_REGISTER_NODE_TYPE(UIntImm).set_rename_type_key("IntImm");
TVM_REGISTER_NODE_TYPE(StringImm).set_rename_type_key("tir.StringImm");
TVM_REGISTER_NODE_TYPE(Cast).set_rename_type_key("tir.Cast");
TVM_REGISTER_NODE_TYPE(Variable).set_rename_type_key("tir.Var");
TVM_REGISTER_NODE_TYPE(Add).set_rename_type_key("tir.Add");
TVM_REGISTER_NODE_TYPE(Sub).set_rename_type_key("tir.Sub");
TVM_REGISTER_NODE_TYPE(Mul).set_rename_type_key("tir.Mul");
TVM_REGISTER_NODE_TYPE(Div).set_rename_type_key("tir.Div");
TVM_REGISTER_NODE_TYPE(Mod).set_rename_type_key("tir.Mod");
TVM_REGISTER_NODE_TYPE(FloorDiv).set_rename_type_key("tir.FloorDiv");
TVM_REGISTER_NODE_TYPE(FloorMod).set_rename_type_key("tir.FloorMod");
TVM_REGISTER_NODE_TYPE(Min).set_rename_type_key("tir.Min");
TVM_REGISTER_NODE_TYPE(Max).set_rename_type_key("tir.Max");
TVM_REGISTER_NODE_TYPE(EQ).set_rename_type_key("tir.EQ");
TVM_REGISTER_NODE_TYPE(NE).set_rename_type_key("tir.NE");
TVM_REGISTER_NODE_TYPE(LT).set_rename_type_key("tir.LT");
TVM_REGISTER_NODE_TYPE(LE).set_rename_type_key("tir.LE");
TVM_REGISTER_NODE_TYPE(GT).set_rename_type_key("tir.GT");
TVM_REGISTER_NODE_TYPE(GE).set_rename_type_key("tir.GE");
TVM_REGISTER_NODE_TYPE(And).set_rename_type_key("tir.And");
TVM_REGISTER_NODE_TYPE(Or).set_rename_type_key("tir.Or");
TVM_REGISTER_NODE_TYPE(Not).set_rename_type_key("tir.Not");
TVM_REGISTER_NODE_TYPE(Select).set_rename_type_key("tir.Select");
TVM_REGISTER_NODE_TYPE(Load).set_rename_type_key("tir.Load");
TVM_REGISTER_NODE_TYPE(Ramp).set_rename_type_key("tir.Ramp");
TVM_REGISTER_NODE_TYPE(Broadcast).set_rename_type_key("tir.Broadcast");
TVM_REGISTER_NODE_TYPE(Shuffle).set_rename_type_key("tir.Shuffle");
TVM_REGISTER_NODE_TYPE(Prefetch).set_rename_type_key("tir.Prefetch");
TVM_REGISTER_NODE_TYPE(Call).set_rename_type_key("tir.Call");
TVM_REGISTER_NODE_TYPE(Let).set_rename_type_key("tir.Let");
TVM_REGISTER_NODE_TYPE(LetStmt).set_rename_type_key("tir.LetStmt");
TVM_REGISTER_NODE_TYPE(AssertStmt).set_rename_type_key("tir.AssertStmt");
TVM_REGISTER_NODE_TYPE(ProducerConsumer).set_rename_type_key("tir.ProducerConsumer");
TVM_REGISTER_NODE_TYPE(For).set_rename_type_key("tir.For");
TVM_REGISTER_NODE_TYPE(Store).set_rename_type_key("tir.Store");
TVM_REGISTER_NODE_TYPE(Provide).set_rename_type_key("tir.Provide");
TVM_REGISTER_NODE_TYPE(Allocate).set_rename_type_key("tir.Allocate");
TVM_REGISTER_NODE_TYPE(Free).set_rename_type_key("tir.Free");
TVM_REGISTER_NODE_TYPE(Realize).set_rename_type_key("tir.Realize");
TVM_REGISTER_NODE_TYPE(Block).set_rename_type_key("tir.SeqStmt");
TVM_REGISTER_NODE_TYPE(IfThenElse).set_rename_type_key("tir.IfThenElse");
TVM_REGISTER_NODE_TYPE(Evaluate).set_rename_type_key("tir.Evaluate");
TVM_REGISTER_NODE_TYPE(IntrinsicOpNode).set_rename_type_key("Op")
.set_repr_bytes([](const Object* n) -> std::string { return static_cast<const IntrinsicOpNode*>(n)->name; });

}  // namespace ir
}  // namespace air
