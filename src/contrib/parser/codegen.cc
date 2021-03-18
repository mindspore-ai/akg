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

#include "contrib/parser/codegen.h"

#include <tvm/ir.h>
#include <tvm/operation.h>

#include <list>
#include <vector>

#include "emit_insn/insn_info.h"
#include "contrib/parser/ast.h"

namespace akg {
namespace ir {
namespace {
class ASTCodeGenerator;
Stmt MakeBlock(const ASTStmtList &list, ASTCodeGenerator &gen);

class ASTCodeGenerator : public ASTVisitor {
 public:
  explicit ASTCodeGenerator(const Map<Tensor, Buffer> &ori_in) : stmt(), expr(), normal_var_(), buffer_collector_() {
    for (auto e : ori_in) {
      PushBuffer(e.second->name, e.first->op, e.second->data);
    }
  }
  ~ASTCodeGenerator() override = default;
  void Visit(const ASTLet &op) final { CHECK(false); }

  void Visit(const ASTAttr &op) final {
    static const std::set<std::string> omitPragma = {
      "buffer_bind_scope",
      "extern_scope",
    };
    op.value->Accept(*this);
    auto value = expr;

    auto body = MakeBlock(op.body, *this);
    NodeRef ref;
    if (op.attr_key == air::ir::attr::realize_scope) {
      auto stat = GetTokStateFromCode(op.node);
      auto tok = GetNextToken(stat);
      CHECK(tok == Token::kID);

      if (stat.sval == "placeholder") {
        tok = GetNextToken(stat);
        CHECK(tok == Token::kLPAR);
        tok = GetNextToken(stat);
        CHECK(tok == Token::kID);

        ref = GetBuffer(stat.sval).first;
        CHECK(ref->IsInstance<PlaceholderOpNode>());
        PopBuffer(stat.sval);
      } else if (stat.sval == "compute") {
        ref = StringImm::make(op.node);
      } else if (stat.sval == "hybrid") {
        stmt = body;
        return;
      } else {
        CHECK(false);
      }
    } else if (op.attr_key == air::ir::attr::storage_scope) {
      ref = GetBuffer(op.node).second;
      PopBuffer(op.node);
    } else if (op.attr_key == air::ir::attr::coproc_scope) {
      ref = GetCceAxis();
    } else if (op.attr_key == "isolate_range") {
      CHECK_EQ(op.node, "0");
      ref = Expr(0);
    } else if (omitPragma.count(op.attr_key) != 0) {
      stmt = body;
      return;
    } else {
      CHECK(false);
    }

    stmt = AttrStmt::make(ref, op.attr_key, value, body);
  }

  void Visit(const ASTAssert &op) final {
    op.condition->Accept(*this);
    auto cond = expr;

    op.message->Accept(*this);
    auto msg = expr;

    auto body = MakeBlock(op.body, *this);
    stmt = AssertStmt::make(cond, msg, body);
  }

  void Visit(const ASTProduce &op) final {
    auto body = MakeBlock(op.body, *this);
    auto hdr = GetBuffer(op.func).first;

    stmt = ProducerConsumer::make(hdr, true, body);
  }

  void Visit(const ASTStore &op) final {
    op.value->Accept(*this);
    auto value = expr;

    op.index->Accept(*this);
    auto index = expr;

    op.predicate->Accept(*this);
    auto pred = expr;

    stmt = Store::make(GetBuffer(op.buffer_var).second, value, index, pred);
  }

  void Visit(const ASTProvide &op) final {
    op.value->Accept(*this);
    auto value = expr;

    Array<Expr> array;
    for (auto e : op.args) {
      e->Accept(*this);
      array.push_back(expr);
    }

    stmt = Provide::make(GetBuffer(op.func).first, 0, value, array);
  }

  void Visit(const ASTAllocate &op) final {
    auto t = GenType(op.type, op.bits);
    auto var = Var(op.buffer_var, t);

    auto hdr = PlaceholderOpNode::make(op.buffer_var, Array<Expr>(), t);
    PushBuffer(op.buffer_var, hdr, var);

    Array<Expr> ext;
    for (auto e : op.extents) {
      e->Accept(*this);
      ext.push_back(expr);
    }

    auto body = MakeBlock(op.body, *this);

    stmt = Allocate::make(var, t, ext, const_true(), body);
  }

  void Visit(const ASTRealize &op) final {
    auto t = GenType(op.type, op.bits);
    auto hdr = PlaceholderOpNode::make(op.func, Array<Expr>(), t);
    PushBuffer(op.func, hdr, Var(op.func));

    CHECK_EQ(op.bounds_min.size(), op.bounds_ext.size());
    Region bounds;
    for (auto it0 = op.bounds_min.cbegin(), it1 = op.bounds_ext.cbegin(); it0 != op.bounds_min.cend(); ++it0, ++it1) {
      (*it0)->Accept(*this);
      auto min = expr;

      (*it1)->Accept(*this);
      auto ext = expr;

      bounds.push_back(Range::make_by_min_extent(min, ext));
    }

    auto body = MakeBlock(op.body, *this);

    stmt = Realize::make(hdr, 0, t, bounds, const_true(), body);
  }

  void Visit(const ASTIfThenElse &op) final {
    op.condition->Accept(*this);
    auto cond = expr;

    auto then_case = MakeBlock(op.then_case, *this);
    auto else_case = op.else_case.empty() ? Stmt() : MakeBlock(op.else_case, *this);

    stmt = IfThenElse::make(cond, then_case, else_case);
  }

  void Visit(const ASTEvaluate &op) final {
    op.value->Accept(*this);
    stmt = Evaluate::make(expr);
  }

  void Visit(const ASTFor &op) final {
    auto var = PushVar(op.loop_var);

    op.min->Accept(*this);
    auto min = expr;

    op.extent->Accept(*this);
    auto ext = expr;

    auto body = MakeBlock(op.body, *this);

    PopVar(op.loop_var);
    stmt = For::make(var, min, ext, ForType::Serial, DeviceAPI::None, body);
  }

  void Visit(const ASTIntImm &op) final { expr = IntImm::make(Int(static_cast<int>(op.bits)), op.value); }

  void Visit(const ASTUIntImm &op) final { expr = UIntImm::make(UInt(static_cast<int>(op.bits)), op.value); }

  void Visit(const ASTFloatImm &op) final { expr = FloatImm::make(Float(static_cast<int>(op.bits)), op.value); }

  void Visit(const ASTStringImm &op) final { expr = StringImm::make(op.value); }

  void Visit(const ASTCast &op) final {
    auto t = GenType(op.type, op.bits);
    op.value->Accept(*this);

    expr = Cast::make(t, expr);
  }

  void Visit(const ASTNot &op) final {
    op.a->Accept(*this);
    expr = Not::make(expr);
  }

  void Visit(const ASTBinaryOp &op) final {
    op.a->Accept(*this);
    auto a = expr;

    op.b->Accept(*this);
    auto b = expr;

    switch (op.op) {
#define GEN_EXPR(tok, ir)  \
  case Token::k##tok:      \
    expr = ir::make(a, b); \
    break
      GEN_EXPR(MAX, Max);
      GEN_EXPR(MIN, Min);
      GEN_EXPR(PLUS, Add);
      GEN_EXPR(MINUS, Sub);
      GEN_EXPR(STAR, Mul);
      GEN_EXPR(SLASH, Div);
      GEN_EXPR(PERCENT, Mod);
      GEN_EXPR(LESS, LT);
      GEN_EXPR(GREATER, GT);
      GEN_EXPR(EQEQUAL, EQ);
      GEN_EXPR(NOTEQUAL, NE);
      GEN_EXPR(AND, And);
      GEN_EXPR(OR, Or);
      GEN_EXPR(GREATEREQUAL, GE);
      GEN_EXPR(LESSEQUAL, LE);
#undef GEN_EXPR
      default:
        CHECK(false);
    }
  }

  void Visit(const ASTSelect &op) final {
    op.condition->Accept(*this);
    auto cond = expr;

    op.true_value->Accept(*this);
    auto true_value = expr;

    op.false_value->Accept(*this);
    auto false_value = expr;

    expr = Select::make(cond, true_value, false_value);
  }

  void Visit(const ASTLoad &op) final {
    op.index->Accept(*this);
    auto index = expr;

    op.predicate->Accept(*this);
    auto pred = expr;

    auto buf = GetBuffer(op.buffer_var);

    CHECK(buf.first.as<PlaceholderOpNode>() != nullptr);
    expr = Load::make(buf.first.as<PlaceholderOpNode>()->dtype, buf.second, index, pred);
  }

  void Visit(const ASTLetExpr &op) final { CHECK(false); }

  void Visit(const ASTCall &op) final {
    Array<Expr> array;
    for (auto e : op.args) {
      e->Accept(*this);
      array.push_back(expr);
    }

    if (op.call_type == Token::kHALIDE) {
      auto func = GetBuffer(op.name).first;
      auto ptr = func.as<PlaceholderOpNode>();
      CHECK(ptr != nullptr);
      expr = Call::make(ptr->dtype, op.name, array, GetCallType(op.call_type), func, 0);
    } else {
      expr = Call::make(GenType(op.type, op.bits), op.name, array, GetCallType(op.call_type));
    }
  }

  void Visit(const ASTVariable &op) final {
    if (normal_var_.find(op.name_hint) != normal_var_.end()) {
      expr = GetVar(op.name_hint);
    } else {
      expr = GetBuffer(op.name_hint).second;
    }
  }

  Stmt stmt;
  Expr expr;

 private:
  Var PushVar(const std::string &s, Type t = Int(32)) {
    auto &list = normal_var_[s];
    list.emplace_back(s, t);
    return list.back();
  }
  void PopVar(const std::string &s) {
    auto it = normal_var_.find(s);
    CHECK(it != normal_var_.end());

    it->second.pop_back();
    if (it->second.empty()) {
      normal_var_.erase(it);
    }
  }
  Var GetVar(const std::string &s) {
    auto it = normal_var_.find(s);
    CHECK(it != normal_var_.end());
    CHECK(!it->second.empty());

    return it->second.back();
  }

  void PushBuffer(const std::string &s, const FunctionRef &f, const Var &v) { buffer_collector_[s].emplace_back(f, v); }

  void PopBuffer(const std::string &s) {
    auto it = buffer_collector_.find(s);
    CHECK(it != buffer_collector_.end());

    it->second.pop_back();
    if (it->second.empty()) {
      buffer_collector_.erase(it);
    }
  }

  std::pair<FunctionRef, Var> GetBuffer(const std::string &s) {
    auto it = buffer_collector_.find(s);
    CHECK(it != buffer_collector_.end()) << "Symbol not found: " << s;
    return it->second.back();
  }

  Type GenType(ImmType t, unsigned b) {
    switch (t) {
      case ImmType::kINT:
        return Int(static_cast<int>(b));
      case ImmType::kUINT:
        return UInt(static_cast<int>(b));
      case ImmType::kFLOAT:
        return Float(static_cast<int>(b));
      case ImmType::kHANDLE:
        return Handle();
      default:
        CHECK(false);
    }
    return Int(32);
  }

  Call::CallType GetCallType(Token t) {
    switch (t) {
      case Token::kEXTERN:
        return Call::Extern;
      case Token::kEXTERNCPP:
        return Call::ExternCPlusPlus;
      case Token::kPUREEXTERN:
        return Call::PureExtern;
      case Token::kHALIDE:
        return Call::Halide;
      case Token::kINTRINSIC:
        return Call::Intrinsic;
      case Token::kPUREINTRIN:
        return Call::PureIntrinsic;
      default:
        CHECK(false);
    }
    return Call::Extern;
  }

  std::map<std::string, std::list<Var>> normal_var_;
  std::map<std::string, std::list<std::pair<FunctionRef, Var>>> buffer_collector_;
};

Stmt MakeBlock(const ASTStmtList &list, ASTCodeGenerator &cg) {
  CHECK(!list.empty());

  std::vector<Stmt> v;
  for (auto e : list) {
    e->Accept(cg);
    CHECK(cg.stmt.defined());
    v.push_back(cg.stmt);
  }
  return Block::make(v);
}
}  // namespace

Stmt GenHalideIR(const ASTStmtList &list, const Map<Tensor, Buffer> &in) {
  ASTCodeGenerator gen(in);
  return MakeBlock(list, gen);
}
}  // namespace ir
}  // namespace akg
