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

#ifndef CONTRIB_PARSER_AST_H_
#define CONTRIB_PARSER_AST_H_

#include <memory>
#include <iostream>
#include <list>
#include <utility>
#include <algorithm>
#include <string>
#include "token.h"

namespace akg {
namespace ir {
class ASTVisitor;

class ASTBase {
 public:
  ASTBase() = default;
  explicit ASTBase(const std::string &s) : baseType(s) {}
  virtual ~ASTBase() {}

  virtual void Accept(ASTVisitor &v) const = 0;
  std::string baseType;
};

class ASTExpr : public ASTBase {
 public:
  ASTExpr() = default;
  ~ASTExpr() override = default;
  using ASTBase::ASTBase;
};
class ASTStmt : public ASTBase {
 public:
  ASTStmt() = default;
  ~ASTStmt() override = default;
  using ASTBase::ASTBase;
};

using ASTBaseNode = std::shared_ptr<ASTBase>;
using ASTStmtNode = std::shared_ptr<ASTStmt>;
using ASTExprNode = std::shared_ptr<ASTExpr>;

template <typename T, typename... Args>
inline std::shared_ptr<T> ASTNode(Args &&... args) {
  return std::make_shared<T>(std::forward<Args>(args)...);
}

template <typename T>
inline std::shared_ptr<T> ASTNode() {
  return nullptr;
}

template <typename Type>
class ASTList : public std::list<Type> {
 public:
  template <typename T, typename... Args>
  T *push(Args &&... args) {
    T *ptr = new (std::nothrow) T(std::forward<Args>(args)...);
    CHECK(ptr);
    this->emplace_back(ptr);
    return ptr;
  }
};

class ASTStmtList : public ASTList<ASTStmtNode> {};
class ASTExprList : public ASTList<ASTExprNode> {};

class ASTLet : public ASTStmt {
 public:
  ASTLet(const std::string &var, const ASTExprNode &v, const ASTStmtList &l) : var(var), value(v), body(l) {}
  ~ASTLet() override = default;

  std::string var;
  ASTExprNode value;
  ASTStmtList body;

  void Accept(ASTVisitor &v) const final;
};

class ASTAttr : public ASTStmt {
 public:
  ASTAttr(const std::string &n, const std::string &k, const ASTExprNode v, const ASTStmtList &l)
      : node(n), attr_key(k), value(v), body(l) {}
  ~ASTAttr() override = default;

  std::string node;
  std::string attr_key;
  ASTExprNode value;
  ASTStmtList body;

  void Accept(ASTVisitor &v) const final;
};

class ASTAssert : public ASTStmt {
 public:
  ASTAssert(const ASTExprNode c, const ASTExprNode m, const ASTStmtList &l) : condition(c), message(m), body(l) {}
  ~ASTAssert() override = default;

  ASTExprNode condition, message;
  ASTStmtList body;

  void Accept(ASTVisitor &v) const final;
};

class ASTProduce : public ASTStmt {
 public:
  ASTProduce(const std::string &s, const ASTStmtList &l) : func(s), body(l) {}
  ~ASTProduce() override = default;

  std::string func;
  ASTStmtList body;

  void Accept(ASTVisitor &v) const final;
};

class ASTStore : public ASTStmt {
 public:
  ASTStore(const std::string &s, const ASTExprNode v, const ASTExprNode i, const ASTExprNode p)
      : buffer_var(s), value(v), index(i), predicate(p) {}
  ~ASTStore() override = default;

  std::string buffer_var;
  ASTExprNode value, index, predicate;

  void Accept(ASTVisitor &v) const final;
};

class ASTProvide : public ASTStmt {
 public:
  ASTProvide(const std::string &s, const ASTExprNode v, const ASTExprList &l) : func(s), value(v), args(l) {}
  ~ASTProvide() override = default;

  std::string func;
  ASTExprNode value;
  ASTExprList args;

  void Accept(ASTVisitor &v) const final;
};

class ASTAllocate : public ASTStmt {
 public:
  ASTAllocate(const std::string &s, ImmType t, unsigned b, const ASTExprList &l, const ASTStmtList &body)
      : buffer_var(s), type(t), bits(b), extents(l), body(body) {}
  ~ASTAllocate() override = default;

  std::string buffer_var;
  ImmType type;
  unsigned bits;
  ASTExprList extents;
  ASTStmtList body;

  void Accept(ASTVisitor &v) const final;
};

class ASTRealize : public ASTStmt {
 public:
  ASTRealize(const std::string &s, ImmType t, unsigned b, const ASTExprList &m, const ASTExprList &e,
             const ASTStmtList &l)
      : func(s), type(t), bits(b), bounds_min(m), bounds_ext(e), body(l) {}
  ~ASTRealize() override = default;

  std::string func;
  ImmType type;
  unsigned bits;
  ASTExprList bounds_min, bounds_ext;
  ASTStmtList body;

  void Accept(ASTVisitor &v) const final;
};

class ASTIfThenElse : public ASTStmt {
 public:
  ASTIfThenElse(const ASTExprNode c, const ASTStmtList &t, const ASTStmtList &e)
      : ASTStmt("if_then_else"), condition(c), then_case(t), else_case(e) {}
  ~ASTIfThenElse() override = default;

  ASTExprNode condition;
  ASTStmtList then_case, else_case;

  void Accept(ASTVisitor &v) const final;
};

class ASTEvaluate : public ASTStmt {
 public:
  explicit ASTEvaluate(const ASTExprNode n) : value(n) {}
  ~ASTEvaluate() override = default;

  ASTExprNode value;

  void Accept(ASTVisitor &v) const final;
};

class ASTFor : public ASTStmt {
 public:
  ASTFor(const std::string &s, const ASTExprNode m, const ASTExprNode e, const ASTStmtList &l)
      : loop_var(s), min(m), extent(e), body(l) {}
  ~ASTFor() override = default;
  std::string loop_var;
  ASTExprNode min, extent;
  ASTStmtList body;

  void Accept(ASTVisitor &v) const final;
};

class ASTIntImm : public ASTExpr {
 public:
  // The number 32 below  represents the bits of INT32
  explicit ASTIntImm(int64_t v, unsigned b = 32) : ASTExpr("Int"), value(v), bits(b) {}
  ~ASTIntImm() override = default;

  int64_t value;
  unsigned bits;

  void Accept(ASTVisitor &v) const final;
};

class ASTUIntImm : public ASTExpr {
 public:
  // The number 32 below represents the bits of INT32
  explicit ASTUIntImm(uint64_t v, unsigned b = 32) : ASTExpr("UInt"), value(v), bits(b) {}
  ~ASTUIntImm() override = default;

  uint64_t value;
  unsigned bits;

  void Accept(ASTVisitor &v) const final;
};

class ASTFloatImm : public ASTExpr {
 public:
  // The number 32 below  marks the bits of INT32
  explicit ASTFloatImm(double v, unsigned b = 32) : value(v), bits(b) {}
  ~ASTFloatImm() override = default;

  double value;
  unsigned bits;

  void Accept(ASTVisitor &v) const final;
};

class ASTStringImm : public ASTExpr {
 public:
  explicit ASTStringImm(const std::string &s) : value(s) {}
  ~ASTStringImm() override = default;

  std::string value;

  void Accept(ASTVisitor &v) const final;
};

class ASTCast : public ASTExpr {
 public:
  ASTCast(ImmType t, unsigned b, const ASTExprNode e) : type(t), bits(b), value(e) {}
  ~ASTCast() override = default;

  ImmType type;
  unsigned bits;
  ASTExprNode value;

  void Accept(ASTVisitor &v) const final;
};

class ASTNot : public ASTExpr {
 public:
  explicit ASTNot(const ASTExprNode a) : a(a) {}
  ~ASTNot() override = default;

  ASTExprNode a;

  void Accept(ASTVisitor &v) const final;
};

class ASTBinaryOp : public ASTExpr {
 public:
  ASTBinaryOp(Token op, const ASTExprNode a, const ASTExprNode b) : op(op), a(a), b(b) {}
  ~ASTBinaryOp() override = default;

  Token op;
  ASTExprNode a, b;

  void Accept(ASTVisitor &v) const final;
};

class ASTSelect : public ASTExpr {
 public:
  ASTSelect(const ASTExprNode c, const ASTExprNode t, const ASTExprNode f)
      : condition(c), true_value(t), false_value(f) {}
  ~ASTSelect() override = default;

  ASTExprNode condition, true_value, false_value;

  void Accept(ASTVisitor &v) const final;
};

class ASTLoad : public ASTExpr {
 public:
  ASTLoad(const std::string &s, const ASTExprNode i, const ASTExprNode p) : buffer_var(s), index(i), predicate(p) {}
  ~ASTLoad() override = default;

  std::string buffer_var;
  ASTExprNode index, predicate;

  void Accept(ASTVisitor &v) const final;
};

class ASTLetExpr : public ASTExpr {
 public:
  ASTLetExpr(const std::string &var, const ASTExprNode &v, const ASTExprNode b) : var(var), value(v), body(b) {}
  ~ASTLetExpr() override = default;

  std::string var;
  ASTExprNode value, body;

  void Accept(ASTVisitor &v) const final;
};

class ASTCall : public ASTExpr {
 public:
  ASTCall(const std::string &s, ImmType t, unsigned b, Token tok, const ASTExprList &l)
      : name(s), type(t), bits(b), call_type(tok), args(l) {}
  ~ASTCall() override = default;

  std::string name;
  ImmType type;
  unsigned bits;
  Token call_type;
  ASTExprList args;

  void Accept(ASTVisitor &v) const final;
};

class ASTVariable : public ASTExpr {
 public:
  explicit ASTVariable(const std::string &s) : name_hint(s) {}
  ~ASTVariable() override = default;

  std::string name_hint;

  void Accept(ASTVisitor &v) const final;
};

class ASTVisitor {
 public:
  virtual ~ASTVisitor() {}
  virtual void Visit(const ASTLet &) = 0;
  virtual void Visit(const ASTAttr &) = 0;
  virtual void Visit(const ASTAssert &) = 0;
  virtual void Visit(const ASTProduce &) = 0;
  virtual void Visit(const ASTStore &) = 0;
  virtual void Visit(const ASTProvide &) = 0;
  virtual void Visit(const ASTAllocate &) = 0;
  virtual void Visit(const ASTRealize &) = 0;
  virtual void Visit(const ASTIfThenElse &) = 0;
  virtual void Visit(const ASTEvaluate &) = 0;
  virtual void Visit(const ASTFor &) = 0;
  virtual void Visit(const ASTIntImm &) = 0;
  virtual void Visit(const ASTUIntImm &) = 0;
  virtual void Visit(const ASTFloatImm &) = 0;
  virtual void Visit(const ASTStringImm &) = 0;
  virtual void Visit(const ASTCast &) = 0;
  virtual void Visit(const ASTNot &) = 0;
  virtual void Visit(const ASTBinaryOp &) = 0;
  virtual void Visit(const ASTSelect &) = 0;
  virtual void Visit(const ASTLoad &) = 0;
  virtual void Visit(const ASTLetExpr &) = 0;
  virtual void Visit(const ASTCall &) = 0;
  virtual void Visit(const ASTVariable &) = 0;
};

void PrintAST(const ASTStmtList &l, std::ostream &os);
}  // namespace ir
}  // namespace akg

#endif  // CONTRIB_PARSER_AST_H_
