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

#include "contrib/parser/ast.h"

#include <dmlc/logging.h>

#include <map>
#include <type_traits>
namespace akg {
namespace ir {
void ASTLet::Accept(ASTVisitor &v) const { v.Visit(*this); }

void ASTAttr::Accept(ASTVisitor &v) const { v.Visit(*this); }

void ASTAssert::Accept(ASTVisitor &v) const { v.Visit(*this); }

void ASTProduce::Accept(ASTVisitor &v) const { v.Visit(*this); }

void ASTStore::Accept(ASTVisitor &v) const { v.Visit(*this); }

void ASTProvide::Accept(ASTVisitor &v) const { v.Visit(*this); }

void ASTAllocate::Accept(ASTVisitor &v) const { v.Visit(*this); }

void ASTRealize::Accept(ASTVisitor &v) const { v.Visit(*this); }

void ASTIfThenElse::Accept(ASTVisitor &v) const { v.Visit(*this); }

void ASTEvaluate::Accept(ASTVisitor &v) const { v.Visit(*this); }

void ASTFor::Accept(ASTVisitor &v) const { v.Visit(*this); }

void ASTIntImm::Accept(ASTVisitor &v) const { v.Visit(*this); }

void ASTUIntImm::Accept(ASTVisitor &v) const { v.Visit(*this); }

void ASTFloatImm::Accept(ASTVisitor &v) const { v.Visit(*this); }

void ASTStringImm::Accept(ASTVisitor &v) const { v.Visit(*this); }

void ASTCast::Accept(ASTVisitor &v) const { v.Visit(*this); }

void ASTBinaryOp::Accept(ASTVisitor &v) const { v.Visit(*this); }

void ASTNot::Accept(ASTVisitor &v) const { v.Visit(*this); }

void ASTSelect::Accept(ASTVisitor &v) const { v.Visit(*this); }

void ASTLoad::Accept(ASTVisitor &v) const { v.Visit(*this); }

void ASTLetExpr::Accept(ASTVisitor &v) const { v.Visit(*this); }

void ASTCall::Accept(ASTVisitor &v) const { v.Visit(*this); }

void ASTVariable::Accept(ASTVisitor &v) const { v.Visit(*this); }

namespace {
class ASTPrinter : public ASTVisitor {
 public:
  explicit ASTPrinter(std::ostream &os) : os_(os), indent_(0) {}
  ~ASTPrinter() override = default;
  void Visit(const ASTLet &op) final {
    PrintIndent();
    os_ << "let " << op.var << " = ";
    op.value->Accept(*this);
    os_ << '\n';
    indent_ += 1;
    PrintStmtList(op.body);
    indent_ -= 1;
  }
  void Visit(const ASTAttr &op) final {
    PrintIndent();
    os_ << "// attr [" << op.node;
    os_ << "] " << op.attr_key << " = ";
    op.value->Accept(*this);
    os_ << '\n';
    indent_ += 1;
    PrintStmtList(op.body);
    indent_ -= 1;
  }
  void Visit(const ASTAssert &op) final {
    PrintIndent();
    os_ << "assert(";
    op.condition->Accept(*this);
    os_ << ", ";
    op.message->Accept(*this);
    os_ << ")\n";
    indent_ += 1;
    PrintStmtList(op.body);
    indent_ -= 1;
  }
  void Visit(const ASTProduce &op) final {
    PrintIndent();
    os_ << "produce " << op.func << " {\n";
    indent_ += 2;
    PrintStmtList(op.body);
    indent_ -= 2;
    PrintIndent();
    os_ << "}\n";
  }
  void Visit(const ASTStore &op) final {
    PrintIndent();
    os_ << op.buffer_var << "[";
    op.index->Accept(*this);
    os_ << "] = ";
    op.value->Accept(*this);
    if (!IsOne(op.predicate)) {
      os_ << " if ";
      op.predicate->Accept(*this);
    }
    os_ << '\n';
  }
  void Visit(const ASTProvide &op) final {
    PrintIndent();
    os_ << op.func;
    os_ << "(";
    for (auto it = op.args.cbegin(); it != op.args.cend(); ++it) {
      (*it)->Accept(*this);
      if (std::next(it) != op.args.cend()) os_ << ", ";
    }
    os_ << ") = ";
    op.value->Accept(*this);
    os_ << '\n';
  }
  void Visit(const ASTAllocate &op) final {
    PrintIndent();
    os_ << "allocate " << op.buffer_var << "[";
    PrintType(op.type);
    os_ << op.bits;
    for (auto it = op.extents.cbegin(); it != op.extents.cend(); ++it) {
      os_ << " * ";
      (*it)->Accept(*this);
    }
    os_ << "]\n";
    indent_ += 1;
    PrintStmtList(op.body);
    indent_ -= 1;
  }
  void Visit(const ASTRealize &op) final {
    PrintIndent();
    os_ << "realize " << op.func << "<";
    PrintType(op.type);
    os_ << op.bits << ">(";
    CHECK_EQ(op.bounds_min.size(), op.bounds_ext.size());
    for (auto it0 = op.bounds_min.cbegin(), it1 = op.bounds_ext.cbegin(); it0 != op.bounds_min.cend(); ++it0, ++it1) {
      os_ << "[";
      (*it0)->Accept(*this);
      os_ << ", ";
      (*it1)->Accept(*this);
      os_ << "]";
      if (std::next(it0) != op.bounds_min.cend()) os_ << ", ";
    }
    os_ << ") {\n";

    indent_ += 2;
    PrintStmtList(op.body);
    indent_ -= 2;

    PrintIndent();
    os_ << "}\n";
  }
  void Visit(const ASTIfThenElse &op) final {
    PrintIndent();
    const auto *ptr = &op;
    while (1) {
      os_ << "if (";
      ptr->condition->Accept(*this);
      os_ << ") {\n";
      indent_ += 2;
      PrintStmtList(ptr->then_case);
      indent_ -= 2;

      if (ptr->else_case.empty()) {
        break;
      }

      if (ptr->else_case.size() == 1) {
        if (ptr->else_case.front()->baseType == "if_then_else") {
          PrintIndent();
          os_ << "} else ";
          ptr = static_cast<ASTIfThenElse *>(ptr->else_case.front().get());
          continue;
        }
      }

      PrintIndent();
      os_ << "} else {\n";
      indent_ += 2;
      PrintStmtList(ptr->else_case);
      indent_ -= 2;
      break;
    }

    PrintIndent();
    os_ << "}\n";
  }
  void Visit(const ASTEvaluate &op) final {
    PrintIndent();
    op.value->Accept(*this);
    os_ << "\n";
  }
  void Visit(const ASTFor &op) final {
    PrintIndent();
    os_ << "for (" << op.loop_var << ", ";
    op.min->Accept(*this);
    os_ << ", ";
    op.extent->Accept(*this);
    os_ << ") {\n";

    indent_ += 2;
    PrintStmtList(op.body);
    indent_ -= 2;

    PrintIndent();
    os_ << "}\n";
  }
  void Visit(const ASTIntImm &op) final {
    if (op.bits == 32) {
      os_ << op.value;
    } else {
      os_ << "(";
      PrintType(ImmType::kINT);
      os_ << op.bits << ")" << op.value;
    }
  }
  void Visit(const ASTUIntImm &op) final {
    os_ << "(";
    PrintType(ImmType::kUINT);
    os_ << op.bits << ")" << op.value;
  }
  void Visit(const ASTFloatImm &op) final {
    os_ << op.value;
    switch (op.bits) {
      case 64:
        break;
      case 32:
        os_ << 'f';
        break;
      case 16:
        os_ << 'h';
        break;
      default:
        CHECK(false) << "Bad bit-width for float:"
                     << "\n";
    }
  }
  void Visit(const ASTStringImm &op) final { os_ << '"' << op.value << '"'; }
  void Visit(const ASTCast &op) final {
    PrintType(op.type);
    os_ << op.bits << '(';
    op.value->Accept(*this);
    os_ << ')';
  }
  void Visit(const ASTNot &op) final {
    os_ << '!';
    op.a->Accept(*this);
  }
  void Visit(const ASTBinaryOp &op) final {
    os_ << '(';
    op.a->Accept(*this);
    auto it = tok2sym_.find(op.op);
    CHECK(it != tok2sym_.end());
    os_ << " " << it->second << " ";
    op.b->Accept(*this);
    os_ << ')';
  }
  void Visit(const ASTSelect &op) final {
    os_ << "select(";
    op.condition->Accept(*this);
    os_ << ", ";
    op.true_value->Accept(*this);
    os_ << ", ";
    op.false_value->Accept(*this);
    os_ << ")";
  }
  void Visit(const ASTLoad &op) final {
    os_ << op.buffer_var << "[";
    op.index->Accept(*this);
    os_ << "]";
    if (!IsOne(op.predicate)) {
      os_ << " if ";
      op.predicate->Accept(*this);
    }
  }
  void Visit(const ASTLetExpr &op) final {
    os_ << "(let " << op.var << " = ";
    op.value->Accept(*this);
    os_ << " in ";
    op.body->Accept(*this);
    os_ << ")";
  }
  void Visit(const ASTCall &op) final {
    os_ << op.name;
    for (auto it = op.args.cbegin(); it != op.args.cend(); ++it) {
      (*it)->Accept(*this);
      if (std::next(it) != op.args.cend()) os_ << ", ";
    }
    os_ << ")";
    if (op.call_type != Token::kHALIDE) {
      os_ << ":";
      PrintType(op.type);
      auto it = tok2sym_.find(op.call_type);
      CHECK(it != tok2sym_.end());
      os_ << op.bits << ":" << it->second;
    }
  }
  void Visit(const ASTVariable &op) final { os_ << op.name_hint; }

 private:
  void PrintIndent() {
    for (unsigned i = 0; i != indent_; ++i) {
      os_ << ' ';
    }
  }
  void PrintStmtList(const ASTStmtList &l) {
    for (auto e : l) {
      e->Accept(*this);
    }
  }
  bool IsOne(const ASTExprNode n) const {
    if (n->baseType == "Int") {
      return static_cast<ASTIntImm *>(n.get())->value == 1;
    }
    if (n->baseType == "UInt") {
      return static_cast<ASTUIntImm *>(n.get())->value == 1;
    }
    return false;
  }
  void PrintType(ImmType t) {
    switch (t) {
      case ImmType::kINT:
        os_ << "int";
        break;
      case ImmType::kUINT:
        os_ << "uint";
        break;
      case ImmType::kFLOAT:
        os_ << "float";
        break;
      default:
        CHECK(false);
    }
  }

  std::ostream &os_;
  unsigned indent_;

  struct CmpToken {
    bool operator()(const Token &a, const Token &b) const {
      return static_cast<typename std::underlying_type<Token>::type>(a) <
             static_cast<typename std::underlying_type<Token>::type>(b);
    }
  };
  static const std::map<Token, std::string, CmpToken> tok2sym_;
};

const std::map<Token, std::string, ASTPrinter::CmpToken> ASTPrinter::tok2sym_ = {
#define TOKEN(name, sym) {Token::k##name, std::string(sym)},
#include "meta_token.md"
#undef TOKEN

#define KEY(name, sym) {Token::k##name, std::string(sym)},
#include "key_word.md"
#undef KEY
};
}  // namespace

void PrintAST(const ASTStmtList &l, std::ostream &os) {
  for (auto e : l) {
    ASTPrinter p(os);
    e->Accept(p);
  }
}
}  // namespace ir
}  // namespace akg
