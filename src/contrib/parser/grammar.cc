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

#include "contrib/parser/grammar.h"

#include <dmlc/logging.h>

#include <queue>
#include <string>
#include <iomanip>
#include <map>

#include "contrib/parser/token.h"

namespace akg {
namespace ir {
namespace {
struct TokInfo {
  Token tok;
  ImmType itype;
  unsigned bits;
  double fval;
  uint64_t uval;
  std::string sval;
};

class TokHandler : public std::queue<TokInfo> {
 public:
  explicit TokHandler(TokState &stat) : std::queue<TokInfo>(), stat_(stat) {}
  ~TokHandler() {}
  TokState &GetTokState() const { return stat_; }

  Token LookNextTok() {
    auto tok = GetNextTok();
    Push(stat_);
    return tok;
  }

  TokInfo Pop() {
    if (empty()) {
      static_cast<void>(LookNextTok());
    }
    auto tok = front();
    pop();
    return tok;
  }

  void Push(const TokState &s) { push(TokInfo{s.tok, s.itype, s.bits, s.fval, s.uval, s.sval}); }

  Token FrontTok() { return empty() ? LookNextTok() : front().tok; }

  TokState &stat_;

 private:
  Token GetNextTok() { return GetNextToken(stat_); }
};

// forward declaration
ASTExprNode ParseExpr(TokHandler &h);

ASTExprList ParseParamList(TokHandler &h);

ASTStmtNode ParseStmt(TokHandler &h);

ASTStmtList ParseStmtList(TokHandler &h);

#define CHECK_AND_POP(hdl, tk) \
  {                            \
    auto _t = hdl.Pop().tok;   \
    CHECK(_t == tk);           \
  }

// definition
ASTExprNode ParseCast(TokHandler &h) {
  auto type_info = h.Pop();
  CHECK(type_info.tok == Token::kTYPE);

  CHECK_AND_POP(h, Token::kLPAR);

  auto expr = ParseExpr(h);
  CHECK(expr);

  CHECK_AND_POP(h, Token::kRPAR);

  return ASTNode<ASTCast>(type_info.itype, type_info.bits, expr);
}

ASTExprNode ParseSelect(TokHandler &h) {
  CHECK_AND_POP(h, Token::kSELECT);

  CHECK_AND_POP(h, Token::kLPAR);

  auto cond = ParseExpr(h);
  CHECK(cond);

  CHECK_AND_POP(h, Token::kCOMMA);

  auto true_value = ParseExpr(h);
  CHECK(true_value);

  CHECK_AND_POP(h, Token::kCOMMA);

  auto false_value = ParseExpr(h);
  CHECK(false_value);

  CHECK_AND_POP(h, Token::kRPAR);

  return ASTNode<ASTSelect>(cond, true_value, false_value);
}

ASTExprNode ParseMaxMin(TokHandler &h) {
  auto tok = h.Pop().tok;
  CHECK(tok == Token::kMAX || tok == Token::kMIN);

  CHECK_AND_POP(h, Token::kLPAR);

  auto a = ParseExpr(h);
  CHECK(a);

  CHECK_AND_POP(h, Token::kCOMMA);

  auto b = ParseExpr(h);
  CHECK(b);

  CHECK_AND_POP(h, Token::kRPAR);

  return ASTNode<ASTBinaryOp>(tok, a, b);
}

ASTExprNode ParseLetExpr(TokHandler &h) {
  CHECK_AND_POP(h, Token::kLPAR);
  CHECK_AND_POP(h, Token::kLET);

  auto id = h.Pop();
  CHECK(id.tok == Token::kID);

  CHECK_AND_POP(h, Token::kEQUAL);

  auto value = ParseExpr(h);
  CHECK(value);

  CHECK_AND_POP(h, Token::kIN);

  auto body = ParseExpr(h);
  CHECK(body);

  CHECK_AND_POP(h, Token::kRPAR);

  return ASTNode<ASTLetExpr>(id.sval, value, body);
}

ASTExprNode ParseNot(TokHandler &h) {
  CHECK_AND_POP(h, Token::kNOT);

  auto a = ParseExpr(h);
  CHECK(a);

  return ASTNode<ASTNot>(a);
}

ASTExprNode ParseBinExpr(TokHandler &h) {
  CHECK_AND_POP(h, Token::kLPAR);

  auto a = ParseExpr(h);
  CHECK(a);

  auto tok = h.Pop().tok;
  CHECK(tok == Token::kEQEQUAL || tok == Token::kNOTEQUAL || tok == Token::kAND || tok == Token::kGREATEREQUAL ||
        tok == Token::kLESSEQUAL || tok == Token::kOR || tok == Token::kGREATER || tok == Token::kPLUS ||
        tok == Token::kMINUS || tok == Token::kSTAR || tok == Token::kSLASH || tok == Token::kPERCENT ||
        tok == Token::kLESS);

  auto b = ParseExpr(h);
  CHECK(b);

  CHECK_AND_POP(h, Token::kRPAR);

  return ASTNode<ASTBinaryOp>(tok, a, b);
}

inline std::pair<TokInfo, TokInfo> ParseCallTail(TokHandler &h) {
  TokInfo t = {Token::kTYPE, ImmType::kINT, 0, 0.0d, 0, ""};
  TokInfo call_t = {Token::kHALIDE, ImmType::kINT, 0, 0.0d, 0, ""};

  if (h.FrontTok() == Token::kCOLON) {
    h.pop();

    t = h.Pop();
    CHECK(t.tok == Token::kTYPE);

    CHECK_AND_POP(h, Token::kCOLON);

    call_t = h.Pop();
  }

  CHECK(call_t.tok == Token::kEXTERN || call_t.tok == Token::kEXTERNCPP || call_t.tok == Token::kPUREEXTERN ||
        call_t.tok == Token::kHALIDE || call_t.tok == Token::kINTRINSIC || call_t.tok == Token::kPUREINTRIN);
  return std::make_pair(t, call_t);
}

ASTExprNode ParseCall(TokHandler &h) {
  CHECK_LE(h.size(), 2);

  auto id = h.Pop();
  CHECK(id.tok == Token::kID);

  CHECK_AND_POP(h, Token::kLPAR);

  ASTExprList list;
  if (h.LookNextTok() != Token::kRPAR) {
    list = ParseParamList(h);
  }

  CHECK_AND_POP(h, Token::kRPAR);

  auto pair = ParseCallTail(h);
  auto &t = pair.first;
  auto &call_t = pair.second;

  return ASTNode<ASTCall>(id.sval, t.itype, t.bits, call_t.tok, list);
}

ASTExprNode ParseImmNum(TokHandler &h) {
  ImmType itype = ImmType::kINT;
  unsigned bits = 32;
  bool sign = true;

  auto cur = h.Pop();
  if (cur.tok == Token::kLPAR) {
    cur = h.Pop();
    CHECK(cur.tok == Token::kTYPE);

    itype = cur.itype;
    bits = cur.bits;

    CHECK_AND_POP(h, Token::kRPAR);
    cur = h.Pop();
  }

  while (cur.tok == Token::kMINUS) {
    cur = h.Pop();
    sign = !sign;
  }

  CHECK(cur.tok == Token::kNUMBER);

  if (cur.itype == ImmType::kFLOAT) {
    if (!sign) {
      cur.fval = -cur.fval;
    }
    return ASTNode<ASTFloatImm>(cur.fval, cur.bits);
  }

  if (itype == ImmType::kUINT) {
    CHECK(sign);
    return ASTNode<ASTUIntImm>(cur.uval, bits);
  }

  CHECK(itype == ImmType::kINT);
  auto val = static_cast<int64_t>(cur.uval);
  if (!sign) {
    val = -val;
  }
  return ASTNode<ASTIntImm>(val, bits);
}

ASTExprNode ParseLoad(TokHandler &h) {
  auto id = h.Pop();
  CHECK(id.tok == Token::kID);

  CHECK_AND_POP(h, Token::kLSQB);

  auto index = ParseExpr(h);
  CHECK(index);

  CHECK_AND_POP(h, Token::kRSQB);

  auto pred = ASTNode<ASTExpr>();
  if (h.LookNextTok() == Token::kIF) {
    h.pop();
    pred = ParseImmNum(h);
  } else {
    pred = ASTNode<ASTIntImm>(1);
  }

  return ASTNode<ASTLoad>(id.sval, index, pred);
}

ASTExprNode ParseExpr(TokHandler &h) {
  switch (h.FrontTok()) {
    case Token::kTYPE:
      return ParseCast(h);
    case Token::kSELECT:
      return ParseSelect(h);
    case Token::kMAX:
    case Token::kMIN:
      return ParseMaxMin(h);
    case Token::kLPAR:
      switch (h.LookNextTok()) {
        case Token::kLET:
          return ParseLetExpr(h);
        case Token::kTYPE:
          return ParseImmNum(h);
        default:
          break;
      }
      return ParseBinExpr(h);
    case Token::kID: {
      switch (h.LookNextTok()) {
        case Token::kLPAR:
          return ParseCall(h);
        case Token::kLSQB:
          return ParseLoad(h);
        default:
          break;
      }
      auto v = h.Pop();
      return ASTNode<ASTVariable>(v.sval);
    }
    case Token::kSTRING: {
      auto s = h.Pop();
      static_cast<void>(h.LookNextTok());
      return ASTNode<ASTStringImm>(s.sval);
    }
    case Token::kNOT:
      return ParseNot(h);
    case Token::kMINUS:
    case Token::kNUMBER:
      return ParseImmNum(h);
    default:
      break;
  }
  return ASTNode<ASTExpr>();
}

ASTExprList ParseParamList(TokHandler &h) {
  ASTExprList list;

  auto expr = ParseExpr(h);
  CHECK(expr);

  list.push_back(expr);

  while (h.FrontTok() == Token::kCOMMA) {
    h.pop();
    auto expr_ = ParseExpr(h);
    CHECK(expr_);
    list.push_back(expr_);
  }

  return list;
}

ASTStmtList ParseStmtBody(TokHandler &h) {
  CHECK_AND_POP(h, Token::kLBRACE);
  CHECK_AND_POP(h, Token::kENDLINE);

  auto list = ParseStmtList(h);

  CHECK_AND_POP(h, Token::kRBRACE);

  return list;
}

ASTStmtNode ParseLet(TokHandler &h) {
  CHECK_AND_POP(h, Token::kLET);

  auto id = h.Pop();
  CHECK(id.tok == Token::kID);

  CHECK_AND_POP(h, Token::kEQUAL);

  auto value = ParseExpr(h);
  CHECK(value);

  return ASTNode<ASTLet>(id.sval, value, ParseStmtList(h));
}

ASTStmtNode ParseAttr(TokHandler &h) {
  CHECK_AND_POP(h, Token::kDOUBLESLASH);
  CHECK_AND_POP(h, Token::kATTR);
  CHECK_AND_POP(h, Token::kLSQB);

  auto &stat = h.GetTokState();
  unsigned cnt = 0;
  std::string node;
  while (true) {
    char c = stat.code[stat.cur++];

    if (c == ']') {
      if (cnt == 0) {
        break;
      } else {
        --cnt;
      }
    } else if (c == '[') {
      ++cnt;
    }
    node.push_back(c);
  }

  auto id = h.Pop();
  CHECK(id.tok == Token::kID);

  CHECK_AND_POP(h, Token::kEQUAL);

  auto value = ParseExpr(h);
  CHECK(value);

  return ASTNode<ASTAttr>(node, id.sval, value, ParseStmtList(h));
}

ASTStmtNode ParseAssert(TokHandler &h) {
  CHECK_AND_POP(h, Token::kASSERT);
  CHECK_AND_POP(h, Token::kLPAR);

  auto cond = ParseExpr(h);
  CHECK(cond);

  auto msg = ParseExpr(h);
  CHECK(msg);

  return ASTNode<ASTAssert>(cond, msg, ParseStmtList(h));
}

ASTStmtNode ParseAllocate(TokHandler &h) {
  CHECK_AND_POP(h, Token::kALLOCATE);

  auto id = h.Pop();
  CHECK(id.tok == Token::kID);

  CHECK_AND_POP(h, Token::kLSQB);

  auto type = h.Pop();
  CHECK(type.tok == Token::kTYPE);

  ASTExprList list;
  while (h.FrontTok() == Token::kSTAR) {
    h.pop();
    auto extent = ParseImmNum(h);
    CHECK(extent);
    list.push_back(extent);
  }

  CHECK_AND_POP(h, Token::kRSQB);

  return ASTNode<ASTAllocate>(id.sval, type.itype, type.bits, list, ParseStmtList(h));
}

ASTStmtNode ParseProduce(TokHandler &h) {
  CHECK_AND_POP(h, Token::kPRODUCE);

  auto id = h.Pop();
  CHECK(id.tok == Token::kID);

  return ASTNode<ASTProduce>(id.sval, ParseStmtBody(h));
}

inline void ParseBound(TokHandler &h, ASTExprList &min, ASTExprList &ext) {
  CHECK_AND_POP(h, Token::kLSQB);

  auto m = ParseImmNum(h);
  CHECK(m);
  min.push_back(m);

  CHECK_AND_POP(h, Token::kCOMMA);

  auto e = ParseImmNum(h);
  CHECK(e);
  ext.push_back(e);

  CHECK_AND_POP(h, Token::kRSQB);
}

ASTStmtNode ParseRealize(TokHandler &h) {
  CHECK_AND_POP(h, Token::kREALIZE);

  auto id = h.Pop();
  CHECK(id.tok == Token::kID);

  CHECK_AND_POP(h, Token::kLESS);

  auto t = h.Pop();
  CHECK(t.tok == Token::kTYPE);

  CHECK_AND_POP(h, Token::kGREATER);
  CHECK_AND_POP(h, Token::kLPAR);

  ASTExprList min, ext;
  if (h.FrontTok() != Token::kRPAR) {
    ParseBound(h, min, ext);
    while (h.FrontTok() == Token::kCOMMA) {
      h.pop();
      ParseBound(h, min, ext);
    }
  }

  CHECK_AND_POP(h, Token::kRPAR);

  return ASTNode<ASTRealize>(id.sval, t.itype, t.bits, min, ext, ParseStmtBody(h));
}

ASTStmtNode ParseIf(TokHandler &h) {
  CHECK_AND_POP(h, Token::kIF);
  CHECK_AND_POP(h, Token::kLPAR);

  auto cond = ParseExpr(h);
  CHECK(cond);

  CHECK_AND_POP(h, Token::kRPAR);

  ASTStmtList then_case, else_case;
  then_case = ParseStmtBody(h);

  if (h.FrontTok() == Token::kELSE) {
    h.pop();
    if (h.FrontTok() == Token::kIF) {
      auto e = ParseIf(h);
      CHECK(e);
      else_case.push_back(e);
    } else {
      else_case = ParseStmtBody(h);
    }
  }

  return ASTNode<ASTIfThenElse>(cond, then_case, else_case);
}

ASTStmtNode ParseFor(TokHandler &h) {
  CHECK_AND_POP(h, Token::kFOR);
  CHECK_AND_POP(h, Token::kLPAR);

  auto id = h.Pop();
  CHECK(id.tok == Token::kID);

  CHECK_AND_POP(h, Token::kCOMMA);

  auto min = ParseExpr(h);
  CHECK(min);

  CHECK_AND_POP(h, Token::kCOMMA);

  auto ext = ParseExpr(h);
  CHECK(ext);

  CHECK_AND_POP(h, Token::kRPAR);

  return ASTNode<ASTFor>(id.sval, min, ext, ParseStmtBody(h));
}

ASTStmtNode ParseStore(TokHandler &h) {
  auto id = h.Pop();
  CHECK(id.tok == Token::kID);

  CHECK_AND_POP(h, Token::kLSQB);

  auto index = ParseExpr(h);
  CHECK(index);

  CHECK_AND_POP(h, Token::kRSQB);
  CHECK_AND_POP(h, Token::kEQUAL);

  auto value = ParseExpr(h);
  CHECK(value);

  auto pred = ASTNode<ASTExpr>();
  if (h.FrontTok() == Token::kIF) {
    h.pop();
    pred = ParseImmNum(h);
  } else {
    pred = ASTNode<ASTIntImm>(1);
  }

  return ASTNode<ASTStore>(id.sval, value, index, pred);
}

ASTStmtNode ParseProvide(TokHandler &h) {
  auto id = h.Pop();
  CHECK(id.tok == Token::kID);

  CHECK_AND_POP(h, Token::kLPAR);

  ASTExprList list;
  if (h.LookNextTok() != Token::kRPAR) {
    list = ParseParamList(h);
  }

  CHECK_AND_POP(h, Token::kRPAR);

  if (h.LookNextTok() == Token::kEQUAL) {
    h.pop();

    auto value = ParseExpr(h);
    CHECK(value);

    return ASTNode<ASTProvide>(id.sval, value, list);
  }

  auto pair = ParseCallTail(h);
  auto &t = pair.first;
  auto &call_t = pair.second;

  auto call = ASTNode<ASTCall>(id.sval, t.itype, t.bits, call_t.tok, list);
  return ASTNode<ASTEvaluate>(call);
}

ASTStmtNode ParseStmt(TokHandler &h) {
  using pfunc = ASTStmtNode (*)(TokHandler &);
  static const std::map<Token, pfunc> funcMap = {
    {Token::kLET, ParseLet},         {Token::kDOUBLESLASH, ParseAttr},
    {Token::kASSERT, ParseAssert},   {Token::kALLOCATE, ParseAllocate},
    {Token::kPRODUCE, ParseProduce}, {Token::kREALIZE, ParseRealize},
    {Token::kIF, ParseIf},           {Token::kFOR, ParseFor},
  };

  auto front = h.FrontTok();
  auto it = funcMap.find(front);
  if (it != funcMap.end()) {
    return (*it->second)(h);
  } else if (Token::kID == front) {
    auto tmp = h.LookNextTok();
    if (Token::kLSQB == tmp) {
      return ParseStore(h);
    } else if (Token::kLPAR == tmp) {
      return ParseProvide(h);
    }
  }
  auto value = ParseExpr(h);
  return value ? ASTNode<ASTEvaluate>(value) : ASTNode<ASTStmt>();
}

ASTStmtList ParseStmtList(TokHandler &h) {
  ASTStmtList list;

  while (h.FrontTok() == Token::kENDLINE) {
    h.pop();
  }

  ASTStmtNode node = ParseStmt(h);
  CHECK(node);
  list.push_back(node);

  while (true) {
    while (h.FrontTok() == Token::kENDLINE) {
      h.pop();
    }
    if ((node = ParseStmt(h))) {
      list.push_back(node);
    } else {
      break;
    }
  }

  return list;
}
}  // namespace

ASTStmtList GenAST(TokState &stat) {
  TokHandler h(stat);
  ASTStmtList list;

  while (h.FrontTok() != Token::kEND) {
    if (h.FrontTok() != Token::kENDLINE) {
      auto stmt = ParseStmt(h);
      CHECK(stmt);
      list.push_back(stmt);
    } else {
      h.pop();
    }
  }

  return list;
}
}  // namespace ir
}  // namespace akg
