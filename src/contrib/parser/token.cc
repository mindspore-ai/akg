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

#include "contrib/parser/token.h"

#include <dmlc/logging.h>

#include <fstream>
#include <sstream>
#include <algorithm>
#include <utility>

namespace akg {
namespace ir {
TokState GetTokStateFromCode(const std::string &code) {
  TokState s = {.tok = Token::kENDLINE,
                .cur = 0,
                .itype = ImmType::kINT,
                .bits = 0,
                .fval = 0,
                .uval = 0,
                .sval = "",
                .code = code};

  std::replace(s.code.begin(), s.code.end(), '\r', '\n');

  if (s.code.back() != '\n') {
    s.code.push_back('\n');
  }
  s.code.push_back('\0');

  return s;
}

TokState GetTokStateFromFile(const std::string &file) {
  std::ifstream f(file);
  CHECK(f);

  std::ostringstream ss;
  ss << f.rdbuf();

  return GetTokStateFromCode(ss.str());
}

namespace {
inline bool IsIdStart(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c == '_'); }

inline bool IsXDigit(char c) { return std::isdigit(c) || (c >= 'a' && c <= 'f'); }

inline bool IsIdBody(char c) { return IsIdStart(c) || std::isdigit(c) || c == '.'; }

Token ParseDoubleChar(char c0, char c1) {
  switch (c0) {
    case '=':
      switch (c1) {
        case '=':
          return Token::kEQEQUAL;
        default:
          break;
      }
      break;
    case '!':
      switch (c1) {
        case '=':
          return Token::kNOTEQUAL;
        default:
          break;
      }
      break;
    case '&':
      switch (c1) {
        case '&':
          return Token::kAND;
        default:
          break;
      }
      break;
    case '|':
      switch (c1) {
        case '|':
          return Token::kOR;
        default:
          break;
      }
      break;
    case '<':
      switch (c1) {
        case '=':
          return Token::kLESSEQUAL;
        default:
          break;
      }
      break;
    case '>':
      switch (c1) {
        case '=':
          return Token::kGREATEREQUAL;
        default:
          break;
      }
      break;
    case '/':
      switch (c1) {
        case '/':
          return Token::kDOUBLESLASH;
        default:
          break;
      }
      break;
    default:
      break;
  }
  return Token::kOPSET;
}

Token ParseSingleChar(char c) {
  switch (c) {
    case '(':
      return Token::kLPAR;
    case ')':
      return Token::kRPAR;
    case '[':
      return Token::kLSQB;
    case ']':
      return Token::kRSQB;
    case ':':
      return Token::kCOLON;
    case ',':
      return Token::kCOMMA;
    case ';':
      return Token::kSEMI;
    case '+':
      return Token::kPLUS;
    case '-':
      return Token::kMINUS;
    case '*':
      return Token::kSTAR;
    case '/':
      return Token::kSLASH;
    case '|':
      return Token::kVBAR;
    case '&':
      return Token::kAMPER;
    case '<':
      return Token::kLESS;
    case '>':
      return Token::kGREATER;
    case '=':
      return Token::kEQUAL;
    case '.':
      return Token::kDOT;
    case '%':
      return Token::kPERCENT;
    case '{':
      return Token::kLBRACE;
    case '}':
      return Token::kRBRACE;
    case '^':
      return Token::kCIRCUMFLEX;
    case '~':
      return Token::kTILDE;
    case '@':
      return Token::kAT;
    default:
      break;
  }
  return Token::kOPSET;
}

bool IsNumberTail(const std::string &str, std::string::size_type pos) {
  for (auto loc = pos; loc != str.size(); ++loc) {
    if (!std::isdigit(str[loc])) {
      return false;
    }
  }
  return true;
}

bool CheckSingleType(TokState &stat, const std::string &str, const std::string &type, ImmType kt) {
  if (str.find(type) == 0 && IsNumberTail(str, type.size()) &&
      (str.size() < type.size() + 2 || str[type.size()] != '0')) {
    auto tmp = std::string(str.begin() + static_cast<int>(type.size()), str.end());
    char *end = nullptr;
    stat.bits = static_cast<unsigned>(std::strtoul(tmp.data(), &end, 10));
    CHECK(end != tmp.data());
    if (stat.bits > 64 || stat.bits == 0) {
      return false;
    }
    stat.itype = kt;
    return true;
  }
  return false;
}

bool CheckHandle(TokState &stat, const std::string &str) {
  if (str == "handle") {
    stat.itype = ImmType::kHANDLE;
    return true;
  }
  return false;
}

bool CheckIsType(TokState &stat, const std::string &str) {
  return CheckSingleType(stat, str, "int", ImmType::kINT) || CheckSingleType(stat, str, "uint", ImmType::kUINT) ||
         CheckSingleType(stat, str, "float", ImmType::kFLOAT) || CheckHandle(stat, str);
}
}  // namespace

Token GetNextToken(TokState &stat) {
  auto &cur = stat.cur;
  auto &code = stat.code;
  CHECK(static_cast<uint32_t>(cur) < code.size());

  char c = code[cur];

#define RETURN_TOK(t) return stat.tok = t;

  while (c == ' ' || c == '\t' || c == '\014') {
    c = code[++cur];
  }

  if (c == '\0') {
    ++cur;
    RETURN_TOK(Token::kEND);
  }

  if (c == '\n') {
    ++cur;
    RETURN_TOK(Token::kENDLINE);
  }

  // id or key
  if (IsIdStart(c)) {
    int start = cur;
    while (IsIdBody(code[++cur])) {
    }

    std::string str(code.begin() + start, code.begin() + cur);

    auto it = g_str2key.find(str);
    if (it != g_str2key.end()) {
      RETURN_TOK(it->second);
    }

    if (CheckIsType(stat, str)) {
      RETURN_TOK(Token::kTYPE);
    }
    stat.sval = std::move(str);
    RETURN_TOK(Token::kID);
  }

  if (c == '.') {
    ++cur;
    RETURN_TOK(Token::kDOT);
  }

  // number
  if (std::isdigit(c)) {
    std::string str;
    if (c == '0') {
      // hex mode
      if (code[++cur] == 'x') {
        str += "0x";
        while (IsXDigit(c = code[++cur])) {
          str.push_back(c);
        }
        char *end = nullptr;
        stat.uval = std::strtoul(str.data(), &end, 16);
        CHECK(str.data() != end);
        stat.itype = ImmType::kHEX;
        RETURN_TOK(Token::kNUMBER);
      }
      c = code[--cur];
    }

    do {
      str.push_back(c);
    } while (std::isdigit(c = code[++cur]));

    if (c == '.' || c == 'h' || c == 'f') {  // float mode
      if (c == '.') {
        str.push_back('.');
        while (std::isdigit(c = code[++cur])) {
          str.push_back(c);
        }
      }
      switch (c) {
        case 'h':
          stat.bits = 16;
          ++cur;
          break;
        case 'f':
          stat.bits = 32;
          ++cur;
          break;
        default:
          stat.bits = 64;
          break;
      }
      char *end = nullptr;
      stat.fval = std::strtod(str.data(), &end);
      CHECK(end != str.data());
      stat.itype = ImmType::kFLOAT;
      RETURN_TOK(Token::kNUMBER);
    }

    // decimal mode
    char *end = nullptr;
    stat.uval = std::strtoul(str.data(), &end, 10);
    CHECK(end != str.data());
    stat.itype = ImmType::kUINT;
    RETURN_TOK(Token::kNUMBER);
  }

  if (c == '"') {
    stat.sval.clear();
    while ((c = code[++cur]) != '"') {
      stat.sval.push_back(c);
    }
    ++cur;
    RETURN_TOK(Token::kSTRING);
  }

  // 'two-or-more-char token
  auto tok = ParseDoubleChar(c, code[++cur]);
  if (tok != Token::kOPSET) {
    ++cur;
    RETURN_TOK(tok);
  }
  --cur;

  // single-char token
  tok = ParseSingleChar(c);
  ++cur;
  RETURN_TOK(tok);
}

namespace {
std::string ToString(const Token &tok) {
  std::string str;
  switch (tok) {
#if 1  // two printing ways
#define KEY(name, ...) \
  case Token::k##name: \
    str = #name;       \
    break;

#define TOKEN(name, ...) \
  case Token::k##name:   \
    str = #name;         \
    break;
#else
#define KEY(name, symbol) \
  case Token::k##name:    \
    str = symbol;         \
    break;

#define TOKEN(name, symbol) \
  case Token::k##name:      \
    str = symbol;           \
    break;
#endif

#include "key_word.md"
#include "meta_token.md"
#undef KEY
#undef TOKEN
    default:
      CHECK(false);
      break;
  }
  return str;
}
}  // namespace

std::ostream &operator<<(std::ostream &os, const Token &tok) {
  os << ToString(tok);
  return os;
}

const std::map<std::string, Token> g_str2key = {
#define KEY(name, symbol) {symbol, Token::k##name},
#include "key_word.md"
#undef KEY
};

std::ostream &operator<<(std::ostream &os, const ImmType &t) {
  switch (t) {
#define PRINT(t)      \
  case ImmType::k##t: \
    os << #t;         \
    break
    PRINT(UINT);
    PRINT(INT);
    PRINT(FLOAT);
    PRINT(HEX);
#undef PRINT
    default:
      CHECK(false);
      break;
  }
  return os;
}

void DumpTokenFromState(TokState &s, std::ostream &os) {
  Token t;

  while ((t = GetNextToken(s)) != Token::kEND) {
    os << t;
    if (t == Token::kNUMBER) {
      os << '(';
      switch (s.itype) {
        case ImmType::kUINT:
          os << "u_" << s.uval;
          break;
        case ImmType::kHEX:
          os << "0x_" << std::hex << s.uval << std::dec;
          break;
        case ImmType::kFLOAT:
          os << 'f' << s.bits << '_' << s.fval;
          break;
        default:
          CHECK(false);
      }
      os << ')';
    }
    if (t == Token::kTYPE) {
      os << '(' << s.itype << s.bits << ')';
    }
    if (t == Token::kID || t == Token::kSTRING) {
      os << '(' << s.sval << ')';
    }
    if (t == Token::kENDLINE) {
      os << std::endl;
    } else {
      os << "   ";
    }
  }
  os << t << std::endl;
}
}  // namespace ir
}  // namespace akg
