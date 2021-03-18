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

#ifndef CONTRIB_PARSER_TOKEN_H_
#define CONTRIB_PARSER_TOKEN_H_

#include <string>
#include <iostream>
#include <map>

namespace akg {
namespace ir {
// TOK_ENUM declaration
enum class Token {
#define KEY(name, ...) k##name,
#include "key_word.md"
#undef KEY

#define TOKEN(name, ...) k##name,
#include "meta_token.md"
#undef TOKEN
};

std::ostream &operator<<(std::ostream &os, const Token &tok);

extern const std::map<std::string, Token> g_str2key;

enum class ImmType {
  kINT,
  kUINT,
  kFLOAT,
  kHEX,
  kHANDLE,
};

std::ostream &operator<<(std::ostream &os, const ImmType &t);

// TOK_STATE declaration
struct TokState {
  Token tok;  // the lastest parsed token

  int cur;  // current location in code

  ImmType itype;
  unsigned bits;     // bits for immediate number
  double fval;       // immediate number for float
  uint64_t uval;     // immediate number for unsigned integer
  std::string sval;  // immediate number for string or Token::kID

  std::string code;  // code to be parsed
};

// TOK_FUNC declarations
TokState GetTokStateFromFile(const std::string &file);
TokState GetTokStateFromCode(const std::string &file);

Token GetNextToken(TokState &s);

void DumpTokenFromState(TokState &s, std::ostream &os);
}  // namespace ir
}  // namespace akg

#endif  // CONTRIB_PARSER_TOKEN_H_
