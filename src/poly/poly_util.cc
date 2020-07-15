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
#include "poly/poly_util.h"

namespace akg {
namespace ir {
namespace poly {
unsigned int WrappedStrtol(const std::string &str) {
  const int base = 10;
  char *endptr = nullptr;
  auto ret = std::strtol(str.c_str(), &endptr, base);
  if (endptr == nullptr || *endptr != '\0') LOG(FATAL) << "failed to convert string '" << str << "' to number";
  return static_cast<unsigned int>(ret);
}

bool IsEndsWith(const std::string &str, const std::string &suffix) {
  if (str.size() < suffix.size()) {
    return false;
  }
  std::string compare = str.substr(str.size() - suffix.size());
  return (compare == suffix);
}

std::vector<std::string> Split(const std::string &str, const std::string &pattern) {
  std::vector<std::string> res;
  if (str.empty()) {
    return res;
  }
  // Add a delimiter at the end of the string to facilitate the interception of the last paragraph.
  std::string strs = str + pattern;
  size_t pos = strs.find(pattern);

  while (pos != std::string::npos) {
    std::string temp = strs.substr(0, pos);
    if (!temp.empty()) {
      res.push_back(temp);
    }
    // Remove the broken strings and segment them in the remaining strings.
    strs = strs.substr(pos + 1, strs.size());
    pos = strs.find(pattern);
  }
  return res;
}

std::vector<int> SplitString(const std::string &str, const std::string &separator) {
  std::vector<int> values;
  size_t pos = 0;
  while (pos < str.length()) {
    int value = static_cast<int>(std::strtol(&str.c_str()[pos], nullptr, 10));
    if (errno != EINVAL) {
      values.push_back(value);
    }

    pos = str.find(separator, pos);
    if (pos == std::string::npos) {
      break;
    }
    pos += separator.length();
  }
  return values;
}

isl::ast_node CanonicalizeBlockInAst(const isl::ast_node &astNode) {
  if (auto block_node = astNode.as<isl::ast_node_block>()) {
    auto children = block_node.get_children();

    auto num_children = children.size();
    isl::ast_node_list new_block = children.drop(0, num_children);
    for (size_t i = 0; i < num_children; ++i) {
      auto new_child = CanonicalizeBlockInAst(children.get_at(i));
      if (auto new_child_block = new_child.as<isl::ast_node_block>()) {
        new_block = new_block.concat(new_child_block.children());
      } else {
        new_block = new_block.add(new_child);
      }
    }

    isl_ast_node *new_block_node = isl_ast_node_from_ast_node_list(new_block.copy());
    return isl::manage(new_block_node);
  }
  return astNode;
}

Expr RemoveCast(Expr e) {
  if (const auto a = e.as<air::ir::Add>()) {
    return air::ir::Add::make(RemoveCast(a->a), RemoveCast(a->b));
  } else if (const auto s = e.as<air::ir::Sub>()) {
    return air::ir::Sub::make(RemoveCast(s->a), RemoveCast(s->b));
  } else if (const auto m = e.as<air::ir::Mul>()) {
    return air::ir::Mul::make(RemoveCast(m->a), RemoveCast(m->b));
  } else if (const auto d = e.as<air::ir::Div>()) {
    return air::ir::Div::make(RemoveCast(d->a), RemoveCast(d->b));
  } else if (const auto cast = e.as<air::ir::Cast>()) {
    return RemoveCast(cast->value);
  } else if (const auto imm = e.as<air::ir::IntImm>()) {
    return air::ir::IntImm::make(Int(32), imm->value);
  }
  return e;
}

Stmt PeelOuterLetStmt(const Stmt &s, std::vector<Stmt> &outer_stmts) {
  auto body = s;
  while (auto op = body.as<LetStmt>()) {
    outer_stmts.push_back(LetStmt::make(op->var, op->value, Evaluate::make(0)));
    body = op->body;
  }
  return body;
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
