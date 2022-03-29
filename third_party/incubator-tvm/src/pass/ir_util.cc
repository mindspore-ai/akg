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
 * \file ir_util.cc
 * \brief Helper functions to construct and compose IR nodes.
 */

/*
 * 2019.12.30 - Add utility functions.
 * 2022.3.28 - Add function VarsFromArgs.
 */

#include "ir_util.h"
#include <tvm/ir_visitor.h>

namespace air {
namespace ir {

int64_t gcd(int64_t a, int64_t b) {
  a = a >= 0 ? a : -a;
  b = b >= 0 ? b : -b;
  if (a < b) std::swap(a, b);
  while (b != 0) {
    int64_t tmp = b;
    b = a % b;
    a = tmp;
  }
  return a;
}

int64_t lcm(int64_t a, int64_t b) {
  if (a == 0 && b == 0) {
    return 0;
  } else {
    return (a * b) / gcd(a, b);
  }
}

Stmt MergeNest(const std::vector<Stmt>& nest, Stmt body) {
  // use reverse iteration
  for (auto ri = nest.rbegin(); ri != nest.rend(); ++ri) {
    Stmt s = *ri;
    if (const auto* for_ = s.as<For>()) {
      auto n = make_node<For>(*for_);
      CHECK(is_no_op(n->body));
      n->body = body;
      body = Stmt(n);
    } else if (const auto* let = s.as<LetStmt>()) {
      auto n = make_node<LetStmt>(*let);
      CHECK(is_no_op(n->body));
      n->body = body;
      body = Stmt(n);
    } else if (const auto* attr = s.as<AttrStmt>()) {
      auto n = make_node<AttrStmt>(*attr);
      CHECK(is_no_op(n->body));
      n->body = body;
      body = Stmt(n);
    } else if (const auto* ite = s.as<IfThenElse>()) {
      auto n = make_node<IfThenElse>(*ite);
      CHECK(is_no_op(n->then_case));
      CHECK(!n->else_case.defined());
      n->then_case = body;
      body = Stmt(n);
    } else if (const auto* block = s.as<Block>()) {
      auto n = make_node<Block>(*block);
      CHECK(is_no_op(n->rest));
      n->rest = body;
      body = Stmt(n);
    } else if (const auto* assert_ = s.as<AssertStmt>()) {
      auto n = make_node<AssertStmt>(*assert_);
      CHECK(is_no_op(n->body));
      n->body = body;
      body = Stmt(n);
    } else if (const auto* alloc = s.as<Allocate>()) {
      auto n = make_node<Allocate>(*alloc);
      CHECK(is_no_op(n->body));
      n->body = body;
      body = Stmt(n);
    } else {
      LOG(FATAL) << "not supported nest type";
    }
  }
  return body;
}

Stmt MergeNest(const std::vector<std::vector<Stmt> >& nest, Stmt body) {
  for (auto ri = nest.rbegin(); ri != nest.rend(); ++ri) {
    body = MergeNest(*ri, body);
  }
  return body;
}

Stmt MergeSeq(const std::vector<Stmt>& seq) {
  if (seq.size() == 0) return Evaluate::make(0);
  Stmt body = seq[0];
  for (size_t i = 1; i < seq.size(); ++i) {
    body = Block::make(body, seq[i]);
  }
  return body;
}

Array<Expr> VarsFromArgs(const Array<Expr> args) {
  Array<Expr> vars;
  size_t vars_size = vars.size();
  for (auto e: args) {
    PostOrderVisit(e, [&vars](const NodeRef &n) {
      if (n.as<Variable>() != nullptr) {
        vars.push_back(Downcast<Expr>(n));
      }
    });
    if (vars.size() <= vars_size) {
      // If no variable is found in the arg, then the arg itself is used as a variable (i.e. in broadcast where e = 0)
      vars.push_back(e);
    }
    vars_size = vars.size();
  }
  return vars;
}

}  // namespace ir
}  // namespace air
