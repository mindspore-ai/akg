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
#include "ir_generictree.h"
#include <stdio.h>

#include <iostream>
#include <list>
#include <numeric>

#include "tvm/expr.h"
#include "tvm/ir.h"
#include "tvm/ir_visitor.h"

using std::list;

// template <class N>
list<string> linesOfGenericTree(const GenericTree<string> &tree) {
  list<string> ret;
  int current_child = 0;
  const int children_number = static_cast<int>(tree.children_.size());
  for (const auto &eachLineOf : tree.children_) {
    bool lastchild = (current_child == (children_number - 1));
    auto currentLines = linesOfGenericTree(*eachLineOf);
    list<string> newCurrentLines;
    int lineNo = 0;
    for (const auto &eachSubline : currentLines) {
      string newEachSubline("");
      if (lineNo == 0) {
        newEachSubline = "()----" + eachSubline;
      } else {
        newEachSubline = (lastchild ? "      " : "||    ") + eachSubline;
      }
      lineNo++;
      newCurrentLines.push_back(newEachSubline);
    }
    ret.splice(ret.end(), newCurrentLines);
    if (lastchild) {
      ret.emplace_back(std::string("      "));
      ret.emplace_back(std::string("      "));
      ret.emplace_back(std::string("      "));
    } else {
      ret.emplace_back(std::string("||    "));
      ret.emplace_back(std::string("||    "));
      ret.emplace_back(std::string("||    "));
    }
    current_child++;
  }
  ret.push_front((tree.principal_));
  return ret;
}

// template <class N>
string printGenericTree(const GenericTree<string> &tree) {
  string ret("");
  auto lines = linesOfGenericTree(tree);
  ret = std::accumulate(lines.begin(), lines.end(), ret,
                        [](const std::string &a, const std::string &b) { return (a + "\n" + b); });
  return ret;
}

namespace akg {
namespace ir {
template <class R>
class IRToGenericTree<R(const NodeRef &)> : public IRVisitor {
 public:
  using dumpFType = std::function<R(const NodeRef &)>;

  IRToGenericTree(GenericTree<R> &parent, const dumpFType f) : parent(parent), f(f) {}
  ~IRToGenericTree() override = default;

  void Visit(const NodeRef &node) final {
    unique_ptr<GenericTree<R>> newTree(new GenericTree<R>(f(node)));
    // create Tree here
    current = newTree.get();

    (this->parent).children_.push_back(std::move(newTree));

    // create sub IRToGenericTree
    IRToGenericTree<R(const NodeRef &)> subtree(*current, f);

    // dynamic dispatch
    subtree.IRVisitor::Visit(node);
  }

 protected:
  // This function can only (sensely) be called in the 'Visit_' function
  // i.e. after 'Visit'
  void AppendNameHint(const std::string &name) {
    CHECK(current != nullptr);
    current->principal_ += " : " + name;
  }

  void Visit_(const Variable *op) override {}

 private:
  GenericTree<R> &parent;
  GenericTree<R> *current = nullptr;
  dumpFType f;
};

string DefaultDumpNode(const NodeRef &x) { return x->GetTypeKey(); }

template <class IRTG>
string PrintDumpTree(const NodeRef &tree, const std::function<string(const NodeRef &)> &f) {
  auto x = ToGenericTree<string, IRTG>(tree, "TreeIs", f);
  auto ret = printGenericTree(*(x.get()));
  return ret;
}

template <class T, class IRTG>
unique_ptr<GenericTree<T>> ToGenericTree(const NodeRef &tree, const T &basecase,
                                         const std::function<T(const NodeRef &)> &f) {
  unique_ptr<GenericTree<T>> ret(new GenericTree<T>(basecase));
  IRTG v(*ret, f);
  v.Visit(tree);
  return ret;
}

class IRToDumpedTree : public IRToGenericTree<string(const NodeRef &)> {
 public:
  IRToDumpedTree(GenericTree<string> &parent, const std::function<string(const NodeRef &)> &f)
      : IRToGenericTree<string(const NodeRef &)>(parent, f) {}
  ~IRToDumpedTree() override = default;

  void Visit_(const Variable *op) final {
    LOG(INFO) << " Meet On Variable";
    AppendNameHint(op->name_hint);
  }
};

string DefaultPrintDumpTree(const NodeRef &tree) { return PrintDumpTree(tree, DefaultDumpNode); }
}  // namespace ir
}  // namespace akg
