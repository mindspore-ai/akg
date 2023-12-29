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

#ifndef INCLUDE_AKG_IR_GENERICTREE_H_
#define INCLUDE_AKG_IR_GENERICTREE_H_

#include "tvm/ir.h"
#include "tvm/ir_visitor.h"
#include <tvm.h>

#include <vector>
#include <memory>
#include <string>

using std::string;
using std::unique_ptr;
using std::vector;

template <class T>
class GenericTree {
 public:
  // GenericTreeReferce
  using GTCref = unique_ptr<GenericTree<T>>;
  T principal_;
  vector<GTCref> children_;
  explicit GenericTree(const T &principal) { this->principal_ = principal; }
  ~GenericTree() {}
};

namespace akg {
namespace ir {
// Any subtyping on this class must have
// GenericTree<R>& , std::function<R (const NodeRef&)> as the first two arguments of constructors
template <class T>
class IRToGenericTree;

template <class T, class IRTG = IRToGenericTree<T(const NodeRef &)>>
unique_ptr<GenericTree<T>> ToGenericTree(const NodeRef &tree, const T &basecase,
                                         const std::function<T(const NodeRef &)> &f);

string DefaultDumpNode(const NodeRef &x);

template <class IRTG = IRToGenericTree<string(const NodeRef &)>>
string PrintDumpTree(const NodeRef &tree, const std::function<string(const NodeRef &)> &f = DefaultDumpNode);

string DefaultPrintDumpTree(const NodeRef &tree);
}  // namespace ir
}  // namespace akg

#endif  // INCLUDE_AKG_IR_GENERICTREE_H_
