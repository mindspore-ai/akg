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

#ifndef INCLUDE_AKG_BUILD_MODULE_H_
#define INCLUDE_AKG_BUILD_MODULE_H_

#include <string>

#include "codegen/util.h"

namespace akg {
extern AttrMap global_attrs;
NodeRef Lower(Schedule sch, const Array<NodeRef> &in_args, const Array<NodeRef> &shape_vars, const std::string &name,
              const Map<Tensor, Buffer> &in_binds, const Map<std::string, NodeRef> &in_attrs, bool simple_mode,
              bool polyhedral, bool tuning, bool aicpu, const BuildConfig &config);

ktvm::runtime::Module BuildModule(const Schedule &inputs, const Array<NodeRef> &in_args,
                                  const Array<NodeRef> &shape_vars, const std::string &target_name,
                                  const std::string &name, const Map<Tensor, Buffer> &in_binds,
                                  const Map<std::string, NodeRef> &in_attrs, bool polyhedral, bool aicpu,
                                  const BuildConfig &config);

class BuildRst;

BuildRst BuildToFunc(const Schedule &inputs, const Array<NodeRef> &in_args, const Array<NodeRef> &shape_vars,
                     const std::string &name, const Map<Tensor, Buffer> &in_binds,
                     const Map<std::string, NodeRef> &in_attrs, bool polyhedral, bool aicpu, const BuildConfig &config);

ktvm::runtime::Module BuildToModule(const NodeRef &ref, const std::string &target_name = "cce");

class BuildRstNode : public Node {
 public:
  NodeRef rst;
  std::string kernel_name;

  TVM_DLL static BuildRst make(const NodeRef &rst, const std::string &kernel_name);

  void VisitAttrs(AttrVisitor *v) {
    v->Visit("rst", &rst);
    v->Visit("kernel_name", &kernel_name);
  }

  static constexpr const char *_type_key = "BuildRst";
  TVM_DECLARE_BASE_NODE_INFO(BuildRstNode, Node);
};

class BuildRst : public NodeRef {
 public:
  ~BuildRst() = default;
  TVM_DEFINE_NODE_REF_METHODS(BuildRst, NodeRef, BuildRstNode);
};
}  // namespace akg

#endif  // INCLUDE_AKG_BUILD_MODULE_H_
