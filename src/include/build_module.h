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
#include <exception>

#include "codegen/util.h"

namespace akg {
extern AttrMap g_attrs;
extern Array<NodeRef> g_external_call_name;

/*
 * Custom exception used when memory allocation fails and triggers micro-tuning to try to recover from failure.
 */
class MemoryAllocationException : public std::exception {
 public:
  MemoryAllocationException(const std::string &scope, uint64_t need_bits, uint64_t alloc_bits)
      : scope_(scope), need_bits_(need_bits), alloc_bits_(alloc_bits){};

  const char *what() const throw() {
    std::runtime_error re(("Allocation exceed bound of memory tag " + scope_ + ": need " + std::to_string(need_bits_) +
                           " bits, total alloc " + std::to_string(alloc_bits_) + " bits.")
                            .c_str());
    return re.what();
  }

  std::string scope_{""};
  uint64_t need_bits_{0};
  uint64_t alloc_bits_{0};
};

NodeRef LowerStmt(Schedule sch, const Array<NodeRef> &in_args, const Array<NodeRef> &shape_vars,
                  const std::string &name, const Map<Tensor, Buffer> &in_binds,
                  const Map<std::string, NodeRef> &in_attrs, bool simple_mode, bool polyhedral, bool tuning,
                  const std::string &target, const BuildConfig &config, Array<NodeRef> *args,
                  Array<NodeRef> *arg_list_0, Map<Tensor, Buffer> *binds, Map<Tensor, Buffer> *binds_0,
                  std::vector<size_t> *split_index, bool lower_list = false);

NodeRef LowerFunc(Stmt &stmt, const std::string &name, const BuildConfig &config, const Array<NodeRef> &all_args);

NodeRef Lower(Schedule sch, const Array<NodeRef> &in_args, const Array<NodeRef> &shape_vars, const std::string &name,
              const Map<Tensor, Buffer> &in_binds, const Map<std::string, NodeRef> &in_attrs, bool simple_mode,
              bool polyhedral, bool tuning, const std::string &target, const BuildConfig &config, bool get_stmt = false);

air::runtime::Module BuildModule(const Schedule &inputs, const Array<NodeRef> &in_args,
                                 const Array<NodeRef> &shape_vars, const std::string &target_name,
                                 const std::string &name, const Map<Tensor, Buffer> &in_binds,
                                 const Map<std::string, NodeRef> &in_attrs, bool polyhedral, const std::string &target,
                                 const BuildConfig &config);

class BuildRst;

BuildRst BuildToFunc(const Schedule &inputs, const Array<NodeRef> &in_args, const Array<NodeRef> &shape_vars,
                     const std::string &name, const Map<Tensor, Buffer> &in_binds,
                     const Map<std::string, NodeRef> &in_attrs, bool polyhedral, const std::string &target,
                     const BuildConfig &config);

air::runtime::Module BuildToModule(const NodeRef &ref, const std::string &target_name = "cce");

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

enum class LowerStage { BEGIN, TUNING, POLY, BEFORE_FLATTEN, FLATTEN, BEFORE_MULTICORE, BEFORE_REWRITE, REWRITE, END };

class LowerData {
 public:
  LowerData(Array<NodeRef> &args, Array<NodeRef> &arg_list_0, Map<Tensor, Buffer> &binds, Map<Tensor, Buffer> &binds_0,
            const Array<NodeRef> &shape_vars, const std::string &name, bool simple_mode, bool polyhedral, bool tuning,
            const std::string &target, const BuildConfig &config, bool get_feature = false)
      : args_(args),
        arg_list_0_(arg_list_0),
        binds_(binds),
        binds_0_(binds_0),
        shape_vars_(shape_vars),
        name(name),
        simple_mode_(simple_mode),
        polyhedral_(polyhedral),
        tuning_(tuning),
        target_(target),
        config_(config),
        get_feature_(get_feature) {}
  Array<NodeRef> &args_;
  Array<NodeRef> &arg_list_0_;
  Map<Tensor, Buffer> &binds_;
  Map<Tensor, Buffer> binds_0_;
  const Array<NodeRef> &shape_vars_;
  std::string name;
  bool simple_mode_{false};
  bool polyhedral_{false};
  bool tuning_{false};
  std::string target_;
  const BuildConfig &config_;
  bool get_feature_{false};
};

NodeRef LowerAscend(Stmt &stmt, LowerData &data, LowerStage begin = LowerStage::BEGIN,
                    LowerStage end = LowerStage::END);
void RenameBinds(Map<Tensor, Buffer> &binds, const BuildConfig &config, Array<NodeRef> &tensor_args_list,
                 Array<NodeRef> &buffer_args_list, Map<Tensor, Tensor> &tensor_replace);

void GetFlattenedBinds(const Array<NodeRef> &args, const Map<Tensor, Buffer> &binds, const BuildConfig &config,
                       Array<NodeRef> &out_args, Map<Tensor, Buffer> &out_binds, bool is_dynamic);

void FixParametricBinds(const Map<Tensor, Buffer> &binds, const Array<NodeRef> &in_args, const BuildConfig &config,
                        Map<Tensor, Buffer> *out_binds, Array<NodeRef> *out_args);
}  // namespace akg

#endif  // INCLUDE_AKG_BUILD_MODULE_H_
