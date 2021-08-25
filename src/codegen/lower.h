/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef CODEGEN_LOEWR_H_
#define CODEGEN_LOEWR_H_
#include <functional>
#include <unordered_map>
#include <utility>
#include <vector>
#include "codegen/util.h"
#include "codegen/pass_mgr.h"

namespace akg {
extern AttrMap g_attrs;

// Use to store the necessary for Lower.
class LowerData;
class LowerDataNode : public Node {
 public:
  Array<NodeRef> args;
  Array<NodeRef> arg_list_0;
  Map<std::string, NodeRef> attrs;
  Map<Tensor, Buffer> binds;
  Map<Tensor, Buffer> binds_0;
  BuildConfig config;
  std::string name;
  bool polyhedral{true};
  Schedule sch;
  Array<NodeRef> shape_vars;
  bool simple_mode{false};
  Array<Integer> split_index;
  std::string target{"Unknown"};
  bool tuning{false};

  TVM_DLL static LowerData make();
  TVM_DLL static LowerData make(Schedule sch, const Array<NodeRef> &args, const Map<Tensor, Buffer> &binds,
                                const Map<std::string, NodeRef> &attrs, const std::string &target,
                                const std::string &name, const BuildConfig &config, bool polyhedral = true,
                                bool tuning = false, bool simple_mode = false,
                                const Array<NodeRef> &shape_vars = Array<NodeRef>(),
                                const Array<Integer> &split_index = Array<Integer>(),
                                const Array<NodeRef> &arg_list_0 = Array<NodeRef>(),
                                const Map<Tensor, Buffer> &binds_0 = Map<Tensor, Buffer>());

  void VisitAttrs(AttrVisitor *v) {
    v->Visit("args", &args);
    v->Visit("arg_list_0", &arg_list_0);
    v->Visit("attrs", &attrs);
    v->Visit("binds", &binds);
    v->Visit("binds_0", &binds_0);
    v->Visit("config", &config);
    v->Visit("name", &name);
    v->Visit("polyhedral", &polyhedral);
    v->Visit("sch", &sch);
    v->Visit("shape_vars", &shape_vars);
    v->Visit("simple_mode", &simple_mode);
    v->Visit("split_index", &split_index);
    v->Visit("target", &target);
    v->Visit("tuning", &tuning);
  }

  static constexpr const char *_type_key = "LowerData";
  TVM_DECLARE_BASE_NODE_INFO(LowerDataNode, Node);
};

class LowerData : public NodeRef {
 public:
  LowerData() {}
  ~LowerData() = default;
  explicit LowerData(::air::ObjectPtr<::air::Object> n) : NodeRef(n) {}
  LowerDataNode *operator->() { return static_cast<LowerDataNode *>(data_.get()); }
  const LowerDataNode *operator->() const { return static_cast<const LowerDataNode *>(data_.get()); }
  operator bool() const { return this->defined(); }
  using ContainerType = LowerDataNode;
};

class LowerImpl {
 public:
  static LowerImpl &Instance();
  void Register(const std::string &target, std::function<NodeRef(const LowerData &, bool)> impl);
  NodeRef Run(const LowerData &data, bool get_stmt);

 private:
  LowerImpl() = default;
  ~LowerImpl() = default;
  LowerImpl(const LowerImpl &) = delete;
  LowerImpl &operator=(const LowerImpl &) = delete;

  std::unordered_map<std::string, std::function<NodeRef(const LowerData &, bool)>> impls_;
};

struct LowerImplRegister {
  LowerImplRegister(const std::string &target, std::function<NodeRef(const LowerData &, bool)> impl) {
    LowerImpl::Instance().Register(target, impl);
  }
};
#define REG_IMPL_LOWER(target, impl) REG_IMPL_LOWER_MID(__COUNTER__, target, impl)
#define REG_IMPL_LOWER_MID(ctr, target, impl) REG_IMPL_LOWER_UNIQ(ctr, target, impl)
#define REG_IMPL_LOWER_UNIQ(ctr, target, impl) \
  static LowerImplRegister g_lower_impl_register_##ctr __attribute__((unused)) = LowerImplRegister((target), (impl))

void ConfigDumpIr(const std::string &name, const BuildConfig &config, bool lower_list);
Stmt LowerInitWithSchedule(LowerData &data);
std::string GetErrorHint(const std::string &target);
using StageResult = std::pair<NodeRef, bool>;
}  // namespace akg
#endif  // CODEGEN_LOEWR_H_
