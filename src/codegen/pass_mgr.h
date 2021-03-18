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

#ifndef CODEGEN_PASS_MGR_H_
#define CODEGEN_PASS_MGR_H_
#include <tvm/ir_pass.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "codegen/util.h"

namespace akg {
using air::runtime::TVMArgs;
using air::runtime::TVMArgsSetter;
using air::runtime::TVMRetValue;

template <typename T>
void DumpRealContent(const T &content, std::ostream &buf) {
  buf << content;
}

template <>
inline void DumpRealContent(const LoweredFunc &lower_func, std::ostream &buf) {
  buf << lower_func->body;
}

template <>
inline void DumpRealContent(const Array<LoweredFunc> &func_list, std::ostream &buf) {
  for (auto func : func_list) {
    buf << "---------" << func->name;
    buf << "\n";
    buf << func->body;
    buf << "\n---------\n";
  }
}

class PassMgr {
 public:
  template <typename... Args>
  PassMgr(const std::string &func, Args &&... args) : pass_name_(func) {
    InitializeSubName();

    int args_num = sizeof...(Args) + 1;
    args_values_.resize(args_num);
    args_types_.resize(args_num);
    air::runtime::detail::for_each(TVMArgsSetter(args_values_.data(), args_types_.data()),
                                    std::forward<Args>(args)...);
  }

  ~PassMgr() = default;

  PassMgr &enable_timer() {
    enable_timer_ = true;
    return *this;
  }

  template <typename T>
  operator T() const {
    auto res = Run().operator T();

    if (tl_config_->dump_pass_ir) {
      DumpIr(std::bind(DumpRealContent<T>, res, std::placeholders::_1));
    }
    TryDumpC(res);
    return res;
  }

  static void ClearPassId() {
    tl_pass_id_ = -1;
  }
  static std::string &GetDir() {
    return tl_dump_ir_dir_;
  }
  static void SetDir(const std::string &str) {
    tl_dump_ir_dir_ = str;
  }
  static void SetArgs(const air::Array<NodeRef> &args) {
    tl_args_ = args;
  }

 private:
  void InitializeSubName();
  TVMRetValue Run() const;
  void DumpIr(std::function<void(std::ostream &os)> print) const;
  bool ShouldDumpC() const;
  std::string GetDumpIrFilePath() const;

  thread_local static int tl_pass_id_;
  thread_local static air::BuildConfig tl_config_;
  thread_local static std::string tl_dump_ir_dir_;
  thread_local static air::Array<NodeRef> tl_args_;

  std::string pass_name_;
  std::string sub_name_;
  std::vector<TVMValue> args_values_;
  std::vector<int> args_types_;

  bool enable_timer_ = false;

  template <typename T>
  void TryDumpC(const T &node) const {
    if (!ShouldDumpC()) {
      return;
    }
    Array<Buffer> extern_buffers;
    for (const auto &arg : tl_args_) {
      extern_buffers.push_back(air::Downcast<Buffer>(arg));
    }
    auto csim_fname = GetDumpIrFilePath().append(".cpp");
    std::ofstream of(csim_fname);
    CHECK(of.is_open()) << "Failed to open " << csim_fname << " to dump C.";

    if (node->template IsInstance<typename Stmt::ContainerType>()) {
      Stmt stmt = air::Downcast<Stmt>(node);
      of << akg::DumpC(stmt, extern_buffers);
    } else {
      LOG(INFO) << "unknown node type, cannot dump C of pass " << pass_name_;
    }
    of.close();
  }
};

template <typename... Args>
PassMgr make_pass(const std::string &func, Args &&... args) {
  return PassMgr(func, std::forward<Args>(args)...);
}

#define NEXT_PASS(PASS, args...) make_pass("ir_pass." #PASS, args).enable_timer()
}  // namespace akg

#endif  // CODEGEN_PASS_MGR_H_
