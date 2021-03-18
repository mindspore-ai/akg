/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef UT_AUTO_POLY_TEST_BASE_H_
#define UT_AUTO_POLY_TEST_BASE_H_
#include <map>
#include <string>
#include <gtest/gtest.h>
#include <tvm/expr.h>
#include "codegen/util.h"
#include "contrib/cce_parm/cceconf.h"
#include "base/expr_builder.h"

namespace akg {
class AutoPolyTestBase : public ::testing::Test {
 public:
  AutoPolyTestBase() = default;
  ~AutoPolyTestBase() = default;
  static std::map<std::string, std::string> InitMapMode();
  void RegisterTensor(const air::Tensor &tensor);
  void SetRunMode(const std::string &mode);
  void GlobalAttrSetIsDynamic(bool arg) {
    global_attrs_.Set("is_dynamic", arg ? UTExprBuilder::IntImm(1, air::Int(1)) :
                                          UTExprBuilder::IntImm(0, air::Int(1)));
  }

  void GlobalAttrSetDynamic(bool arg) {
    global_attrs_.Set("dynamic", arg ? UTExprBuilder::IntImm(1, air::Int(1)) :
                                       UTExprBuilder::IntImm(0, air::Int(1)));
  }

  void GlobalAttrSetDumpPassIR(bool arg) {
    global_attrs_.Set("dump_pass_ir", arg ? UTExprBuilder::IntImm(1, air::Int(1)) :
                                            UTExprBuilder::IntImm(0, air::Int(1)));
  }

  void GlobalAttrSetDumpPolyDir(const std::string &path) {
    global_attrs_.Set("dump_poly_dir", air::ir::StringImm::make(path));
  }

  void GlobalAttrSetKernalName(const std::string &name) {
    global_attrs_.Set("kernel_name", air::ir::StringImm::make(name));
  }

  static std::map<std::string, std::string> map_mode_;

 protected:
  air::Map<air::Tensor, air::Buffer> binds_;
  AttrMap global_attrs_;
};  // class AutoPolyTestBase
}  // namespace akg
#endif
