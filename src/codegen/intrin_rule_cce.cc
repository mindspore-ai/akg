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
#include <codegen/intrin_rule.h>

namespace air {
namespace codegen {
namespace intrin {
struct CCEMatch : public Direct {
  std::string operator()(Type t, const std::string &name) const {
    return "cce_" + name;
  }
};

TVM_REGISTER_GLOBAL("tvm.intrin.rule.cce.round").set_body(DispatchExtern<CCEMatch>);
TVM_REGISTER_GLOBAL("tvm.intrin.rule.cce.mod").set_body(DispatchExtern<CCEMatch>);
}  // namespace intrin
}  // namespace codegen
}  // namespace air
