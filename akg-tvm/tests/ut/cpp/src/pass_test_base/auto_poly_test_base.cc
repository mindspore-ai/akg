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
#include "pass_test_base/auto_poly_test_base.h"

namespace akg {
std::map<std::string, std::string> AutoPolyTestBase::map_mode_ =
    AutoPolyTestBase::InitMapMode();

std::map<std::string, std::string> AutoPolyTestBase::InitMapMode() {
  std::map<std::string, std::string> res;
  return res;
}

void AutoPolyTestBase::SetRunMode(const std::string &mode) {
  auto it = map_mode_.find(mode);
  CHECK(it != map_mode_.end());
  cceconf::CceConf::getInstance()->setSection(it->second);
}

void AutoPolyTestBase::RegisterTensor(const air::Tensor &tensor) {
  const TensorNode *tensor_node = tensor.as<TensorNode>();
  std::string name = tensor_node->op->name;
  air::Buffer buf = air::BufferNode::make(
      air::Variable::make(Handle(), name),
      tensor_node->dtype,
      tensor_node->shape,
      Array<Expr>(),
      Expr(),
      name,
      "",
      -1,
      0,
      air::BufferType::kDefault);
  binds_.Set(GetRef<Tensor>(tensor_node), buf);
}
}  // namespace akg
