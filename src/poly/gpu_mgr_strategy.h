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

#ifndef POLY_GPU_MGR_STRATEGY_H_
#define POLY_GPU_MGR_STRATEGY_H_

#include "poly/pass_mgr_strategy.h"

namespace akg {
namespace ir {
namespace poly {
class GPUMgrStrategy : public PassMgrStrategy {
 public:
  explicit GPUMgrStrategy(ScopInfo &scop_info) : PassMgrStrategy(scop_info) {}
  ~GPUMgrStrategy() override = default;

  void RegisterTilingPasses() override;
  void RegisterMemPromPasses() override;
  void RegisterPasses() override;
};

}  // namespace poly
}  // namespace ir
}  // namespace akg
#endif  // POLY_GPU_MGR_STRATEGY_H_
