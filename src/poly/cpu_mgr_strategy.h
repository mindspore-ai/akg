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

#ifndef POLY_CPU_MGR_STRATEGY_H_
#define POLY_CPU_MGR_STRATEGY_H_

#include "poly/pass_mgr_strategy.h"

namespace akg {
namespace ir {
namespace poly {
class CPUMgrStrategy : public PassMgrStrategy {
 public:
  explicit CPUMgrStrategy(ScopInfo &scop_info) : PassMgrStrategy(scop_info) {
    pass_info_.coincident_ = scop_info_.user_config_.GetConsiderCoincidence();
  }
  ~CPUMgrStrategy() override = default;

  void RegisterTilingPasses() override;
  void RegisterMemPromPasses() override;
  void RegisterPasses() override;
};

}  // namespace poly
}  // namespace ir
}  // namespace akg
#endif  // POLY_CPU_MGR_STRATEGY_H_
