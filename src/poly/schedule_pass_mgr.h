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

#ifndef POLY_PASS_MGR_H_
#define POLY_PASS_MGR_H_

#include "poly/schedule_pass.h"
#include "poly/pass_mgr_strategy.h"

namespace akg {
namespace ir {
namespace poly {

class SchedulePassMgr {
 public:
  SchedulePassMgr(ScopInfo &scop_info) : scop_info_(scop_info){}
  const std::vector<std::shared_ptr<SchedulePass>> &GetSchedulePasses() const;
  void RegisterPass(std::shared_ptr<SchedulePass>pass);
  isl::schedule Run(const isl::schedule &sch);
  isl::schedule Run(const isl::schedule &sch, const std::vector<std::shared_ptr<SchedulePass>> &passes);
  isl::schedule Run(const isl::schedule &sch, std::shared_ptr<PassMgrStrategy> strategy);
  ~SchedulePassMgr() {}

  bool need_restart_{false};
  ScopInfo &scop_info_;
 private:
  std::vector<std::shared_ptr<SchedulePass>> schedule_passes_;
};
}  // namespace poly
}  // namespace ir
}  // namespace akg
#endif  // POLY_PASS_MGR_H_
