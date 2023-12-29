/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#ifndef POLY_ANALYZE_SCHEDULE_H_
#define POLY_ANALYZE_SCHEDULE_H_

#include "poly/schedule_pass.h"

namespace akg {
namespace ir {
namespace poly {

class AnalyzeSchedule : public SchedulePass {
 public:
  AnalyzeSchedule(ScopInfo &scop_info) : scop_info_(scop_info) {
    pass_name_ = __FUNCTION__;
    target_ = scop_info.user_config_.GetTarget();
    stmt_ = scop_info.user_config_.GetBody();
  }
  ~AnalyzeSchedule() {}

  virtual isl::schedule Run(isl::schedule sch);

 private:
  void ConstructOuterBandNode();
  void ConstructInnerBandNode();
  void AppendBandNode(const isl::schedule_node &node, const std::function<void(const isl::schedule_node_band)> &f);

  std::string target_;
  Stmt stmt_;
  isl::schedule sch_;
  ScopInfo &scop_info_;
};

}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif  // POLY_ANALYZE_SCHEDULE_H_