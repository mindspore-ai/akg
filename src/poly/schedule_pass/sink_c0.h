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
#ifndef POLY_SINK_C0_H_
#define POLY_SINK_C0_H_

#include "poly/schedule_pass.h"

namespace akg {
namespace ir {
namespace poly {

/*
 * For each band node in schedule tree, get multi_union_pw_aff from the current band node. Then, move the last axis C0
 * schedule to the end of this multi_union_pw_aff and add the updated multi_union_pw_aff to the current band node.
 */
class SinkC0 : public SchedulePass {
 public:
  SinkC0() { pass_name_ = __FUNCTION__; }
  ~SinkC0() {}

  virtual isl::schedule Run(isl::schedule sch);

 private:
  bool FindC0Schedule(const isl::pw_aff_list &paList);
  void ExchangeCoincident(std::vector<int> &coincident, const isl::schedule_node &node,
                          const std::unordered_map<int, bool> lastIdxSchedule, const int &n);
  isl::schedule_node SinkC0Schedule(isl::schedule_node &node);
};

}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif  // POLY_SINK_C0_H_