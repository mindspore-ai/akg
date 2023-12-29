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
#ifndef POLY_SINK_LAST_AXIS_H_
#define POLY_SINK_LAST_AXIS_H_

#include "poly/schedule_pass.h"
#include "poly/pass_info.h"

namespace akg {
namespace ir {
namespace poly {

/*
 * Try to sink the last axis of outer band to the leaves of the schedule tree.
 *
 * The criteria that the last axis can be sinked:
 * 1) the axis is the last axis in the outer band schedule.
 * 2) the axis is the last axis in the domain of each statement.
 * 3) all dependencies of the last axis are equality constraints. (i.e. S_1[c0] -> S_2[c0' = c0])
 * 4) all dependencies of the last axis do not appear in other non-last axes.
 * 5) the domain of the last axis is not larger than a threshold (otherwise it still should be tiled).
 *
 * SinkLastAxis will:
 * 1) remove the C0 axis from the outer band schedule, and
 * 2) add a partial schedule (C0) to each leaf filter node that contains the last axis.
 */
class SinkLastAxis : public SchedulePass {
 public:
  SinkLastAxis(PassInfo &pass_info) : pass_info_(pass_info) { pass_name_ = __FUNCTION__; }
  ~SinkLastAxis() {}

  virtual isl::schedule Run(isl::schedule sch);

 private:
  PassInfo &pass_info_;
};

}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif  // POLY_SINK_LAST_AXIS_H_