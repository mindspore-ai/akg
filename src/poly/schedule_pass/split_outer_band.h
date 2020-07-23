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
#ifndef POLY_SPLIT_OUTER_BAND_H_
#define POLY_SPLIT_OUTER_BAND_H_

#include "poly/schedule_pass.h"

namespace akg {
namespace ir {
namespace poly {

/*
 * Split the consecutive parallelled nodes (i.e. coincident equals to 1) from the most-outer band,
 * resulting in an outer band with an inner band containing all untileable nodes as its child.
 * Note that this transfrom can prevent shift case when post-fusion exists in dynamic shape.
 */
class SplitOuterBand : public SchedulePass {
 public:
  SplitOuterBand() { pass_name_ = __FUNCTION__; };
  ~SplitOuterBand(){};

  virtual isl::schedule Run(isl::schedule sch);
};

}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif  // POLY_SPLIT_OUTER_BAND_H_
