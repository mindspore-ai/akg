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
#ifndef POLY_KEEP_OUTER_BAND_ORDER_H_
#define POLY_KEEP_OUTER_BAND_ORDER_H_

#include "poly/schedule_pass.h"
#include "poly/scop_info.h"

namespace akg {
namespace ir {
namespace poly {

/*
 * Reorder axes in the outer band to the same as input IR.
 */
class KeepOuterBandOrder : public SchedulePass {
 public:
  KeepOuterBandOrder(ScopInfo &info) : info_(info) { pass_name_ = __FUNCTION__; }
  ~KeepOuterBandOrder() {}

  virtual isl::schedule Run(isl::schedule sch);

 private:
  ScopInfo &info_;
};

}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif  // POLY_KEEP_OUTER_BAND_ORDER_H_