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
#ifndef POLY_SET_ALL_COINCIDENCE_H_
#define POLY_SET_ALL_COINCIDENCE_H_

#include "poly/schedule_pass.h"

namespace akg {
namespace ir {
namespace poly {

/*
 * Sometimes, coincident is set to `0` for some axes that can actually be parallelised in computed schedule tree.
 * Since we have no idea why these cases happen, we offer such transfrom to set all coincident to `1`.
 * Please be careful to do such transfrom since it may cause some incorrect result.
 */
class SetAllCoincidence : public SchedulePass {
 public:
  SetAllCoincidence() { pass_name_ = __FUNCTION__; };
  ~SetAllCoincidence(){};

  virtual isl::schedule Run(isl::schedule sch);
};

}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif  // POLY_SET_ALL_COINCIDENCE_H_
