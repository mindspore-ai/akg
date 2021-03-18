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
#ifndef POLY_REORDER_INNER_BAND_H_
#define POLY_REORDER_INNER_BAND_H_

#include "poly/schedule_pass.h"

namespace akg {
namespace ir {
namespace poly {

/*
 * Reorder the members of the leaf-band partial schedule (if it is permutable)
 * such that loop vars that appear in "if" conditions are the outer loops.
 * This aims to promote the "if" condition to the outermost loop, and maximize
 * the size of unconditional vectorized computation.
 */
class ReorderInnerBand : public SchedulePass {
 public:
  ReorderInnerBand(const CondVarsMap &cond_vars) : cond_vars_(cond_vars) { pass_name_ = __FUNCTION__; };
  ~ReorderInnerBand(){};

  virtual isl::schedule Run(isl::schedule sch);

 private:
 CondVarsMap cond_vars_;
};

}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif  // POLY_REORDER_INNER_BAND_H_
