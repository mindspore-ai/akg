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
#ifndef POLY_RESET_COINCIDENCE_OF_REDUCE_H_
#define POLY_RESET_COINCIDENCE_OF_REDUCE_H_

#include "poly/schedule_pass.h"

namespace akg {
namespace ir {
namespace poly {

/*
 * Reset ths coincidence of reduce axis in partial schedule to 0 if the original coincidence is 1.
 * This transform can prevent reduce axes from being parallelled.
 */
class ResetCoincidenceOfReduce : public SchedulePass {
 public:
  ResetCoincidenceOfReduce(ScopInfo &scop_info, PassInfo &pass_info) : scop_info_(scop_info), pass_info_(pass_info) {
    pass_name_ = __FUNCTION__;
  };
  ~ResetCoincidenceOfReduce(){};

  virtual isl::schedule Run(isl::schedule sch);

 private:
  ScopInfo &scop_info_;
  PassInfo &pass_info_;

  bool IsStmtScheduleContainsReduceAxis(const isl::pw_aff &stmt,
                                        const std::unordered_set<std::string> &reduce_axis_list);

  bool IsDimScheduleContainsReduceAxis(const isl::union_pw_aff &schedule);
};

}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif  // POLY_RESET_COINCIDENCE_OF_REDUCE_H_
