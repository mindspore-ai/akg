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
#ifndef POLY_CHANGE_MARKNODE_POSITION_H_
#define POLY_CHANGE_MARKNODE_POSITION_H_

#include "poly/schedule_pass.h"
#include <unordered_set>

namespace akg {
namespace ir {
namespace poly {

/*
 * "with" stmt aims to work around the irregular problem.
 * By default, the "realize_UB" mark is on the outer band. However, for tensor-of-tensor,
 * the intermediate tensor may be too large if realized in the outermost scope.
 * To narrow down the scope, we move "realize_UB" mark to the filter node.
 * If all filter nodes of the band are "with" stmts, we remove the outer "realize_UB" mark.
 */
class ChangeMarkNodePosition : public SchedulePass {
 public:
  ChangeMarkNodePosition(const std::unordered_set<std::string> &with_stmts_ids) : with_stmts_ids_(with_stmts_ids) {
    pass_name_ = __FUNCTION__;
  };
  ~ChangeMarkNodePosition(){};

  virtual isl::schedule Run(isl::schedule sch);

 private:
  std::unordered_set<std::string> with_stmts_ids_;
};

}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif  // POLY_CHANGE_MARKNODE_POSITION_H_
