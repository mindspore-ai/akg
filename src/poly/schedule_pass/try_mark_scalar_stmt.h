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
#ifndef POLY_TRY_MARK_SCALAR_STMT_H_
#define POLY_TRY_MARK_SCALAR_STMT_H_

#include "poly/pass_info.h"
#include "poly/schedule_pass.h"

namespace akg {
namespace ir {
namespace poly {

/*
 * Mark each scalar statement with a "realize_UB" mark node. "root" should be
 * either a domain node or a filter node.
 *
 * First, check whether each statement in "root" is scalar. Each set of the
 * union set represented by "root" represents a statement. We determine a scalar
 * statement with "HasNoDims" function, checking whether a give "set" has dims.
 *
 * Next, check whether the subtree of "root" has permutable bands, and return
 * "root" if there are any permutable bands.
 *
 * Obtain the outermost permutable band, and this would go down to either a leaf
 * node or a sequence/set node.
 *
 * If it comes to a leaf node, "root" represents a single scalar statement. Insert
 * an empty band and mark this empty band with a "realize_UB" mark.
 *
 * If a sequence/set node is encountered, meaning "root" represents multiple
 * scalar statements. Mark each child recursively with a "realize_UB" mark.
 *
 * Return the original "root" in other cases.
 */
class TryMarkScalarStmt : public SchedulePass {
 public:
  TryMarkScalarStmt(PassInfo &pass_info) : pass_info_(pass_info) { pass_name_ = __FUNCTION__; };
  ~TryMarkScalarStmt(){};

  virtual isl::schedule Run(isl::schedule sch);

 private:
  PassInfo &pass_info_;

  bool SubtreeHasPermutableBands(const isl::schedule_node &node) const;

  isl::schedule_node InsertEmptyPermutableBand(isl::schedule_node node);
};

}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif  // POLY_TRY_MARK_SCALAR_STMT_H_
