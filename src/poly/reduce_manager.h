/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef POLY_REDUCTION_MANAGER_H_
#define POLY_REDUCTION_MANAGER_H_

#include "isl.h"

#include <memory>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <stack>

namespace akg {
namespace ir {
namespace poly {

class ReduceManager {
 public:
  ReduceManager(PassInfo &pass_info, ScopInfo &scop_info) : pass_info_(pass_info), scop_info_(scop_info) {}
  ~ReduceManager() {}

  bool SplitReduceStatements(isl::schedule_node &node, isl::union_set reduce_statements, isl::union_map dependences);
  isl::union_set GetCurrentNodeReduceStatements(const isl::schedule_node node, ReduceTensorInfoMap &all_reduce_map,
                                                const bool need_delete_reduce = true);
  // Check whether the reduce operator is included, and split the reduce statement into a separate filter.
  isl::schedule DetectAndMarkReduce(const isl::schedule &sch);
  isl::schedule InsertReduceMarker(const isl::schedule &sch);

 private:
  isl::union_set GetReduceStatements(isl::union_set domain, isl::union_map reduce_statement_map,
                                     StatementMap all_statements);
  isl::schedule_node ReorderStatements(const isl::schedule_node &node, isl::union_set before, isl::union_set after);
  bool AreSequentialStatements(isl::union_set first_statements, isl::union_set second_statements,
                               isl::union_map dependences);
  // After splitting the reduce fusion operator, reschedule all the filters, mainly because the reduce statement affects
  // other statements after the fusion.
  isl::schedule RescheduleForReduce(const isl::schedule &sch);
  size_t GetReduceId() const;

  PassInfo &pass_info_;
  ScopInfo &scop_info_;
};

}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif  // POLY_REDUCTION_MANAGER_H_