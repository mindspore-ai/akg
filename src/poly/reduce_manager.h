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
  ReduceManager() {}
  ~ReduceManager() {}

  bool SplitReduceStatements(isl::schedule_node &node, isl::union_set reduce_statements, isl::union_map dependences,
                             const bool split_reduce_dependent);
  isl::union_set GetReduceStatements(isl::union_set domain, isl::union_map reduce_statement_map,
                                     StatementMap all_statements);

 private:
  isl::schedule_node ReorderStatements(const isl::schedule_node &node, isl::union_set before, isl::union_set after,
                                       const bool split_reduce_dependent);
  bool AreSequentialStatements(isl::union_set first_statements, isl::union_set second_statements,
                               isl::union_map dependences);
};

}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif  // POLY_REDUCTION_MANAGER_H_