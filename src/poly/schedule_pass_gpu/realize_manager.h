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

#ifndef INSERT_REALIZE_H_
#define INSERT_REALIZE_H_

#include "poly/schedule_pass.h"

namespace akg {
namespace ir {
namespace poly {

class RealizeManager : public SchedulePass {
 public:
  explicit RealizeManager(PassInfo &pass_info, ScopInfo &scop_info) : pass_info_(pass_info), scop_info_(scop_info) {
    pass_name_ = __FUNCTION__;
  };
  ~RealizeManager() {}

  virtual isl::schedule Run(isl::schedule sch);

  isl::schedule_node InsertRealize(const isl::schedule_node &root);

 private:
  PassInfo &pass_info_;
  ScopInfo &scop_info_;
  std::set<std::string> names_set_{};

  isl::id GetRealizeId(const isl::schedule_node &node, std::string tensor_name) const;

  isl::map GetExtensionSpace(const isl::schedule_node &node, const std::string tensor_name);

  isl::schedule_node InsertExtensionNodeBefore(const isl::schedule_node &node, const std::string tensor_name);

  std::string GetFilterName(const isl::schedule_node_filter &filter_node);

  std::string GetTensorName(const isl::schedule_node_filter &filter_node);

  isl::schedule_node BreadthFirstTopDown(const isl::schedule_node &node, bool &end);
};

}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif  // INSERT_REALIZE_H_
