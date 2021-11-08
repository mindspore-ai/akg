/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef BAND_NODE_ANALYSIS_H
#define BAND_NODE_ANALYSIS_H

#include "poly/scop_info.h"

namespace akg {
namespace ir {
namespace poly {

class AnalyzeBandNode {
 public:
  AnalyzeBandNode(const isl::schedule &sch, ScopInfo &scop_info) : sch_(sch), scop_info_(scop_info) {
    target_ = scop_info.user_config_.GetTarget();
  }
  ~AnalyzeBandNode() = default;

  void Run();

 private:
  void CollectStmtInfo();
  void AnalyzeOuterBandTemplate();
  void ShowBandInfo();
  void AnalyzeLastAxis();
  void DetermineTemplateOfBand(BandNode *bn);
  void SetTensorLastAxis(const isl::schedule_node &node, int index);
  int GetCoalescedAccess(const isl::schedule_node &node);

  std::string target_;
  const isl::schedule &sch_;
  ScopInfo &scop_info_;
  std::unordered_map<isl::id, std::pair<std::string, ReduceDirection>, isl::IslIdIslHash> stmt_info_;

};

}  // namespace poly
}  // namespace ir
}  // namespace akg
#endif  // BAND_NODE_ANALYSIS_H