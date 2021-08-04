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

#ifndef POLY_MAPPING_H_
#define POLY_MAPPING_H_

#include "poly/schedule_pass.h"

namespace akg {
namespace ir {
namespace poly {

/*
 * Mapping the outer band to blocks and threads to enable parallelism in Gpu.
 */
class MappingOuterBand : public SchedulePass {
 public:
  MappingOuterBand(PassInfo &pass_info, ScopInfo &scop_info) : pass_info_(pass_info), scop_info_(scop_info) {
    pass_name_ = __FUNCTION__;
  };
  ~MappingOuterBand() {}

  using RoadMap = std::vector<std::pair<isl::schedule_node, size_t>>;

  virtual isl::schedule Run(isl::schedule sch);

  isl::schedule DoThreadMapping(const isl::schedule &sch);

  isl::schedule DoBlockMapping(const isl::schedule &sch);

  isl::schedule DoCustomMapping(const isl::schedule &sch);
  isl::schedule_node MapCustomHelper(const isl::schedule_node orig_node, const bool is_inner, MappingCfg *mapping_cfg);
  isl::schedule_node InsertCustomMappingFilter(const isl::schedule_node &node, isl::union_pw_aff_list upa_list,
                                               MappingCfg *mapping_cfg, Mapping &mapping,
                                               std::unordered_map<int, std::string> custom_mapping,
                                               std::unordered_set<std::string> outer_mapping_cfg = {});

  size_t NumMappedDescendant(const RoadMap &thread_roadmap, const isl::schedule_node parent);

  bool CanBeMappedToThread(const isl::schedule_node node, const RoadMap &thread_record);

  isl::schedule_node FillRemainingThreads(isl::schedule_node &node, size_t begin);

  isl::schedule_node MapSequenceNode(const isl::schedule_node &orig_node, const RoadMap &thread_record);

  /*
   * Functions related to synchronization.
   */
  isl::schedule_node DoThreadSynchronization(const isl::schedule_node &node,
                                             const std::vector<MappingCfg *> &other_mapping_cfg = {});

  // preparation for synchronization
  isl::multi_union_pw_aff MapDomainToWarp(const isl::schedule_node &nod, MappingCfg *mapping_cfg,
                                          isl::multi_union_pw_aff domain_threads);
  bool IsOuterBandWithNoCoincident(const isl::schedule_node &node);

  // strategies for determine optimal sync position between mapped threads in an isl sequence node.
  std::vector<Synchronization> DetermineOptSyncPos(SyncCandidate *head, int start = 0);
  SyncCandidate *InitSyncLinkedList(const isl::schedule_node &seq_node, const isl::multi_union_pw_aff &domain_to_thread,
                                    const isl::multi_union_pw_aff &domain_to_warp);
  SyncCandidate *CountSyncNumberAmongLoop(SyncCandidate *head);
  int GetBestSyncStartPoint(bool is_outer);

 private:
  PassInfo &pass_info_;
  ScopInfo &scop_info_;
};

}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif  // POLY_MAPPING_H_