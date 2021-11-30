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

#ifndef POLY_OPRATOR_MAPPING_STRATEGY_H_
#define POLY_OPRATOR_MAPPING_STRATEGY_H_

#include "poly/schedule_pass.h"

namespace akg {
namespace ir {
namespace poly {

class OperatorMappingStrategy {
 public:
  explicit OperatorMappingStrategy(ScopInfo &scop_info) : scop_info_(scop_info) {}
  explicit OperatorMappingStrategy(ScopInfo &scop_info, MappingCfg *mapping_cfg, int filter_pos = 0,
                                   bool is_thread_mapping = true, bool is_promotion_mapping = false)
      : scop_info_(scop_info),
        mapping_cfg_(mapping_cfg),
        band_index_(filter_pos),
        is_thread_mapping_(is_thread_mapping),
        is_promotion_mapping_(is_promotion_mapping) {
    current_outer_bn_ = scop_info_.analysis_result_.GetOuterBandNode(band_index_);
    CHECK(mapping_cfg != nullptr) << "mapping config is null";
  }
  ~OperatorMappingStrategy() {}

  virtual size_t MapThreadHelper(isl::schedule_node &thread_root);
  virtual isl::schedule_node MapBlockHelper(const isl::schedule_node &orig_node);
  isl::schedule_node MapThreadBlockHelper(const isl::schedule_node &orig_node);

  // Core mapping function
  isl::schedule_node MapDimToThreadsBlocks(const isl::schedule_node &node);
  isl::schedule_node AnalysisNodeAndInsertMapFilter(const isl::schedule_node &node,
                                                    const isl::union_pw_aff_list &upa_list);
  isl::schedule_node InsertMapFilter(const isl::schedule_node &node);
  isl::schedule_node CheckMapSizeAndApplyTile(const isl::schedule_node &mapping_root,
                                              const isl::union_pw_aff_list &aff_list);
  isl::union_pw_aff_list GetUpaList(const isl::schedule_node &node, isl::multi_union_pw_aff &partial_schedule);
  isl::union_pw_aff_list GetPrefixPartialSchedule(const isl::multi_union_pw_aff &partial_schedule,
                                                  const isl::schedule_node &node);

  // Modify the mapping strategy
  std::string SetOneConfigForMulAxis(const isl::schedule_node &node, const int orig_total_cfg,
                                     const std::unordered_set<int> &excluded_axis_pos = {});
  void SetRequiredMappingCfg(const isl::schedule_node &node, int start_pos = -1, int end_pos = INT_MAX);
  void InitRepeatedMappingConfig();
  void SetRepeatedMappingStrategy(const std::string &mapping_str);
  MappingCfg *GetRepeatedReplaceMappingConfig(const isl::schedule_node &node, const std::string &replace_mapping_name);
  void ReadjustRequireddMappingStrategy(const bool is_repeated_mapping, const std::string &repeated_mapping_idx = "",
                                        const std::string &mapping_str = "");

  // The mapping strategy for each axis.
  MappingStrategyAxisMap required_mapping_strategy_;
  // Store the information of the corresponding filter after mapping.
  MappingScheduleInfoMap mapping_sch_info_map_;

  // The mapping relationship between axis and threadIdx/blockIdx is many-to-one.
  std::unordered_map<std::string, std::unordered_set<int>> repeated_mapping_cfg_axis_;
  // The mapping relationship between axis and threadIdx/blockIdx is one-to-one.
  std::unordered_map<std::string, std::unordered_set<int>> non_repeated_mapping_cfg_axis_;

  bool is_insert_filter_{true};    // Whether to insert the filter node generated after the mapping.
  bool is_need_reverse_{true};     // Whether to start mapping from the inner axis.
  bool is_set_config_zero_{true};  // Whether to set the redundant configuration to 0 when mapping.
  OuterBandNode *current_outer_bn_;

 protected:
  ScopInfo &scop_info_;
  MappingCfg *mapping_cfg_{nullptr};
  int band_index_;
  bool is_thread_mapping_;
  bool is_promotion_mapping_;
};

class ReduceMappingStrategy : public OperatorMappingStrategy {
 public:
  explicit ReduceMappingStrategy(ScopInfo &scop_info, MappingCfg *mapping_cfg, int filter_pos = 0,
                                 bool is_thread_mapping = true, bool is_promotion_mapping = false)
      : OperatorMappingStrategy(scop_info, mapping_cfg, filter_pos, is_thread_mapping, is_promotion_mapping) {}
  ~ReduceMappingStrategy() {}

  size_t MapThreadHelper(isl::schedule_node &thread_root);

  bool NeedAtomicAdd(const isl::schedule_node_band &band, size_t n_block_map);
  void MarkAtomicAddTensor(const isl::schedule_node_band &band);
  void UpadateSplitMappingStatregy(const int split_pos);
  size_t GetFinalMappingThreadNumber(isl::schedule_node &node, const size_t n_thread_map);

 private:
  isl::schedule_node InsertReduceExtension(const isl::schedule_node &node);
};

class BatchMatmulMappingStrategy : public OperatorMappingStrategy {
 public:
  explicit BatchMatmulMappingStrategy(ScopInfo &scop_info, MappingCfg *mapping_cfg, int filter_pos = 0,
                                      bool is_thread_mapping = true, bool is_promotion_mapping = false)
      : OperatorMappingStrategy(scop_info, mapping_cfg, filter_pos, is_thread_mapping, is_promotion_mapping) {}
  ~BatchMatmulMappingStrategy() {}

  size_t MapThreadHelper(isl::schedule_node &thread_root);
};

class ConvMappingStrategy : public OperatorMappingStrategy {
 public:
  explicit ConvMappingStrategy(ScopInfo &scop_info) : OperatorMappingStrategy(scop_info) {}
  ~ConvMappingStrategy() {}

  isl::schedule MoveKernelHWBand(const isl::schedule &sch);
};

}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif  // POLY_OPRATOR_MAPPING_STRATEGY_H_