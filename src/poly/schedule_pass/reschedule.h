/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#ifndef POLY_RESCHEDULE_H_
#define POLY_RESCHEDULE_H_

#include "poly/schedule_pass.h"

namespace akg {
namespace ir {
namespace poly {

struct PointBandInfo {
  isl::multi_union_pw_aff mupa;
  size_t n_member{0};
  bool permutable{false};
  std::vector<bool> coincident;
};

// data structure for recording tile band data
struct TileBandData {
  // flag indicating whether L0 tiled
  bool l0_tiled;
  // mark node of the tile band, if any
  isl::schedule_node mark;
  // mark node of conv_gemm, if any
  isl::schedule_node gemm_mark;
  // members of tile band
  unsigned int n_member;
  // schedule mupa
  isl::multi_union_pw_aff mupa;
  // permutable
  bool permutable;
  // coincident
  std::vector<bool> coincident;
  // ast build options
  isl::union_set ast_build_options;
};

class Reschedule : public SchedulePass {
 public:
  Reschedule(ScopInfo &scop_info, PassInfo &pass_info) : scop_info_(scop_info), pass_info_(pass_info) {
    pass_name_ = __FUNCTION__;
  };
  ~Reschedule() {}

  virtual isl::schedule Run(isl::schedule sch);
  isl::schedule RescheduleSerializeSccs(const isl::union_set &active_domain, const bool need_dist) const;

 private:
  static bool IsL1OrUbMark(const isl::schedule_node &node);
  static bool IsL0OrUbL0Mark(const isl::schedule_node &node);
  void CollectTileBandData(const isl::schedule_node &node, TileBandData *tile_band_data);
  static isl::schedule_node RetrieveTileBandData(isl::schedule_node node, TileBandData *tile_band_data);
  static isl::schedule_node RetrieveNodeList(isl::schedule_node node, const std::vector<isl::schedule_node> &node_list);
  static isl::schedule_node RetrieveAstBuildOptions(isl::schedule_node node, const isl::union_set &options);
  bool ValidateReorderedSchedule(const isl::schedule &new_schedule);
  isl::schedule_node TryRestoreStmtOrder(const isl::schedule_node &node, const std::vector<isl::id> &filter_total_order,
                                         const std::vector<std::vector<isl::id>> &filter_partial_order);
  isl::schedule_node ReschedulePreserveFilterOrder(const isl::schedule_node &node, const isl::union_set &active_domain,
                                                   const bool need_dist);
  static PointBandInfo SavePointBand(const isl::schedule_node &node);
  static isl::schedule_node SetPointBandInfo(isl::schedule_node node, const PointBandInfo &point_band_info);
  static isl::schedule_node RestorePointBandInfo(isl::schedule_node node, const PointBandInfo &point_band_info);
  isl::schedule_node RescheduleSchTree(const isl::schedule_node &root);
  isl::schedule_node RescheduleInnerBand(const isl::schedule_node &root);
  void Dump();

 private:
  ScopInfo &scop_info_;
  PassInfo &pass_info_;
  // for recording L1/UB tile band build options
  std::vector<isl::union_set> l1_build_options_;

  // for recording L0 tile band build options
  std::vector<isl::union_set> l0_build_options_;

  // for recording nodes along the path from root to L1/UB band
  std::vector<isl::schedule_node> node_list_0_;

  // for recording nodes along the path from L1/UB band to L0/UBL0 band
  std::vector<isl::schedule_node> node_list_1_;

  // for recording nodes along the path from L0/UBL0 band to point band
  std::vector<isl::schedule_node> node_list_2_;
};

}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif  // POLY_RESCHEDULE_H_
