/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef SCHEDULE_TREE_UTIL_H
#define SCHEDULE_TREE_UTIL_H

#include <vector>
#include "isl.h"
#include "poly/scop.h"

namespace akg {
namespace ir {
namespace poly {

struct PromoteMarkerInfo {
  std::vector<std::string> markers;  // Insert markers from back to front
  int axis_pos{1};
};

isl::union_set CollectDomain(const isl::schedule_node &node);

isl::schedule_node MapDescendantTopDown(isl::schedule_node node,
                                        const std::function<isl::schedule_node(isl::schedule_node)> &fn);

void GetVisitedStmts(const isl::schedule_node &root);

bool IsThreadMappedMark(const isl::schedule_node &node);

bool IsAncestorMapToThread(const isl::schedule_node &curr_node);

template <typename IslNode>
std::vector<isl::schedule_node> CollectNode(isl::schedule sch) {
  std::vector<isl::schedule_node> res_nodes;
  auto GetIslNode = [&res_nodes](isl::schedule_node node) -> isl::schedule_node {
    if (node.isa<IslNode>()) {
      res_nodes.push_back(node);
    }
    return node;
  };
  sch.get_root().map_descendant_bottom_up(GetIslNode);
  return res_nodes;
}

std::vector<isl::schedule_node> FilterNode(std::vector<isl::schedule_node> nodes, const std::vector<isl::id> &filters);

// return a vector with mapping function
template <typename Otype, typename Itype>
std::vector<Otype> MapWithFunc(std::function<Otype(Itype)> func, const std::vector<Itype> &input) {
  std::vector<Otype> output;
  output.reserve(input.size());
  for (auto &i : input) {
    output.push_back(func(i));
  }
  output.shrink_to_fit();
  return output;
}

// return a vector that elements return true in function func
template <typename Func, typename T>
std::vector<T> FilterWithFunc(Func func, const std::vector<T> &input) {
  // TODO: function_traits files for static check

  std::vector<T> output;
  output.reserve(input.size());
  for (const auto &i : input) {
    if (func(i)) {
      output.push_back(i);
    }
  }
  output.shrink_to_fit();
  return output;
}

isl::schedule_node GenerateEmptyBandInRoot(isl::schedule_node &root);

bool ContainsDepth(isl::schedule_node &node, size_t depth);

int GetScheduleDepth(isl::schedule &root);

std::vector<isl::schedule_node> BandsContainingScheduleDepth(isl::schedule_node &root, size_t depth);

void CollectBandsOnTree(isl::schedule_node &root, std::vector<isl::schedule_node> &bands);

isl::schedule_node BandSplitAtDepth(isl::schedule_node &band, size_t depth);

std::vector<isl::schedule_node> BandsSplitAfterDepth(const std::vector<isl::schedule_node> &bands,
                                                     isl::schedule_node &root, size_t depth);

bool IsEqualNode(const isl::schedule_node node1, const isl::schedule_node node2);
isl::multi_union_pw_aff MapDomainToThread(const isl::schedule_node &node, MappingCfg *mapping_cfg,
                                          const UpaNodeMapping &upa_node_mapping);
isl::multi_union_pw_aff MapDomainAllWithType(const isl::schedule_node &node, MappingCfg *mapping_cfg,
                                             const UpaNodeMapping &upa_node_mapping, const std::string &map_type);

isl::map CreateMapIncreaseDim(isl::space space, unsigned dim);
bool IsSubsetForIncreaseDim(const isl::map access, size_t tensor_dim, size_t node_dim);
int GetLastAxis(const isl::schedule_node node, isl::union_map original_access,
                std::unordered_set<std::string> skip_tensors = {});

std::vector<isl::schedule_node> CollectFnNode(const std::function<bool(const isl::schedule_node &)> &fn,
                                              const isl::schedule_node &root);

isl::val GetInstancesBound(isl::schedule_node &node, isl::union_map ancestors_schedule, isl::val unroll_val);
isl::schedule_node UnrollByMarkOptions(isl::schedule_node &node, uint64_t unroll);

isl::map GetExtensionSpace(const isl::schedule_node &node, const isl::id &id);
isl::schedule_node InsertExtensionNodeBeforeOrAfter(const isl::schedule_node &node, const isl::id &id, bool before);

isl::schedule_node InsertMarkerForPromotedNode(
  const isl::schedule_node &orig_node, const std::unordered_map<std::string, PromoteMarkerInfo> &filter_marker_map);
std::string GetMarkerName(const isl::schedule_node &node, std::string find_name);

isl::union_set GetMappingFilterInfo(const isl::schedule_node node, MappingCfg *mapping_cfg,
                                    const std::unordered_map<std::string, MappingCfg *> &replace_cfg,
                                    const std::unordered_set<std::string> &non_repeated_idx = {});
isl::union_set GatherMappingsTo(const isl::schedule_node &root, MappingCfg *mapping_cfg,
                                const std::unordered_set<std::string> &non_repeated_idx = {});
std::unordered_set<std::string> GetNonRepeatedIdx(const MappingStrategyAxisMap &mapping_strategy);

bool ReuseTensorCluster(const TensorFootprintCluster &cluster, const isl::multi_union_pw_aff &outer_pw_aff);

std::vector<isl::schedule_node> CollectMarkNode(const isl::schedule_node &tree, const std::string &mark_tag);

std::unordered_map<std::string, std::string> GetMatmulTensorsName(ScopInfo &scop_info);

std::string GetTensorMark(const std::string &item, ScopInfo &scop_info);
isl::schedule_node AdjustAxisPosition(const isl::schedule_node &orig_node, const int orig_pos, const int new_pos);

isl::schedule_node InsertEmptyPermutableBand(const isl::schedule_node &orig_node);

int GetVectorizationTileSize(ScopInfo &scop_info);

isl::multi_val CheckAndGetMapSize(const isl::schedule_node &mapping_root, const isl::union_pw_aff_list &aff_list,
                                  MappingStrategyAxisMap &required_mapping_strategy, MappingCfg *mapping_cfg,
                                  const std::vector<int> &additional_tile_size = {});
isl::multi_union_pw_aff GetCurrentPartialSchedule(const isl::schedule_node_band &node, const bool is_promotion = false);
isl::schedule_node GetMarkerNode(const isl::schedule_node &orig_node, const std::string &marker_name);
isl::schedule_node DeleUselessMarker(const isl::schedule_node &orig_node,
                                     const std::unordered_set<std::string> &mark_names);

isl::schedule_node ReplaceMarker(const isl::schedule_node &orig_node, const std::string &orig_name,
                                 const std::string &replaced_name);

isl::schedule_node ReConstructScheduleTree(const isl::schedule_node &cur_node, const isl::schedule_node &orig_node,
                                           const isl::schedule_node &exit_node);
isl::schedule_node ReConstructChildScheduleTree(const isl::schedule_node &cur_node, const isl::schedule_node &orig_node,
                                                const isl::schedule_node &exit_node);
isl::schedule_node ReConstructSetOrSequenceNode(const isl::schedule_node &cur_node, const isl::schedule_node &orig_node,
                                                const isl::schedule_node &exit_node,
                                                const std::vector<size_t> &pos = {});
isl::schedule_node ReConstructBandNode(const isl::schedule_node &cur_node, const isl::schedule_node &orig_node);
bool IsContainBandNode(const isl::schedule_node &orig_node);

isl::schedule_node InsertMarkerForLoop(const isl::schedule_node &orig_node, const std::string &marker_name,
                                       const bool is_promotion = false, const int insert_pos = 0);

}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif
