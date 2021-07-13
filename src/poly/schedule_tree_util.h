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

#ifndef SCHEDULE_TREE_UTIL_H
#define SCHEDULE_TREE_UTIL_H

#include <vector>
#include "isl.h"
#include "poly/scop.h"

namespace akg {
namespace ir {
namespace poly {

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
isl::union_pw_aff_list GetUPAList(const isl::schedule_node &node, isl::multi_union_pw_aff &partial_schedule,
                                  const bool is_promotion, const bool need_reverse);
isl::schedule_node MapInnerDimToThreads(const isl::schedule_node &node, MappingCfg *mapping_cfg, Mapping &mapping,
                                        const bool is_promotion, const bool need_reverse);
isl::schedule_node AnalysisNodeAndInsertMapFilter(const isl::schedule_node &node, const bool is_promotion,
                                                  isl::union_pw_aff_list upa_list, MappingCfg *mapping_cfg,
                                                  Mapping &mapping,
                                                  std::unordered_map<size_t, size_t> map_idx_shift = {});
isl::schedule_node InsertMapFilter(const isl::schedule_node &node, const bool is_promotion, Mapping &mapping);
isl::schedule_node CheckMapSizeAndApplyTile(const isl::schedule_node &thread_root,
                                            const isl::union_pw_aff_list &aff_list, MappingCfg *mapping_cfg,
                                            const bool need_reverse,
                                            std::unordered_map<int, std::string> custom_mapping = {});

bool IsEqualNode(const isl::schedule_node node1, const isl::schedule_node node2);
isl::multi_union_pw_aff MapDomainToThread(const isl::schedule_node &node, MappingCfg *mapping_cfg,
                                          const UpaNodeMapping &upa_node_mapping);
isl::multi_union_pw_aff MapDomainAllWithType(const isl::schedule_node &node, MappingCfg *mapping_cfg,
                                             const UpaNodeMapping &upa_node_mapping, const std::string &map_type);

isl::map CreateMapIncreaseDim(isl::space space, unsigned dim);

std::vector<isl::schedule_node> CollectFnNode(const std::function<bool(const isl::schedule_node &)> &fn,
                                              const isl::schedule_node &root);

isl::val GetInstancesBound(isl::schedule_node &node, isl::union_map ancestors_schedule, isl::val unroll_val);
isl::schedule_node UnrollByMarkOptions(isl::schedule_node &node, uint64_t unroll);

isl::map GetExtensionSpace(const isl::schedule_node &node, const isl::id &id);
isl::schedule_node InsertExtensionNodeBeforeOrAfter(const isl::schedule_node &node, const isl::id &id, bool before);

isl::schedule InsertMarkerForThreadGroup(const isl::schedule &sch, const std::string &write_name,
                                         const std::string &marker_name);
std::string GetMarkerName(const isl::schedule_node &node, std::string find_name);

isl::union_set GetBlockMappingFilterInfo(const isl::schedule_node node, MappingCfg *block_cfg,
                                         std::unordered_map<std::string, MappingCfg *> replace_cfg);
isl::union_set GatherMappingsTo(const isl::schedule_node &root, MappingCfg *cfg);

bool ReuseTensorCluster(const TensorFootprintCluster &cluster, const isl::multi_union_pw_aff &outer_pw_aff);

isl::schedule_node CollectMarkNodeOnPromotion(const isl::schedule_node &root, const std::string &mark);

std::unordered_map<std::string, std::string> GetMatmulTensorsName(ScopInfo &scop_info);

bool IsTensorAB(const std::string &item, ScopInfo &scop_info);

isl::schedule_node AdjustConvScheduleTreeStructure(const isl::schedule_node &orig_node, const bool is_promotion = true);

isl::union_pw_aff_list GetPrefixPartialSchedule(const isl::multi_union_pw_aff partial_schedule,
                                                const isl::schedule_node node, const bool need_reverse);

}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif
