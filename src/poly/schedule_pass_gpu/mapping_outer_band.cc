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

#include "mapping_outer_band.h"
#include "poly/schedule_pass_gpu/operator_mapping_strategy.h"

#include <numeric>

#include "poly/schedule_tree_util.h"
#include "poly/sync_manager.h"
#include "poly/scop.h"
#include "poly/gpu_emit/gpu_isl_emitter.h"

namespace akg {
namespace ir {
namespace poly {

isl::schedule_node MappingOuterBand::DoThreadSynchronization(const isl::schedule_node &node,
                                                             const std::vector<MappingCfg *> other_mapping_cfg) {
  auto sync_node = node;
  auto sync_manager = scop_info_.sync_manager_;
  auto thread_cfg = scop_info_.user_config_.GetThreadConfig();
  CHECK(thread_cfg != nullptr) << "thread config is null";

  // Step 1. prepare info
  bool is_outer = IsOuterBandWithNoCoincident(node);
  auto domain_thread = MapDomainToThread(node, thread_cfg, scop_info_.upa_node_mapping_);
  for (size_t i = 0; i < other_mapping_cfg.size(); ++i) {
    auto mapping_cfg = other_mapping_cfg[i];
    CHECK(mapping_cfg != nullptr) << "mapping config is null";
    auto domain_other_mapping = MapDomainToThread(node, mapping_cfg, scop_info_.upa_node_mapping_);
    domain_thread = domain_thread.union_add(domain_other_mapping);
  }
  auto domain_node = CollectDomain(node);
  bool sub_set = domain_node.is_subset(domain_thread.domain());
  CHECK(sub_set) << "There are remaining domains that have not been mapped to threadID";

  auto domain_warp = MapDomainToWarp(node, thread_cfg, domain_thread);

  // Step 2. construct a linked list for all nodes in the input sequence node
  auto head = InitSyncLinkedList(node, domain_thread, domain_warp);

  // Step 3. Use "fewest synchronization number first" strategy to determine the
  //         optimization sync position in the sequence node.
  head = CountSyncNumberAmongLoop(head);
  auto start = GetBestSyncStartPoint(is_outer);
  auto all_syncs = DetermineOptSyncPos(head, start);
  std::sort(all_syncs.begin(), all_syncs.end(),
            [](Synchronization s1, Synchronization s2) { return s1.pos >= s2.pos; });

  // Step 4. Insert sync node (extension and filter) in the sequence node
  for (const auto &sync : all_syncs) {
    auto target = sync_node.child(sync.pos).child(0);
    sync_node = sync_manager.InsertExtensionNode(target, sync.level, true).parent().parent();
  }

  auto next = head->next.release();
  delete next;

  return sync_node;
}

std::vector<Synchronization> MappingOuterBand::DetermineOptSyncPos(SyncCandidate *head, int start) {
  std::vector<Synchronization> all_syncs;
  auto start_node = head->NextNCandidate(start);

  auto SplitList = [&start_node, &all_syncs](SyncLevel level) {
    if (level == SyncLevel::EMPTY) {
      return;
    }
    auto cur = start_node;
    while (cur) {
      auto opt = cur->GetOptimalSyncPos(level);
      cur = opt.first;
      bool exit = opt.second == 0;
      auto new_sync = Synchronization(level, cur->idx);
      for (const auto &old_sync : all_syncs) {
        if (new_sync.IsEqual(old_sync)) {
          exit = true;
          break;
        }
      }
      if (exit) {
        break;
      }
      all_syncs.emplace_back(new_sync);
    }
  };
  SplitList(SyncLevel::BLOCK);
  SplitList(SyncLevel::WARP);
  return all_syncs;
}

SyncCandidate *MappingOuterBand::InitSyncLinkedList(const isl::schedule_node &seq_node,
                                                    const isl::multi_union_pw_aff &domain_to_thread,
                                                    const isl::multi_union_pw_aff &domain_to_warp) {
  auto context_params = scop_info_.analysis_result_.GetContextParams();
  auto dependency = pass_info_.dependences_;
  auto seq_len = static_cast<int>(seq_node.n_children());
  auto root = std::unique_ptr<SyncCandidate>(new (std::nothrow) SyncCandidate(-1, seq_len));
  CHECK(root) << "memory alloc fail.";
  std::vector<SyncCandidate *> cands;
  auto cur = root.get();
  for (auto i = 0; i < seq_len; ++i) {
    auto sync_node = std::unique_ptr<SyncCandidate>(new (std::nothrow) SyncCandidate(i, seq_len));
    CHECK(sync_node) << "memory alloc fail.";
    sync_node->domain = CollectDomain(seq_node.child(i).child(0));
    cur->next = std::move(sync_node);
    cur = cur->next.get();
    cands.emplace_back(cur);
  }
  cur->next = std::move(root->next);  // link end and start

  for (auto cand : cands) {
    auto DetermineSyncLevel = [seq_node, dependency, context_params, domain_to_thread, domain_to_warp,
                               &cand](SyncCandidate *node) {
      auto new_dep = dependency.intersect_domain(cand->domain);
      new_dep = new_dep.intersect_range(node->domain);
      if (new_dep.is_empty()) {
        cand->InsertSyncBetween(node, Synchronization(SyncLevel::EMPTY));
      } else {
        new_dep = new_dep.intersect_params(context_params);
        if (new_dep.is_subset(new_dep.eq_at(domain_to_thread))) {
          cand->InsertSyncBetween(node, Synchronization(SyncLevel::EMPTY));
        } else if (new_dep.is_subset(new_dep.eq_at(domain_to_warp))) {
          cand->InsertSyncBetween(node, Synchronization(SyncLevel::WARP));
        } else {
          cand->InsertSyncBetween(node, Synchronization(SyncLevel::BLOCK));
        }
      }
    };
    cand->ForEachCandidateTopDown(DetermineSyncLevel);
  }

  return cur->next.get();
}

SyncCandidate *MappingOuterBand::CountSyncNumberAmongLoop(SyncCandidate *head) {
  head->ForEachCandidateTopDown([](SyncCandidate *n1) {
    auto accum_block_count = 0;
    auto accum_warp_count = 0;
    n1->ForEachCandidateTopDown([&n1, &accum_block_count, &accum_warp_count](SyncCandidate *n2) {
      auto block_count = n1->GetNumOfSyncBetween(n2, SyncLevel::BLOCK);
      auto warp_count = n1->GetNumOfSyncBetween(n2, SyncLevel::WARP);
      warp_count = std::max(warp_count - block_count, 0);

      if (accum_block_count < block_count) {
        accum_block_count = block_count;
      }
      n1->num_block_sync_to[n2] = accum_block_count;

      if (accum_warp_count < warp_count) {
        accum_warp_count = warp_count;
      }
      n1->num_warp_sync_to[n2] = accum_warp_count;
    });
  });
  return head;
}

int MappingOuterBand::GetBestSyncStartPoint(bool is_outer) {
  // When there is only one outer-band, which is the most common case, it is best to start from the beginning;
  // otherwise, we need a strategy to determine the best start point.
  return 0;
}

isl::multi_union_pw_aff MappingOuterBand::MapDomainToWarp(const isl::schedule_node &node, MappingCfg *mapping_cfg,
                                                          isl::multi_union_pw_aff domain_threads) {
  isl::space space = isl::space(node.ctx(), 0);
  auto block_space = space.add_named_tuple_id_ui(isl::id(node.ctx(), SYNC_BLOCK), mapping_cfg->bound);
  auto bspace = block_space;
  auto warp_space = space.add_named_tuple_id_ui(isl::id(node.ctx(), SYNC_WARP), 1);

  auto block_aff = isl_aff_zero_on_domain(isl_local_space_from_space(bspace.release()));
  isl::aff aff = isl::manage(block_aff);

  auto identity = isl::multi_aff::identity(block_space.map_from_set());
  for (int i = mapping_cfg->bound - 1; i >= 0; --i) {
    auto bi = mapping_cfg->GetAt(i);
    aff = aff.scale(isl::val(node.ctx(), bi.second));
    aff = aff.add(identity.get_aff(i));
  }

  aff = aff.scale_down(isl::val(node.ctx(), WARP_SIZE)).floor();
  auto map_space = block_space.product(warp_space).unwrap();
  isl::multi_aff thread_warp = isl::multi_aff(map_space, isl::aff_list(aff));
  return domain_threads.apply(thread_warp);
}

bool MappingOuterBand::IsOuterBandWithNoCoincident(const isl::schedule_node &node) {
  int depth = node.get_tree_depth();
  isl::schedule_node ancestor_node;

  for (int i = 0; i < depth; ++i) {
    ancestor_node = node.ancestor(depth - i);
    if (auto band = ancestor_node.as<isl::schedule_node_band>()) {
      auto n_coincident = CountConsecutiveCoincident(band);
      if (band.n_member() > n_coincident) {
        return true;
      }
    }
    if (ancestor_node.isa<isl::schedule_node_sequence>()) {
      return false;
    }
  }

  return false;
}

isl::schedule_node MappingOuterBand::FillRemainingThreads(isl::schedule_node &node, size_t begin) {
  auto thread_cfg = scop_info_.user_config_.GetThreadConfig();
  CHECK(thread_cfg != nullptr) << "threadconfig is null";
  size_t end = thread_cfg->bound;
  if (begin == end) {
    return node;
  }

  CHECK(node.isa<isl::schedule_node_filter>()) << "The child of set or sequence must be a filter!";
  node = node.child(0);

  isl::schedule_node_band band_node = node.as<isl::schedule_node_band>();
  Mapping mapping;
  auto after_map_node = MapInnerDimToThreads(band_node, false, thread_cfg, mapping, false);
  bool is_tiled = GetMarkerName(after_map_node, THREAD_MARKER).empty();
  after_map_node = is_tiled ? after_map_node.child(0) : after_map_node;
  scop_info_.upa_node_mapping_.emplace_back(std::make_pair(after_map_node, mapping));
  return after_map_node;
}

size_t MappingOuterBand::NumMappedDescendant(const RoadMap &thread_roadmap, const isl::schedule_node parent) {
  size_t max_thread_size = 0;
  for (const auto &record : thread_roadmap) {
    auto child_node = record.first;
    auto thread_size = record.second;
    bool is_child = IsEqualNode(parent, child_node);
    while (!is_child && child_node && child_node.has_parent()) {
      child_node = child_node.parent();
      is_child = IsEqualNode(parent, child_node);
    }
    if (is_child) {
      max_thread_size = std::max(max_thread_size, thread_size);
    }
  }
  return max_thread_size;
}

bool MappingOuterBand::CanBeMappedToThread(const isl::schedule_node node, const RoadMap &thread_record) {
  auto IsInnerMostBand = [this, &thread_record](const isl::schedule_node node) {
    auto band = node.as<isl::schedule_node_band>();
    return band && band.permutable() && NumMappedDescendant(thread_record, node) == 0;
  };

  auto HasMapped = [&thread_record](const isl::schedule_node node) -> bool {
    for (size_t i = 0; i < thread_record.size(); ++i) {
      if (IsEqualNode(thread_record[i].first, node)) {
        return true;
      }
    }
    return false;
  };

  if (!IsInnerMostBand(node)) {
    return false;
  }

  auto band = node.as<isl::schedule_node_band>();

  // make sure a band node in a sequence node only be mapped when all its siblings can be mapped together
  if (band.ancestor(2) && band.ancestor(2).isa<isl::schedule_node_sequence>()) {
    auto seq = band.ancestor(2).as<isl::schedule_node_sequence>();
    for (size_t i = 0; i < seq.n_children(); ++i) {
      auto filter = seq.child(i);
      if (filter.child(0).isa<isl::schedule_node_mark>()) {
        continue;
      }
      if (!IsInnerMostBand(filter.child(0)) && !HasMapped(filter)) {
        return false;
      }
    }
  }
  return true;
}

isl::schedule_node MappingOuterBand::MapSequenceNode(const isl::schedule_node &orig_node,
                                                     const RoadMap &thread_record) {
  // deal with band that has children mapped to threads
  auto node = orig_node;
  auto num_children = node.n_children();
  int start_node_depth = node.get_tree_depth();
  for (size_t i = 0; i < num_children; ++i) {
    isl::schedule_node node_child = node.child(i);
    for (const auto &record : thread_record) {
      auto child_node = record.first;
      auto thread_size = record.second;
      if (child_node.has_parent() && child_node.parent().isa<isl::schedule_node_filter>()) {
        child_node = child_node.parent();
      }
      bool is_child = IsEqualNode(node_child, child_node);
      if (is_child) {
        node_child = FillRemainingThreads(node_child, thread_size);
        node = node_child.ancestor(node_child.get_tree_depth() - start_node_depth);
        break;
      }
    }
  }

  return node;
}

isl::schedule MappingOuterBand::DoThreadMapping(const isl::schedule &sch) {
  auto final_schedule = sch;
  auto thread_cfg = scop_info_.user_config_.GetThreadConfig();
  CHECK(thread_cfg != nullptr) << "thread config is null";
  if (thread_cfg->bound < 1) {
    return final_schedule;
  }

  // Step 1. Find inner-most permutable band to map threads.
  RoadMap thread_record;
  bool is_reduce_stmt = false;

  auto MapFromInner = [&thread_record, &is_reduce_stmt, thread_cfg,
                       this](isl::schedule_node node) -> isl::schedule_node {
    // batch matmul operator
    bool is_bmm_stmt = false;
    if (scop_info_.user_config_.GetEnableTensorCoreUsePoly()) {
      if (node.has_parent() && !GetMarkerName(node.parent(), SKIP_MARKER).empty()) {
        node = node.parent().del();
        return node;
      }
      if ((node.has_parent() && !GetMarkerName(node.parent(), MAP_TO_WARP).empty())) {
        node = node.parent().del();
        is_bmm_stmt = true;
      }
    }

    // swizzle
    if (node.has_parent() && node.parent().isa<isl::schedule_node_mark>()) {
      const std::string &marker = node.parent().as<isl::schedule_node_mark>().get_id().get_name();
      if (marker == MIND_TRICKS_SWIZZLE_MARKER) {
        return node;
      }
    }

    if (CanBeMappedToThread(node, thread_record)) {
      auto node_bak = node;
      size_t mapped_threads = 0;
      if (scop_info_.user_config_.GetEnableAkgReduceLib() && node.has_parent() &&
          !GetMarkerName(node.parent(), REDUCE_MARKER).empty()) {
        // reduce operator
        is_reduce_stmt = true;
        ReduceMappingStrategy reduce_op(pass_info_, scop_info_);
        mapped_threads = reduce_op.MapThreadHelper(node);
      } else if (is_bmm_stmt) {
        // batch matmul operator
        BatchMatmulMappingStrategy bmm_op(pass_info_, scop_info_);
        if (scop_info_.user_config_.GetEnableConvTensorCore()) {
          // conv operator
          node = AdjustConvScheduleTreeStructure(node, false);
        }
        mapped_threads = bmm_op.MapThreadHelper(node);
      } else {
        // others operator
        OperatorMappingStrategy others_op(pass_info_, scop_info_);
        bool need_reverse = scop_info_.analysis_result_.GetReduceDirection() == Y_DIRECTION;
        mapped_threads = others_op.MapThreadHelper(node, need_reverse);
      }

      if (!node_bak.is_equal(node)) {
        // if successfully mapped current node, we insert a map filter beyond and need to return to band node
        node = node.parent();
      }
      thread_record.emplace_back(std::make_pair(node, mapped_threads));
      return node;
    }

    if (node.n_children() <= 1 || NumMappedDescendant(thread_record, node) <= 0) {
      return node;
    }
    node = MapSequenceNode(node, thread_record);

    auto need_sync = node.isa<isl::schedule_node_sequence>();
    if (need_sync) {
      if (is_reduce_stmt && node.has_parent() && !GetMarkerName(node.parent(), INSERT_SYNC).empty()) {
        node = node.parent().del();
        node = DoThreadSynchronization(node);
      } else if (!is_reduce_stmt && scop_info_.user_config_.GetEnableTensorCoreUsePoly()) {
        std::vector<MappingCfg *> other_mapping_cfg;
        other_mapping_cfg.push_back(scop_info_.user_config_.GetReplaceConfig()[WARP_COMPUTE]);
        node = DoThreadSynchronization(node, other_mapping_cfg);
      } else if (!is_reduce_stmt) {
        node = DoThreadSynchronization(node);
      }
    }

    return node;
  };
  final_schedule = sch.get_root().map_descendant_bottom_up(MapFromInner).get_schedule();
  return final_schedule;
}

isl::schedule MappingOuterBand::DoBlockMapping(const isl::schedule &sch) {
  isl::schedule_node root = sch.get_root();
  isl::schedule_node node = GetOuterBand(root);
  auto band_node = node.as<isl::schedule_node_band>();
  if (!band_node || !band_node.permutable()) {
    LOG(WARNING) << "No permutable outer band node to map block.";
    return sch;
  }

  // Step 1. Determine max num dimension of blocks that can be mapped.
  auto block_cfg = scop_info_.user_config_.GetBlockConfig();
  CHECK(block_cfg != nullptr) << "block config is null";
  auto n_block_map = (scop_info_.user_config_.GetEnableAkgReduceLib() || scop_info_.user_config_.EnableStitchFusion())
                       ? band_node.n_member()
                       : CountConsecutiveCoincident(band_node);
  n_block_map = std::min(block_cfg->MaxDim(), n_block_map);
  n_block_map = std::min(block_cfg->bound, n_block_map);
  if (n_block_map < 1) {
    return sch;
  }
  // For scalar case that do not consider coincidence (reset during restart in pass mgr), there is usually only one
  // member in outer band and we can map the maximal block size to that member.
  bool need_shift = n_block_map < block_cfg->bound && !scop_info_.user_config_.GetConsiderCoincidence();
  std::unordered_map<size_t, size_t> map_idx_shift;
  if (need_shift) {
    auto new_idx = 0;
    for (size_t i = 0; i < block_cfg->bound; ++i) {
      if (block_cfg->GetAt(i).second > block_cfg->GetAt(new_idx).second) {
        new_idx = i;
      }
    }
    if (scop_info_.analysis_result_.GetEnabledAutoTiling()) {
      // for auto configs, simply exchange the value of configs idx, for example:
      // [before] bx = 1(map), by = 1024; [after] bx = 1024(map), by = 1
      block_cfg->SwapConfig(0, new_idx);
    } else {
      // for manual configs, we need to use the user-specifed config idx, so that we can record the shifted idx and it
      // will be used in CreateAndInsertMapFilter, for example:
      // [before] bx = 1(map), by = 1024; [after] bx = 1, by = 1024(map);
      map_idx_shift.insert({0, new_idx});
      map_idx_shift.insert({new_idx, 0});
    }
  }

  // Step 2. Map outer-most band for c1 tile as usual (and do not check extent when c0 tile is applied manually).
  if (scop_info_.user_config_.GetEnableAkgReduceLib()) {
    // reduce operator
    ReduceMappingStrategy reduce_op(pass_info_, scop_info_);
    if (scop_info_.user_config_.GetEnableAtomicAdd() && reduce_op.NeedAtomicAdd(band_node, n_block_map)) {
      reduce_op.MarkAtomicAddTensor(band_node);
    }
    node = reduce_op.MapBlockHelper(node, block_cfg, n_block_map, map_idx_shift.empty(), map_idx_shift);
  } else if (scop_info_.user_config_.GetEnableConvTensorCore()) {
    // conv operator
    ConvMappingStrategy conv_op(pass_info_, scop_info_);
    node = conv_op.ResetConvBlockMappingConfig(node, block_cfg, map_idx_shift.empty());
  } else {
    // others operator
    OperatorMappingStrategy others_op(pass_info_, scop_info_);
    node = others_op.MapBlockHelper(node, block_cfg, n_block_map, map_idx_shift.empty(), map_idx_shift);
  }

  auto final_schedule = node.get_schedule();
  return final_schedule;
}

isl::schedule MappingOuterBand::Run(isl::schedule sch) {
  sch = InsertContextNode(sch, scop_info_);

  if (scop_info_.user_config_.GetEnableAkgReduceLib()) {
    ReduceMappingStrategy reduce_op(pass_info_, scop_info_);
    sch = reduce_op.DetectAndMarkReduce(sch);
  }

  sch = DoThreadMapping(sch);

  sch = DoBlockMapping(sch);

  if (scop_info_.user_config_.GetEnableConvTensorCore()) {
    ConvMappingStrategy conv_op(pass_info_, scop_info_);
    sch = conv_op.MoveKernelHWBand(sch);
  }
  return sch;
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
