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

#include <numeric>

#include "poly/schedule_tree_util.h"
#include "poly/sync_manager.h"
#include "poly/scop.h"

namespace akg {
namespace ir {
namespace poly {

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

size_t MappingOuterBand::GetReduceId() const {
  static size_t reduce_count = 0;
  return reduce_count++;
}

std::string MappingOuterBand::GetMarkerName(const isl::schedule_node &node, std::string find_name) {
  std::string reduce_marker_name = "";
  if (node.isa<isl::schedule_node_mark>()) {
    reduce_marker_name = node.as<isl::schedule_node_mark>().get_id().get_name();
    if (reduce_marker_name.find(find_name) != std::string::npos) {
      return reduce_marker_name;
    }
    reduce_marker_name = "";
  }
  return reduce_marker_name;
}

size_t MappingOuterBand::CountConsecutiveCoincident(const isl::schedule_node_band &band_node) {
  size_t count = 0;
  while (count < band_node.n_member()) {
    if (!band_node.member_get_coincident(static_cast<int>(count))) {
      break;
    }
    ++count;
  }
  return count;
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

  isl::union_set domain = CollectDomain(node);
  isl::space space = domain.get_space();
  space = space.set_from_params();
  isl::multi_val mv = isl::multi_val::zero(space);
  isl::multi_union_pw_aff mupa = isl::multi_union_pw_aff(domain, mv);
  node.insert_partial_schedule(mupa);

  isl::schedule_node_band band_node = node.as<isl::schedule_node_band>();
  Mapping mapping;
  auto after_map_pair = MapInnerDimToThreads(band_node, false, thread_cfg, mapping,
                                             scop_info_.analysis_result_.GetReduceDirection() == Y_DIRECTION);
  auto after_map_node = after_map_pair.first;
  scop_info_.upa_node_mapping_.emplace_back(std::make_pair(after_map_node, mapping));
  after_map_node = after_map_node.parent();
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
    if (scop_info_.user_config_.GetEnableAkgReduceLib() && node.has_parent() &&
        !GetMarkerName(node.parent(), REDUCE_MARKER).empty()) {
      is_reduce_stmt = true;
    }

    if (scop_info_.user_config_.GetEnableTensorCoreUsePoly() && node.get_tree_depth() >= 2 &&
        !GetMarkerName(node.ancestor(2), PROMOTE_SHARED_TO_REGISTER).empty()) {
      return node;
    }

    if (node.has_parent() && node.parent().isa<isl::schedule_node_mark>()) {
      const std::string &marker = node.parent().as<isl::schedule_node_mark>().get_id().get_name();
      if (marker == "mind_trick_swizzle_marker")
        return node;
    }

    size_t num_mapped_desc = NumMappedDescendant(thread_record, node);

    if (CanBeMappedToThread(node, thread_record)) {
      auto node_bak = node;
      auto mapped_threads = MapThreadHelper(node);
      if (!node_bak.is_equal(node)) {
        // if successfully mapped current node, we insert a map filter beyond and need to return to band node
        node = node.parent();
      }
      thread_record.emplace_back(std::make_pair(node, mapped_threads));
      return node;
    }

    // deal with band that has children mapped to threads
    if (node.n_children() > 1 && num_mapped_desc > 0) {
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

      auto need_sync = node.isa<isl::schedule_node_sequence>();
      if (need_sync) {
        if (is_reduce_stmt && node.has_parent() && !GetMarkerName(node.parent(), INSERT_SYNC).empty()) {
          node = node.parent().del();
          node = DoThreadSynchronization(node);
        } else if (!is_reduce_stmt) {
          node = DoThreadSynchronization(node);
        }
      }

      auto band = node.as<isl::schedule_node_band>();
      if (band && CountConsecutiveCoincident(band) < band.n_member()) {
        CHECK_EQ(num_mapped_desc, thread_cfg->bound) << "Must be mapped to all threads.";
        auto sync_manager = scop_info_.sync_manager_;
        sync_manager.InsertExtensionNode(band.child(0), SyncLevel::BLOCK, true);
      }
    }
    return node;
  };
  final_schedule = sch.get_root().map_descendant_bottom_up(MapFromInner).get_schedule();
  return final_schedule;
}

isl::schedule_node MappingOuterBand::DoThreadSynchronization(const isl::schedule_node &node) {
  auto sync_node = node;
  auto sync_manager = scop_info_.sync_manager_;
  auto thread_cfg = scop_info_.user_config_.GetThreadConfig();
  CHECK(thread_cfg != nullptr) << "thread config is null";

  // Step 1. prepare info
  bool is_outer = IsOuterBandWithNoCoincident(node);
  auto domain_thread = MapDomainToThread(node, thread_cfg, scop_info_.upa_node_mapping_);

  if (scop_info_.user_config_.GetEnableTensorCoreUsePoly()) {
    auto warp_cfg = scop_info_.user_config_.GetReplaceConfig()[WARP_COMPUTE];
    CHECK(warp_cfg != nullptr) << "warp config is null";
    auto domain_thread_warp = MapDomainToThread(node, warp_cfg, scop_info_.upa_node_mapping_);
    domain_thread = domain_thread.union_add(domain_thread_warp);
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

isl::schedule_node MappingOuterBand::InsertReduceExtension(const isl::schedule_node &node) {
  auto thread_cfg = scop_info_.user_config_.GetThreadConfig();
  CHECK(thread_cfg != nullptr) << "thread config is null";

  isl::schedule_node insert_node = node;
  isl::schedule_node parent_node = node;
  isl::schedule_node ancestor_node = node;
  if (insert_node.has_parent()) {
    parent_node = parent_node.parent();
    if (parent_node.has_parent()) {
      ancestor_node = parent_node.parent();
    }
  }

  std::string reduce_marker_name = "";
  if (!GetMarkerName(parent_node, REDUCE_MARKER).empty()) {
    reduce_marker_name = GetMarkerName(parent_node, REDUCE_MARKER);
    insert_node = parent_node.del();
  }

  if (!GetMarkerName(ancestor_node, REDUCE_MARKER).empty()) {
    reduce_marker_name = GetMarkerName(ancestor_node, REDUCE_MARKER);
    insert_node = ancestor_node.del();
  }

  if (reduce_marker_name.empty()) {
    return node;
  }

  reduce_marker_name.erase(0, strlen(REDUCE_MARKER));
  isl::id sync_id = isl::id(insert_node.ctx(), REDUCE_UPDATE + reduce_marker_name);
  isl::id reduction_id = isl::id(insert_node.ctx(), REDUCE_INIT + reduce_marker_name);

  insert_node = InsertExtensionNodeBeforeOrAfter(insert_node, reduction_id, true);
  insert_node = InsertExtensionNodeBeforeOrAfter(insert_node, sync_id, false).parent();

  return insert_node;
}

size_t MappingOuterBand::MapThreadHelper(isl::schedule_node &thread_root) {
  isl::schedule_node_band band_node = thread_root.as<isl::schedule_node_band>();
  auto thread_cfg = scop_info_.user_config_.GetThreadConfig();
  CHECK(thread_cfg != nullptr) << "thread config is null";
  if (thread_cfg->bound < 1) {
    return 0;
  }

  if (!band_node) {
    LOG(WARNING) << "No permutable band to map thread.";
    return 0;
  }

  int start_node_depth = thread_root.get_tree_depth();
  // Step 1. Determine max num dimension of threads that can be mapped.
  auto n_thread_map = CountConsecutiveCoincident(band_node);

  bool is_bmm_statement = false;
  if (scop_info_.user_config_.GetEnableTensorCoreUsePoly() && thread_root.has_parent() &&
      !GetMarkerName(thread_root.parent(), PROMOTE_SHARED_TO_REGISTER).empty()) {
    auto warp_cfg = scop_info_.user_config_.GetReplaceConfig()[WARP_COMPUTE];
    CHECK(warp_cfg != nullptr) << "warp config is null";
    thread_cfg = warp_cfg;
    is_bmm_statement = true;
  }

  bool is_reduce_stmt = false;
  std::string reduce_marker_name = "";
  if (band_node.has_parent()) {
    reduce_marker_name = GetMarkerName(band_node.parent(), REDUCE_MARKER);
    if (!reduce_marker_name.empty()) {
      ++n_thread_map;
      is_reduce_stmt = true;
    }
  }

  // When akg reduce lib is enabled, we can try to map other injective statements whose coincidence equals 0
  if (n_thread_map < thread_cfg->bound && scop_info_.user_config_.GetEnableAkgReduceLib()) {
    n_thread_map = thread_cfg->bound;
  }

  if (n_thread_map < 1) {
    return 0;
  }

  // Step 2. Split band node according to mapping config and coincidence of band node.
  if (n_thread_map > thread_cfg->bound) {
    thread_root = band_node.split(n_thread_map - thread_cfg->bound);
    thread_root = thread_root.child(0);
    n_thread_map = thread_cfg->bound;
    if (is_reduce_stmt) {
      isl::schedule_node ancestor_node = thread_root.ancestor(2);
      ancestor_node = ancestor_node.del().child(0);
      thread_root = ancestor_node.insert_mark(reduce_marker_name);
      thread_root = thread_root.child(0);
    }
    band_node = thread_root.as<isl::schedule_node_band>();
  }

  // split to keep nodes with coincident equals to 1
  if (n_thread_map < band_node.n_member()) {
    thread_root = band_node.split(n_thread_map);
    band_node = thread_root.as<isl::schedule_node_band>();
  } else {
    n_thread_map = static_cast<size_t>(band_node.n_member());
  }

  // Step 3. Map band under thread_root from inner dim to outer dim.
  Mapping mapping;
  bool is_y_reduce =
    scop_info_.analysis_result_.GetReduceDirection() == Y_DIRECTION || scop_info_.user_config_.GetEnableTensorCore();
  auto after_map_pair = MapInnerDimToThreads(band_node, false, thread_cfg, mapping, is_y_reduce);
  thread_root = after_map_pair.first;
  if (is_bmm_statement && !GetMarkerName(thread_root, THREAD_MARKER).empty()) {
    thread_root = thread_root.del().insert_mark(isl::id(thread_root.ctx(), WARP_MARKER));
  }
  scop_info_.upa_node_mapping_.emplace_back(std::make_pair(thread_root, mapping));
  int end_node_depth = thread_root.get_tree_depth() - start_node_depth;

  if (is_reduce_stmt) {
    // Split the reduce axis and non-reduce axis of the outer band.
    if (thread_root.ancestor(2) && !GetMarkerName(thread_root.ancestor(2), REDUCE_MARKER).empty() && n_thread_map > 1) {
      thread_root = thread_root.ancestor(2).del();
      band_node = thread_root.as<isl::schedule_node_band>();
      thread_root = band_node.split(n_thread_map - 1).child(0);
      thread_root = thread_root.insert_mark(reduce_marker_name);
      thread_root = thread_root.child(0);
    }
    // Add the filter that initializes and calls the akg_reduce library for the reduce statement.
    thread_root = InsertReduceExtension(thread_root);
    end_node_depth = thread_root.get_tree_depth() - start_node_depth;
    ++end_node_depth;
  }
  thread_root = thread_root.ancestor(end_node_depth);

  // Step 4. Do unroll if needed.
  if (scop_info_.user_config_.GetMaxUnrollLoop() != 1) {
    isl::schedule_node after_fix_node = thread_root.child(0);
    if (!IsEqualNode(after_map_pair.second, after_map_pair.first)) {
      after_fix_node = after_fix_node.parent();
    }
    thread_root = UnrollByMarkOptions(after_fix_node, scop_info_.user_config_.GetMaxUnrollLoop());
  }

  return thread_cfg->bound;
}

isl::schedule MappingOuterBand::DetectAndMarkReduce(const isl::schedule &sch) {
  auto final_schedule = sch;
  auto thread_cfg = scop_info_.user_config_.GetThreadConfig();
  CHECK(thread_cfg != nullptr) << "threadconfig is null";
  if (thread_cfg->bound == 0) {
    return final_schedule;
  }

  auto all_reduce_map = scop_info_.analysis_result_.GetReduceTensorInfoMap();
  ReduceManager reduce_manager;
  bool done_separate = false;
  auto GetInnerMostBand = [&done_separate, &all_reduce_map, &reduce_manager, thread_cfg,
                           this](isl::schedule_node node) -> isl::schedule_node {
    if (done_separate) {
      return node;
    }
    auto band_node = node.as<isl::schedule_node_band>();
    if (!band_node || !band_node.permutable()) {
      return node;
    }

    auto band_node_domain = band_node.get_partial_schedule().domain();
    StatementMap all_statements = scop_info_.analysis_result_.GetStatementMap();
    isl::union_map reduce_statement_map = isl::union_map::empty(node.ctx());
    isl::union_set reduce_statements = isl::union_set::empty(node.ctx());

    for (auto it = all_reduce_map.begin(); it != all_reduce_map.end();) {
      reduce_statement_map = reduce_statement_map.unite(it->second.stmt_map);
      auto this_reduce = reduce_manager.GetReduceStatements(band_node_domain, reduce_statement_map, all_statements);
      if (!this_reduce.is_empty()) {
        reduce_statements = reduce_statements.unite(this_reduce);
        all_reduce_map.erase(it++);
      } else {
        ++it;
      }
    }

    if (reduce_statements.n_set() < 1) {
      return node;
    }

    isl::union_map dependences = pass_info_.dependences_.subtract(pass_info_.force_dependences_);
    auto node_bak = node;
    if (!reduce_manager.SplitReduceStatements(node, reduce_statements, dependences, true)) {
      return node_bak;
    }
    done_separate = all_reduce_map.empty();
    return node;
  };
  final_schedule = sch.get_root().map_descendant_bottom_up(GetInnerMostBand).get_schedule();
  if (done_separate) {
    final_schedule = InsertReduceMarker(final_schedule);
  }
  return final_schedule;
}

isl::schedule MappingOuterBand::InsertReduceMarker(const isl::schedule &sch) {
  isl::schedule final_schedule = sch;
  auto all_reduce_map = scop_info_.analysis_result_.GetReduceTensorInfoMap();
  auto InsertMarker = [&all_reduce_map, this](isl::schedule_node node) -> isl::schedule_node {
    ReduceManager reduce_manager;
    auto band_node = node.as<isl::schedule_node_band>();
    if (!band_node) {
      return node;
    }

    for (auto it = all_reduce_map.begin(); it != all_reduce_map.end();) {
      isl::union_map reduce_statement_map = it->second.stmt_map;
      isl::id reduce_id = it->first;
      auto band_node_domain = band_node.get_partial_schedule().domain();
      auto op_type = scop_info_.analysis_result_.GetReduceOpType(reduce_id) + "_";

      StatementMap all_statements = scop_info_.analysis_result_.GetStatementMap();
      isl::union_set reduce_statements =
        reduce_manager.GetReduceStatements(band_node_domain, reduce_statement_map, all_statements);
      if (reduce_statements.n_set() != 1) {
        ++it;
        continue;
      }

      all_reduce_map.erase(it++);
      std::string reduce_marker_name =
        REDUCE_MARKER + op_type + reduce_id.get_name() + "_" + std::to_string(GetReduceId());
      auto reduce_node = band_node.insert_mark(reduce_marker_name);
      return reduce_node;
    }
    return band_node;
  };
  final_schedule = final_schedule.get_root().map_descendant_bottom_up(InsertMarker).get_schedule();
  return final_schedule;
}

std::pair<std::string, std::string> MappingOuterBand::GetC1C0BlockConfig(size_t n_block_map, int member_size) {
  auto block_cfg = scop_info_.user_config_.GetBlockConfig();
  CHECK(block_cfg != nullptr) << "block config is null";
  auto title_size = static_cast<int>(pass_info_.tile_sizes_.size());
  auto dim_num = (member_size <= title_size) ? member_size : title_size;
  std::vector<int> c1_tile_size = GetTileSizeOfLevel(member_size, dim_num, TILE_WITH_C1, pass_info_.tile_sizes_);
  std::vector<int> c0_tile_size = GetTileSizeOfLevel(member_size, dim_num, TILE_WITH_C0, pass_info_.tile_sizes_);
  CHECK_EQ(c1_tile_size.size(), c0_tile_size.size());

  for (size_t i = 0; i < c1_tile_size.size(); ++i) {
    auto c0 = c0_tile_size[i] <= 0 ? 1 : c0_tile_size[i];
    c0_tile_size[i] = std::max(1, c1_tile_size[i] / c0);
  }
  std::string c1_cfg = "";
  std::string c0_cfg = "";
  std::vector<int> c1_cfg_list;
  std::vector<int> c0_cfg_list;
  if (std::accumulate(c0_tile_size.begin(), c0_tile_size.end(), 1, std::multiplies<int>()) > 1) {
    for (size_t i = 0; i < n_block_map; ++i) {
      auto c0 = i < c0_tile_size.size() ? c0_tile_size[i] : 1;
      c0_cfg_list.emplace_back(c0);
      auto block_idx = scop_info_.analysis_result_.GetReduceDirection() == Y_DIRECTION ? i : n_block_map - 1 - i;
      auto c1 = block_cfg->GetAt(block_idx).second / c0;
      c1_cfg_list.emplace_back(c1);
    }
    if (scop_info_.analysis_result_.GetReduceDirection() != Y_DIRECTION) {
      std::reverse(c1_cfg_list.begin(), c1_cfg_list.end());
      std::reverse(c0_cfg_list.begin(), c0_cfg_list.end());
    }
    scop_info_.user_config_.SetC0BlockSize(c0_cfg_list);
    for (size_t i = 0; i < n_block_map; ++i) {
      c1_cfg += (std::to_string(c1_cfg_list[i]) + " ");
      c0_cfg += (std::to_string(c0_cfg_list[i]) + " ");
    }
  }
  return std::make_pair(c1_cfg, c0_cfg);
}

isl::schedule_node MappingOuterBand::MapBlockHelper(const isl::schedule_node &orig_node, MappingCfg *block_cfg,
                                                    size_t n_block_map, bool check_extent) {
  auto node = orig_node;
  auto band_node = node.as<isl::schedule_node_band>();
  if (!band_node || !band_node.permutable()) {
    LOG(WARNING) << "No permutable outer band node to map block.";
    return node;
  }

  auto partial_schedule = band_node.get_partial_schedule();
  auto upa_list = partial_schedule.get_union_pw_aff_list();

  if (check_extent) {
    auto domain = band_node.get_schedule().get_domain();
    isl::union_pw_aff_list range_aff_list(band_node.ctx(), static_cast<int>(upa_list.size()));
    for (int i = upa_list.size() - 1; i >= 0; --i) {
      auto idx = scop_info_.analysis_result_.GetReduceDirection() == Y_DIRECTION ? upa_list.size() - 1 - i : i;
      auto range = upa_list.get_at(idx).intersect_domain(domain);
      range_aff_list = range_aff_list.add(range);
    }
    node = CheckMapSizeAndApplyTile(node, range_aff_list, block_cfg, false);
  }

  upa_list = upa_list.drop(n_block_map, upa_list.size() - n_block_map).reverse();
  if (scop_info_.analysis_result_.GetReduceDirection() == Y_DIRECTION) {
    upa_list = upa_list.reverse();
  }
  node = node.insert_mark(isl::id(node.ctx(), BLOCK_MARKER));
  node = node.child(0);

  Mapping mapping;
  node = CreateAndInsertMapFilter(node, false, upa_list, block_cfg, mapping);
  scop_info_.upa_node_mapping_.emplace_back(std::make_pair(node.parent(), mapping));

  return node;
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
  auto n_block_map =
    scop_info_.user_config_.GetEnableAkgReduceLib() ? band_node.n_member() : CountConsecutiveCoincident(band_node);
  n_block_map = std::min(block_cfg->MaxDim(), n_block_map);
  n_block_map = std::min(block_cfg->bound, n_block_map);
  if (n_block_map < 1) {
    return sch;
  }
  // For scalar case that do not consider coincidence (reset during restart in pass mgr), there is usually only one
  // member in outer band and we can map the maximal block size to that member.
  if (n_block_map == 1 && n_block_map < block_cfg->bound && !scop_info_.user_config_.GetConsiderCoincidence()) {
    auto new_idx = 0;
    for (size_t i = 0; i < block_cfg->bound; ++i) {
      if (block_cfg->GetAt(i).second > block_cfg->GetAt(new_idx).second) {
        new_idx = i;
      }
    }
    block_cfg->SwapConfig(0, new_idx);
  }

  if (scop_info_.user_config_.GetEnableAtomicAdd() && NeedAtomicAdd(band_node, n_block_map)) {
    MarkAtomicAddTensor(band_node);
  }

  // Step 2. Separate original block config according to tile levels.
  auto c1_block_cfg = block_cfg;
  MappingCfg *c0_block_cfg = nullptr;
  if (scop_info_.user_config_.GetEnableTileC0()) {
    auto c1_c0_block_cfg = GetC1C0BlockConfig(n_block_map, band_node.n_member());
    if (!c1_c0_block_cfg.first.empty() && !c1_c0_block_cfg.second.empty()) {
      scop_info_.user_config_.RecordReplaceConfig(TILE_WITH_C1, c1_c0_block_cfg.first, MappingType::REPLACE_BLOCKS);
      scop_info_.user_config_.RecordReplaceConfig(TILE_WITH_C0, c1_c0_block_cfg.second, MappingType::REPLACE_BLOCKS);
      auto rep_cfg = scop_info_.user_config_.GetReplaceConfig();
      c1_block_cfg = rep_cfg[TILE_WITH_C1];
      c0_block_cfg = rep_cfg[TILE_WITH_C0];
    }
  }

  // Step 3. Map outer-most band for c1 tile as usual (and do not check extent when c0 tile is applied manually).
  auto map_c0_block = c0_block_cfg != nullptr;
  node = MapBlockHelper(node, c1_block_cfg, n_block_map, !map_c0_block);
  auto final_schedule = node.get_schedule();

  // Step 4. Map middle-level band (i.e. c0 tile band).
  if (map_c0_block) {
    isl::schedule_node middle_node = GetOuterBand(final_schedule.get_root()).child(0);
    middle_node = MapBlockHelper(middle_node, c0_block_cfg, n_block_map, false);
    final_schedule = middle_node.get_schedule();
  }

  return final_schedule;
}

bool MappingOuterBand::NeedAtomicAdd(const isl::schedule_node_band &band, size_t n_block_map) {
  if (!scop_info_.user_config_.GetEnableAkgReduceLib()) {
    return false;
  }

  auto non_coin_start_idx = CountConsecutiveCoincident(band);
  bool is_all_reduce =
    band.n_member() == 1 && scop_info_.analysis_result_.GetReduceDirection() == X_DIRECTION && non_coin_start_idx == 1;
  if (is_all_reduce) {
    non_coin_start_idx = 0;  // Compare block size of position 0 to enable atomic add for all reduce ops
  }
  if (n_block_map < non_coin_start_idx) {
    return false;
  }

  auto block_cfg = scop_info_.user_config_.GetBlockConfig();
  CHECK(block_cfg != nullptr) << "block config is null";
  while (non_coin_start_idx < block_cfg->bound) {
    auto idx = scop_info_.analysis_result_.GetReduceDirection() == Y_DIRECTION
                 ? non_coin_start_idx
                 : block_cfg->bound - non_coin_start_idx - 1;
    if (block_cfg->GetAt(idx).second > 1) {
      return true;
    }
    ++non_coin_start_idx;
  }
  return false;
}

void MappingOuterBand::MarkAtomicAddTensor(const isl::schedule_node_band &band) {
  auto target_stmt = scop_info_.analysis_result_.GetReduceWriteStmt(band);
  auto tensor = target_stmt.range();
  std::unordered_set<isl::id, isl::IslIdIslHash> stmt_ids;
  target_stmt.foreach_map(
    [this, &stmt_ids](const isl::map m) { stmt_ids.insert(m.get_tuple_id(isl_dim_type::isl_dim_in)); });
  tensor.foreach_set([this, &stmt_ids](const isl::set &s) -> void {
    for (auto it : scop_info_.analysis_result_.GetReduceTensorInfoMap()) {
      auto provide = static_cast<const Provide *>(it.second.stmt_node);
      if (stmt_ids.count(it.first) == 0 || provide->func->func_name() != s.get_tuple_name()) {
        continue;
      }
      auto type = scop_info_.analysis_result_.GetReduceOpType(it.first);
      scop_info_.analysis_result_.RecordAtomicTensors(AtomicInfo{s.get_tuple_name(), type});
    }
  });
}

isl::schedule MappingOuterBand::Run(isl::schedule sch) {
  auto node = sch.root().child(0);
  node = InsertContextNode(node, scop_info_);
  sch = node.schedule();

  if (scop_info_.user_config_.GetEnableAkgReduceLib()) {
    sch = DetectAndMarkReduce(sch);
  }

  sch = DoThreadMapping(sch);

  sch = DoBlockMapping(sch);
  return sch;
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
