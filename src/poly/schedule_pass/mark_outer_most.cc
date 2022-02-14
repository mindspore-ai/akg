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

#include "mark_outer_most.h"

namespace akg {
namespace ir {
namespace poly {

std::vector<bool> getIsolateVector(const isl::schedule_node_band &node) {
  auto build_options = node.get_ast_build_options().get_set_list();
  std::vector<bool> isolate_vector(node.n_member(), true);
  for (auto idx = 0u; idx < build_options.size(); ++idx) {
    if (build_options.get_at(idx).get_tuple_name() == "isolate") {
      const isl::set &isolate_set = build_options.get_at(idx);
      for (int dim = 0; dim < static_cast<int>(node.n_member()); dim++) {
        isolate_vector[dim] = (isolate_set.simple_hull().dim_max_val(dim).get_num_si() > 0);
      }
      break;
    }
  }
  return isolate_vector;
}

bool InjectMulticoreToBand(isl::schedule_node &band_node) {
  auto node = band_node.as<isl::schedule_node_band>();
  if (node.is_null()) return false;
  if (node.n_member() < 1) return false;
  if (!node.get_permutable()) return false;

  auto isolate_vector = getIsolateVector(node);
  bool has_coincident = false;
  std::string mark = "multicore_coincident";
  for (int dim = 0; dim < static_cast<int>(node.n_member()); ++dim) {
    bool is_dim_coincident = isolate_vector[dim] && node.member_get_coincident(dim);
    has_coincident = has_coincident || is_dim_coincident;
    mark += "_" + std::to_string(is_dim_coincident);
  }
  if (has_coincident) {
    band_node = band_node.insert_mark(isl::id(band_node.ctx(), mark));
  }
  return has_coincident;
}

isl::schedule_node &ObtainSequenceOrSetNodeAncestor(isl::schedule_node &node) {
  do {
    node = node.parent();
  } while (!node.isa<isl::schedule_node_sequence>() && !node.isa<isl::schedule_node_set>());
  return node;
}

bool InjectMulticoreToChildrenBands(isl::schedule_node &sequence_node) {
  bool has_multicore = false;
  for (unsigned int filter = 0; filter < sequence_node.n_children(); ++filter) {
    auto filter_node = sequence_node.get_child(filter);
    auto band = GetOuterBand(filter_node);
    if (InjectMulticoreToBand(band)) {
      has_multicore = true;
      sequence_node = ObtainSequenceOrSetNodeAncestor(band);
    }
  }
  return has_multicore;
}

bool MarkOuterMost::SingleMulticoreBand(isl::schedule_node &outer_band) {
  if (outer_band.as<isl::schedule_node_sequence>() || outer_band.as<isl::schedule_node_set>()) {
    int multi_core_band = 0;
    for (unsigned int i = 0; i < outer_band.n_children(); ++i) {
      isl::schedule_node node = outer_band.get_child(i);
      if (node.isa<isl::schedule_node_filter>()) {
        auto filter = node.as<isl::schedule_node_filter>();
        if (filter.has_children()) {
          auto node0 = filter.get_child(0);
          while (node0.isa<isl::schedule_node_mark>()) {
            node0 = node0.get_child(0);
          }
          if (node0.isa<isl::schedule_node_band>() && node0.as<isl::schedule_node_band>().n_member() >= 1) {
            multi_core_band++;
          }
        }
      }
    }
    if (multi_core_band == 1) {
      return true;
    }
  }
  return false;
}

bool MarkOuterMost::InjectMulticoreToSchedule(isl::schedule_node &outer_band) {
  if (outer_band.as<isl::schedule_node_band>()) {
    return InjectMulticoreToBand(outer_band);
  } else if (outer_band.as<isl::schedule_node_sequence>() || outer_band.as<isl::schedule_node_set>()) {
    if (SingleMulticoreBand(outer_band)) {
      for (unsigned int i = 0; i < outer_band.n_children(); ++i) {
        isl::schedule_node node = outer_band.get_child(i);
        if (node.isa<isl::schedule_node_filter>()) {
          auto filter = node.as<isl::schedule_node_filter>();
          if (filter.has_children()) {
            auto node0 = filter.get_child(0);
            while (node0.isa<isl::schedule_node_mark>()) {
              node0 = node0.get_child(0);
            }
            if (node0.isa<isl::schedule_node_band>() &&
              node0.as<isl::schedule_node_band>().n_member() >= 1) {
              bool injected = InjectMulticoreToBand(node0);
              outer_band = ObtainSequenceOrSetNodeAncestor(node0);
              return injected;
            }
          }
        }
      }
    }
    bool is_bands_independent = scop_info_.analysis_result_.GetInnerBandDependency().is_empty();
    if (!is_bands_independent) {
      // Conv outer bands indeed have inter-band dependency, but it will be fixed in post_fusion,
      // so Conv can still use multicore. This is actually dangerous and may need double check.
      if (!this->scop_info_.mmu_info_.IsConv()) return false;
    }
    return InjectMulticoreToChildrenBands(outer_band);
  }
  return false;
}

isl::schedule MarkOuterMost::Run(isl::schedule schedule_mark) {
  isl::schedule_node root = schedule_mark.get_root();
  isl::schedule_node outer_band = GetOuterBand(root);
  bool has_multi_core = InjectMulticoreToSchedule(outer_band);
  if (has_multi_core) {
    return outer_band.get_schedule();
  } else {
    LOG(INFO) << "This operator is not capable of using multi-core. "
              << "Possible reasons are: "
              << "1) there is dependency between outer bands. "
              << "2) outer bands are not tiled (only tiles of outer band can use multicore).";
    return schedule_mark;
  }
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
