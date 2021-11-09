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
#include "analyze_schedule.h"
#include "poly/scop.h"
#include "poly/scop_builder.h"
#include "poly/schedule_analysis/gpu_dma_analysis.h"
#include "poly/schedule_analysis/operator_info_collector.h"
#include "poly/schedule_analysis/band_node_analysis.h"

namespace akg {
namespace ir {
namespace poly {

void AnalyzeSchedule::ConstructBandNode() {
  // Step 1. Construct outer band.
  isl::schedule_node root_node = GetOuterBand(sch_.get_root());
  int cnt = 0;
  auto append_outer_band = [this, &cnt](const isl::schedule_node_band &outer_band) {
    auto prefix_schedule = outer_band.get_partial_schedule();
    if (prefix_schedule.is_null()) {
      return;
    }
    std::unique_ptr<OuterBandNode> out(new (std::nothrow) OuterBandNode(outer_band, BandScope::OUTER, cnt++));
    CHECK(out) << "memory alloc fail";
    scop_info_.analysis_result_.RecordOuterBandNode(out);
  };
  if (root_node.isa<isl::schedule_node_band>()) {  // single outer band
    append_outer_band(root_node.as<isl::schedule_node_band>());
  } else if (root_node.isa<isl::schedule_node_set>() ||
             root_node.isa<isl::schedule_node_sequence>()) {  // multiple outer bands
    for (unsigned int i = 0; i < root_node.n_children(); ++i) {
      isl::schedule_node node = root_node.get_child(i);
      if (node.isa<isl::schedule_node_filter>()) {
        auto filter = node.as<isl::schedule_node_filter>();
        if (filter.get_filter().is_empty()) {
          continue;
        }
        if (filter.has_children() && filter.get_child(0).isa<isl::schedule_node_band>()) {
          append_outer_band(filter.get_child(0).as<isl::schedule_node_band>());
        }
      }
    }
  }

  // Step 2. Construct inner band for each outer band.
  auto &band_nodes = scop_info_.analysis_result_.GetAllOuterBandNode();
  std::vector<OuterBandNode *> stack;
  for (auto &band_node : band_nodes) {
    auto node = band_node.get();
    stack.emplace_back(node);
    size_t seq = 0;
    while (!stack.empty()) {
      auto *bn = stack.back();
      seq += bn->children.size();
      auto prefix_schedule = bn->node.get_partial_schedule();
      auto upa_list = prefix_schedule.get_union_pw_aff_list();
      stack.pop_back();
      auto AppendInnerBand = [&stack, &seq, &bn](const isl::schedule_node_band &inner_band, const size_t upa_size) {
        if (inner_band.get_partial_schedule().is_null()) {
          return;
        }
        seq += upa_size;
        std::unique_ptr<OuterBandNode> in(new (std::nothrow) OuterBandNode(inner_band, BandScope::INNER, seq));
        CHECK(in) << "memory alloc fail";
        in->parent = bn;
        bn->children.emplace_back(std::move(in));
        stack.emplace_back(bn->children.back().get());
      };
      for (int i = 0; i < static_cast<int>(bn->node.n_children()); ++i) {
        if (bn->node.get_child(i).as<isl::schedule_node_band>()) {  // single inner band
          AppendInnerBand(bn->node.get_child(i).as<isl::schedule_node_band>(), upa_list.size());
        } else if (bn->node.get_child(i).isa<isl::schedule_node_set>() ||
                   bn->node.get_child(i).isa<isl::schedule_node_sequence>()) {  // multiple inner bands
          int n = bn->node.get_child(i).n_children();
          for (int j = 0; j < n; ++j) {
            if (bn->node.get_child(i).get_child(j).isa<isl::schedule_node_filter>()) {
              auto filter = bn->node.get_child(i).get_child(j).as<isl::schedule_node_filter>();
              if (filter.get_filter().is_empty()) {
                continue;
              }
              if (filter.has_children() && filter.get_child(0).isa<isl::schedule_node_band>()) {
                AppendInnerBand(filter.get_child(0).as<isl::schedule_node_band>(), upa_list.size());
              }
            }
          }
        }
      }
    }
  }
}

isl::schedule AnalyzeSchedule::Run(isl::schedule sch) {
  sch_ = sch;
  ConstructBandNode();

  OpTypeCollector op_type_collector(scop_info_, stmt_);
  op_type_collector.Run();

  AnalyzeBandNode analyzer(sch, scop_info_);
  analyzer.Run();

  if (target_ == TARGET_CUDA) {
    GpuDmaAnalysis dma_analysis(sch, scop_info_);
    dma_analysis.Run();
  }
  return sch_;
}

}  // namespace poly
}  // namespace ir
}  // namespace akg