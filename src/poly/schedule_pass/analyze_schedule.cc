/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

void AnalyzeSchedule::AppendBandNode(const isl::schedule_node &node,
                                     const std::function<void(const isl::schedule_node_band)> &f) {
  if (node.isa<isl::schedule_node_band>()) {
    f(node.as<isl::schedule_node_band>());
  } else if (node.isa<isl::schedule_node_set>() || node.isa<isl::schedule_node_sequence>()) {
    for (unsigned int i = 0; i < node.n_children(); ++i) {
      isl::schedule_node child_node = node.get_child(i);
      if (!child_node.isa<isl::schedule_node_filter>()) {
        continue;
      }
      auto filter_node = child_node.as<isl::schedule_node_filter>();
      if (filter_node.get_filter().is_empty()) {
        continue;
      }
      if (filter_node.has_children() && filter_node.get_child(0).isa<isl::schedule_node_band>()) {
        f(filter_node.get_child(0).as<isl::schedule_node_band>());
      }
    }
  }
}

void AnalyzeSchedule::ConstructOuterBandNode() {
  // Construct outer band.
  isl::schedule_node root_node = GetOuterBand(sch_.get_root());
  int cnt = 0;
  auto AppendOuterBand = [this, &cnt](const isl::schedule_node_band &outer_band) {
    auto prefix_schedule = outer_band.get_partial_schedule();
    if (prefix_schedule.is_null()) {
      return;
    }
    std::unique_ptr<OuterBandNode> out(new (std::nothrow) OuterBandNode(outer_band, BandScope::OUTER, cnt++));
    CHECK(out) << "memory alloc fail";
    scop_info_.analysis_result_.RecordOuterBandNode(out);
  };
  AppendBandNode(root_node, AppendOuterBand);
}

void AnalyzeSchedule::ConstructInnerBandNode() {
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
      auto upa_size = upa_list.size();
      stack.pop_back();
      auto AppendInnerBand = [&stack, &seq, &bn, upa_size](const isl::schedule_node_band &inner_band) {
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
        auto child_node = bn->node.get_child(i);
        AppendBandNode(child_node, AppendInnerBand);
      }
    }
  }
}

isl::schedule AnalyzeSchedule::Run(isl::schedule sch) {
  sch_ = sch;
  // Step 1. Construct outer band.
  ConstructOuterBandNode();
  // Step 2. Construct inner band for each outer band.
  ConstructInnerBandNode();

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