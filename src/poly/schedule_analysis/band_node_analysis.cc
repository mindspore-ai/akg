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
#include "band_node_analysis.h"
#include "poly/schedule_tree_util.h"
#include "poly/schedule_pass.h"

namespace akg {
namespace ir {
namespace poly {

void AnalyzeBandNode::Run() {
  if (target_ == TARGET_CCE) {
    return;
  }
  CollectStmtInfo();
  AnalyzeOuterBandTemplate();
  if (target_ == TARGET_CPU) {
    AnalyzeAxisPosition();
  }
  ShowBandInfo();
}

void AnalyzeBandNode::AnalyzeAxisPosition() {
  auto &bands = scop_info_.analysis_result_.GetBandNodes();
  for (auto &band_node : bands) {
    auto *bn = band_node.get();
    if (!bn->node.isa<isl::schedule_node_band>()) {
      continue;
    }
    SetVectorizationAxis(bn->node, bn->index);
  }
}

void AnalyzeBandNode::SetVectorizationAxis(const isl::schedule_node &orig_node, const int index) {
  if (!orig_node.isa<isl::schedule_node_band>()) {
    return;
  }

  auto op_template = scop_info_.analysis_result_.GetOpTemplateOfBand(index);
  auto n_member = static_cast<int>(orig_node.as<isl::schedule_node_band>().n_member());
  bool is_reduce_op = (op_template == Template::REDUCTION || op_template == Template::BITWISE_REDUCTION);

  int vectorization_axis = -1;
  if (is_reduce_op) {
    if (scop_info_.analysis_result_.GetReduceDirectionOfBand(index) == ReduceDirection::Y) {
      vectorization_axis = 0;
    } else {
      vectorization_axis = n_member - 1;
    }
  } else if (op_template == Template::BROADCAST_OP) {
    vectorization_axis = n_member - 1;
  } else {
    vectorization_axis = GetElemVectorizationAxisPos(orig_node);
  }
  scop_info_.analysis_result_.SetLastAxisOfBand(vectorization_axis, index);
}

int AnalyzeBandNode::GetElemVectorizationAxisPos(const isl::schedule_node &orig_node) {
  if (!orig_node.isa<isl::schedule_node_band>()) {
    return -1;
  }

  auto node = orig_node;
  auto band_node = node.as<isl::schedule_node_band>();
  auto n_parallel_axis = CountConsecutiveCoincident(band_node);
  node = band_node.split(n_parallel_axis);

  std::unordered_set<std::string> skip_tensors;
  // Get read and write tensor information.
  auto reads_access = scop_info_.analysis_result_.GetReads().domain_factor_domain();
  int last_axis = GetLastAxis(node, reads_access, skip_tensors);
  if (last_axis != -1) {
    return last_axis;
  }

  auto write_access = scop_info_.analysis_result_.GetWrites().domain_factor_domain();
  last_axis = GetLastAxis(node, write_access, skip_tensors);
  if (last_axis != -1) {
    return last_axis;
  }
  return -1;
}

void AnalyzeBandNode::CollectStmtInfo() {
  auto prov_entry = scop_info_.analysis_result_.GetProvideAnalysis();
  auto provides = scop_info_.analysis_result_.GetStatementMap();
  if (prov_entry.empty() || provides.empty()) {
    return;
  }
  std::vector<ProvideEntry> entries;
  for (auto &provs : prov_entry) {
    for (auto &p : provs.second) {
      entries.emplace_back(p);
    }
  }
  auto direct_map = scop_info_.analysis_result_.GetReduceDirectionMap();
  for (auto &pro : provides) {
    if (!pro.second->IsInstance<Provide>()) {
      continue;
    }
    auto stmt = pro.first;
    for (auto &entry : entries) {
      if (entry.op != pro.second) {
        continue;
      }
      std::string s_type = entry.basic_op_type;
      ReduceDirection direct{ReduceDirection::UNKNOWN};
      if (direct_map.find(stmt) != direct_map.end()) {
        direct = direct_map[stmt];
      }
      stmt_info_[stmt] = std::make_pair(s_type, direct);
    }
  }
}

void AnalyzeBandNode::DetermineTemplateOfBand(BandNode *bn) {
  if (!bn || bn->stmts.empty()) {
    return;
  }
  std::string concated_op_type;
  ReduceDirection direct{ReduceDirection::UNKNOWN};
  isl::id red_stmt;
  for (auto &st : bn->stmts) {
    if (stmt_info_.find(st) == stmt_info_.end()) {
      continue;
    }
    concated_op_type += stmt_info_[st].first + ",";
    if (stmt_info_[st].second != ReduceDirection::UNKNOWN) {
      direct = stmt_info_[st].second;
      red_stmt = st;
    }
  }
  if (concated_op_type.find(AT_REDUCE) != std::string::npos) {
    auto type = scop_info_.analysis_result_.GetReduceOpType(red_stmt);
    if (type == AKG_REDUCE_AND || type == AKG_REDUCE_OR) {
      bn->info.type = Template::BITWISE_REDUCTION;
    } else {
      bn->info.type = Template::REDUCTION;
    }
    bn->info.direction = direct;
  } else if (concated_op_type.find(AT_TRANSPOSE) != std::string::npos) {
    bn->info.type = Template::TRANSPOSE_OP;
  } else if (concated_op_type.find(AT_PAD) != std::string::npos) {
    bn->info.type = Template::PAD_OP;
  } else if (concated_op_type.find(AT_BROADCAST) != std::string::npos ||
             concated_op_type.find(AT_TRANSFORM) != std::string::npos) {
    bn->info.type = Template::BROADCAST_OP;
  } else if (concated_op_type.find(AT_CALL) != std::string::npos) {
    bn->info.type = Template::EXTERN_CALL;
  } else {
    bn->info.type = Template::PURE_ELEM;
  }
}

void AnalyzeBandNode::AnalyzeOuterBandTemplate() {
  auto &bands = scop_info_.analysis_result_.GetBandNodes();
  for (auto &band_node : bands) {
    auto *bn = band_node.get();
    if (!bn->node || bn->node.get_partial_schedule().is_null()) {
      continue;
    }
    isl::union_pw_aff_list aff_list = bn->node.get_partial_schedule().get_union_pw_aff_list();
    for (unsigned int i = 0; i < aff_list.size(); ++i) {
      isl::pw_aff_list pw_list = aff_list.get_at(i).get_pw_aff_list();
      for (unsigned int j = 0; j < pw_list.size(); ++j) {
        isl::pw_aff pw = pw_list.get_at(j);
        std::string stmt_id = pw.domain().get_tuple_name();
        isl::ctx ctx = bn->node.ctx();
        isl::id id(ctx, stmt_id);
        bn->stmts.emplace(id);
      }
    }
    DetermineTemplateOfBand(bn);
  }
}

void AnalyzeBandNode::ShowBandInfo() {
  auto &bands = scop_info_.analysis_result_.GetBandNodes();
  std::stringstream s;
  s << "Outer bands template: {";
  for (size_t i = 0; i < bands.size(); ++i) {
    auto *bn = bands[i].get();
    s << scop_info_.analysis_result_.ShowOpTemplateOfBand(bn->index) << ", ";
  }
  s << "}";
  LOG(INFO) << s.str();
}

}  // namespace poly
}  // namespace ir
}  // namespace akg