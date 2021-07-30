/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "gpu_dma_analysis.h"
#include "poly/scop.h"

namespace akg {
namespace ir {
namespace poly {

isl::schedule GpuDmaAnalysis::Run(isl::schedule sch) {
  auto original_setting = scop_info_.user_config_.GetPragmaSpeedUpTiling();
  scop_info_.user_config_.SetPragmaSpeedUpTiling(true);
  scop_info_.analysis_result_.SetIsGpuDmaAnalysed(true);

  schedule_ = sch;
  Analysis();
  RemoveInjectiveTensorFromMemFlows(sch);

  scop_info_.user_config_.SetPragmaSpeedUpTiling(original_setting);
  scop_info_.analysis_result_.SetIsGpuDmaAnalysed(false);
  return schedule_;
}

void GpuDmaAnalysis::RemoveInjectiveTensorFromMemFlows(isl::schedule schedule) {
  isl::schedule_node root = schedule.get_root();
  isl::schedule_node node = GetOuterBand(root);
  if (node.isa<isl::schedule_node_band>()) {
    const bool schedule_from_mindtrick = scop_info_.user_config_.GetMindTrickWasUsed();
    const bool mindtrick_has_mapping = scop_info_.user_config_.GetMindTrickGpuHasMapping();
    if (!schedule_from_mindtrick && !mindtrick_has_mapping) {
      node = GetTiledNode(schedule, node);
    }
    ResetMemFlows(root, node);
  }
}

void GpuDmaAnalysis::ResetMemFlows(isl::schedule_node root, isl::schedule_node node) {
  auto partial_sched_mupa = ShortScheduleMupa(root, node);

  isl::union_map reads = scop_info_.analysis_result_.GetReads();
  isl::union_map writes = scop_info_.analysis_result_.GetWrites();

  auto reads_access = reads.domain_factor_domain();
  auto writes_access = writes.domain_factor_domain();
  auto original_access = reads_access.unite(writes_access);
  isl::union_map out_schedules = isl::union_map::from(partial_sched_mupa);
  std::map<std::string, MemFlow> tensor_mem_flows = scop_info_.analysis_result_.GetTensorMemFlows();
  std::map<std::string, std::vector<std::string>> tensor_name_flows = scop_info_.analysis_result_.GetTensorNameFlows();

  /* Record the tensor union_map info, such as,
   mapping statements, injective and bijective properties for auto tiling */
  TensorScheduleRepo tensor_repo;

  std::map<std::string, isl::union_map> tensors_map;
  std::map<std::string, std::vector<isl::id>> tensor_state_map;
  for (auto access : original_access.get_map_list()) {
    std::string tensor = access.get_tuple_id(isl_dim_out).to_str();
    if (tensors_map.find(tensor) == tensors_map.end()) {
      tensors_map[tensor] = isl::union_map(access);
    } else {
      tensors_map[tensor] = tensors_map[tensor].unite(isl::union_map(access));
    }
    if (tensor_state_map.find(tensor) == tensor_state_map.end()) {
      tensor_state_map.insert({tensor, {access.get_tuple_id(isl_dim_in)}});
    } else {
      tensor_state_map[tensor].push_back(access.get_tuple_id(isl_dim_in));
    }
  }

  MemFlow orig_mem_flow;
  orig_mem_flow.push_back(MemType::DDR);

  for (auto it = tensors_map.begin(); it != tensors_map.end(); ++it) {
    auto tmp_out_schedules = out_schedules.range_product(it->second);
    if (tmp_out_schedules.is_injective()) {
      tensor_mem_flows[it->first] = orig_mem_flow;
      std::vector<std::string> orig_name_flow;
      orig_name_flow.push_back(it->first);
      tensor_name_flows[it->first] = orig_name_flow;
    }
    if (tensor_repo.find(it->first) == tensor_repo.end()) {
      StatementUnionMappingInfo info;
      if (tensor_state_map.find(it->first) != tensor_state_map.end()) {
        info.stmt_vec = tensor_state_map[it->first];
      }
      info.inject_mapping = tmp_out_schedules.is_injective();
      info.biject_mapping = tmp_out_schedules.is_bijective();
      tensor_repo.insert({it->first, info});
    }
  }

  scop_info_.analysis_result_.SetTensorMemFlows(tensor_mem_flows);
  scop_info_.analysis_result_.SetTensorNameFlows(tensor_name_flows);
  scop_info_.analysis_result_.SetTensorScheduleRepo(tensor_repo);
}

isl::schedule_node GpuDmaAnalysis::GetTiledNode(isl::schedule schedule, isl::schedule_node node) {
  auto space = node.as<isl::schedule_node_band>().get_space();
  isl::ctx ctx = space.ctx();
  auto tile_size = isl::multi_val::zero(space);

  auto partial_schedule = node.as<isl::schedule_node_band>().get_partial_schedule();
  auto upa_list = partial_schedule.get_union_pw_aff_list().reverse();
  auto tiling_res = GenerateTiling(schedule, scop_info_, GenHalide(scop_info_, schedule, true));
  auto title_size = static_cast<unsigned int>(tiling_res.first.size());
  const unsigned int n_member = node.as<isl::schedule_node_band>().n_member();
  unsigned int dim_num = (n_member <= title_size) ? n_member : title_size;

  std::vector<int> tmp_sizes(n_member, 0);
  for (size_t j = 0; j < n_member; ++j) {
    tmp_sizes[j] = MAX_STRIDE;
    if (j < dim_num) tmp_sizes[j] = static_cast<int>(tiling_res.first[j].c1_tiling_size);
  }

  for (unsigned int i = 0; i < n_member; ++i) {
    tile_size = tile_size.set_val(i, isl::val(ctx, tmp_sizes[i]));
  }

  return TileBand(node, tile_size);
}

void GpuDmaAnalysis::Analysis() {
  auto shared_tensor_list = SharedTensorAnalysis();
  auto all_tensors = AllTensors();
  auto local_tensor_list = LocalTensorAnalysis(all_tensors, shared_tensor_list);
  SetTensorFlow(shared_tensor_list, local_tensor_list);
}

std::vector<isl::id> GpuDmaAnalysis::SharedTensorAnalysis() {
  auto read_map = scop_info_.StmtReadMap();
  auto write_map = scop_info_.StmtWriteMap();
  auto stmt_map = scop_info_.analysis_result_.GetStmtOpInfoMap();
  std::vector<isl::id> tensor_list;
  std::set<std::string> id_sets;
  std::set<std::string> read_sets;
  std::set<std::string> write_sets;
  for (auto item : read_map) {
    for (auto item_id : item.second) {
      if (read_sets.count(item_id.get_name()) == 0) {
        read_sets.insert(item_id.get_name());
      }
    }
  }
  for (auto item : write_map) {
    for (auto item_id : item.second) {
      if (write_sets.count(item_id.get_name()) == 0) {
        write_sets.insert(item_id.get_name());
      }
    }
  }

  /*********************************************************
   * manage only read tensors to share memory
   * for read and write tensor, should be managed to local memory
   ********************************************************/
  std::set_difference(read_sets.begin(), read_sets.end(), write_sets.begin(), write_sets.end(),
                      std::inserter(id_sets, id_sets.begin()));

  // mark reduce out tensors for auto tiling
  if (scop_info_.user_config_.GetTarget() == TARGET_CUDA && scop_info_.user_config_.GetEnableAkgReduceLib()) {
    auto reduce_out_tensors = scop_info_.analysis_result_.GetReduceTensorInfoMap();
    if (!reduce_out_tensors.empty()) {
      id_sets.clear();
      for (auto tensor : reduce_out_tensors) {
        id_sets.emplace(tensor.second.write_tensor_name);
      }
    }
  }

  if (!configed_share_tensors_.empty()) {
    id_sets.clear();
    for (const auto &item : configed_share_tensors_) {
      if (id_sets.count(item) == 0) {
        id_sets.emplace(item);
      }
    }
  }
  for (auto item : id_sets) {
    tensor_list.push_back(isl::id(scop_info_.ctx_, item));
  }
  return tensor_list;
}

TensorSets GpuDmaAnalysis::AllTensors() {
  auto read_map = scop_info_.StmtReadMap();
  auto write_map = scop_info_.StmtWriteMap();
  TensorSets id_sets;
  for (auto item : read_map) {
    for (auto item_id : item.second) {
      id_sets.insert(item_id);
    }
  }
  for (auto item : write_map) {
    for (auto item_id : item.second) {
      id_sets.insert(item_id);
    }
  }
  return id_sets;
}

std::vector<isl::id> GpuDmaAnalysis::LocalTensorAnalysis(const TensorSets &all_tensors,
                                                         const std::vector<isl::id> &shared_tensors) {
  std::set<std::string> shared_tensor_ids;
  for (auto item : shared_tensors) {
    shared_tensor_ids.insert(item.get_name());
  }

  std::vector<isl::id> local_tensors;
  for (auto item : all_tensors) {
    if (shared_tensor_ids.count(item.get_name()) == 0) {
      local_tensors.push_back(item);
    }
  }

  return local_tensors;
}

void GpuDmaAnalysis::SetTensorFlow(const std::vector<isl::id> &shares, const std::vector<isl::id> &locals) {
  std::map<std::string, MemFlow> tensor_mem_flows;
  std::map<std::string, std::vector<std::string>> tensor_name_flows = scop_info_.analysis_result_.GetTensorNameFlows();
  MemFlow shared_flow;
  shared_flow.push_back(MemType::DDR);
  shared_flow.push_back(MemType::SHARED_);
  for (auto item : shares) {
    if (tensor_mem_flows.count(item.name()) == 0) {
      tensor_mem_flows[item.name()] = shared_flow;
    }
    std::vector<std::string> shared_name_flow;
    shared_name_flow.push_back(item.name());
    shared_name_flow.push_back(item.name() + "_shared");
    tensor_name_flows[item.name()] = shared_name_flow;
  }

  MemFlow local_flow;
  local_flow.push_back(MemType::DDR);
  local_flow.push_back(MemType::LOCAL_);
  for (auto item : locals) {
    if (tensor_mem_flows.count(item.name()) == 0) {
      tensor_mem_flows[item.name()] = local_flow;
    }
    std::vector<std::string> local_name_flow;
    local_name_flow.push_back(item.name());
    local_name_flow.push_back(item.name() + "_local");
    tensor_name_flows[item.name()] = local_name_flow;
  }

  scop_info_.analysis_result_.SetTensorMemFlows(tensor_mem_flows);
  scop_info_.analysis_result_.SetTensorNameFlows(tensor_name_flows);
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
