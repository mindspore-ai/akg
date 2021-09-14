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
#include "compute_schedule.h"

namespace akg {
namespace ir {
namespace poly {

isl::union_map ComputeSchedule::ModDependences(const isl::union_map &dependences) {
  isl::union_map umap = isl::union_map::empty(dependences.ctx());
  dependences.foreach_map([&](const isl::map &m) -> void {
    isl::map mm = m;
    if (mm.get_tuple_id(isl_dim_in) != mm.get_tuple_id(isl_dim_out)) {
      isl_map *pmap = mm.copy();
      int n_in = isl_map_dim(pmap, isl_dim_in);
      for (int i = 0; i < n_in; ++i) {
        pmap = isl_map_plain_update_val_if_fixed(pmap, isl_dim_in, i);
      }
      mm = isl::manage(pmap);
    }
    umap = umap.unite(isl::union_map(mm));
  });
  return umap;
}

void ComputeSchedule::SetIslOptions() {
  auto ctx = pass_info_.constraints_.ctx().get();
  int status = isl_options_set_schedule_unit_max_var_coefficient_sum(ctx, 1);
  CHECK(status == isl_stat_ok);

  if (scop_info_.user_config_.GetTarget() != TARGET_CUDA) {
    if (scop_info_.user_config_.GetDisableWholeComponent()) {
      status = isl_options_set_schedule_whole_component(ctx, 0);
      CHECK(status == isl_stat_ok);
    } else {
      status = isl_options_set_schedule_maximize_coincidence(ctx, 0);
      CHECK(status == isl_stat_ok);
      status = isl_options_set_schedule_whole_component(ctx, 1);
      CHECK(status == isl_stat_ok);
    }
  } else {
    if (scop_info_.user_config_.GetComputeReschedule()) {
      status = isl_options_set_schedule_whole_component(ctx, 0);
      CHECK(status == isl_stat_ok);
    } else {
      status = isl_options_set_schedule_maximize_coincidence(ctx, 0);
      CHECK(status == isl_stat_ok);
      status = isl_options_set_schedule_whole_component(ctx, 1);
      CHECK(status == isl_stat_ok);
    }
  }

  if (scop_info_.user_config_.GetDisableScheduleShift()) {
    status = isl_options_set_schedule_max_constant_term(ctx, 0);
    CHECK(status == isl_stat_ok);
    status = isl_options_set_schedule_nonneg_var_coefficient(ctx, 1);
    CHECK(status == isl_stat_ok);
  }

  if (scop_info_.user_config_.GetEnableScheduleMaxConstant()) {
    status = isl_options_set_schedule_max_constant_term(ctx, 0);
    CHECK(status == isl_stat_ok);
  }

  if (scop_info_.user_config_.GetDisableLoopReversal()) {
    status = isl_options_set_schedule_nonneg_var_coefficient(ctx, 1);
    CHECK(status == isl_stat_ok);
  }

  if (scop_info_.user_config_.GetDisableLoopFusion()) {
    status = isl_options_set_schedule_serialize_sccs(ctx, 1);
    CHECK(status == isl_stat_ok);
  }
}

isl::schedule ComputeSchedule::PermuteOuterBand(const isl::schedule sch) {
  if (!scop_info_.analysis_result_.GetUseGpuReduceLib()) {
    return sch;
  }
  auto node = GetOuterBand(sch.get_root());
  if (!node.isa<isl::schedule_node_band>()) {
    return sch;
  }

  // We only permute reduction case
  auto original_band = node.as<isl::schedule_node_band>();
  auto permutable = original_band.get_permutable();
  if (!permutable || original_band.n_member() != 2) {
    return sch;
  }
  std::unordered_map<int, int> coincidence;
  for (int i = 0; i < static_cast<int>(original_band.n_member()); ++i) {
    coincidence[i] = original_band.member_get_coincident(i);
  }

  auto domain = sch.get_domain();
  auto paritial_schedule = original_band.get_partial_schedule();
  auto upa_list = paritial_schedule.get_union_pw_aff_list();
  if (upa_list.size() != 2) {
    return sch;
  }
  isl::union_pw_aff_list new_partial_schedule(original_band.ctx(), original_band.n_member());
  auto aff_with_domain1 = upa_list.get_at(0).intersect_domain(domain);
  auto aff_with_domain2 = upa_list.get_at(1).intersect_domain(domain);
  long reduce_extent1 = CollectReductionExtent(aff_with_domain1);
  long reduce_extent2 = CollectReductionExtent(aff_with_domain2);
  if (reduce_extent1 == -1 || reduce_extent2 == -1 || reduce_extent1 == reduce_extent2 || reduce_extent1 == 1 ||
      reduce_extent2 == 1) {
    return sch;
  }
  std::unordered_set<std::string> swap_ids1 = CollectSwapIds(aff_with_domain1, reduce_extent1);
  std::unordered_set<std::string> swap_ids2 = CollectSwapIds(aff_with_domain2, reduce_extent2);
  if (swap_ids1.size() != swap_ids2.size()) {
    return sch;
  }
  for (auto it = swap_ids1.begin(); it != swap_ids1.end(); ++it) {
    if (swap_ids2.find(*it) == swap_ids2.end()) {
      return sch;
    }
  }
  // generate new aff

  new_partial_schedule = new_partial_schedule.add(GenerateNewAffine(upa_list.get_at(0), upa_list.get_at(1), swap_ids1));
  new_partial_schedule = new_partial_schedule.add(GenerateNewAffine(upa_list.get_at(1), upa_list.get_at(0), swap_ids1));
  isl::multi_union_pw_aff mupa = isl::multi_union_pw_aff(paritial_schedule.space(), new_partial_schedule);
  auto child = node.del();
  child = child.insert_partial_schedule(mupa);
  auto new_node = GetOuterBand(child.get_schedule().get_root());
  if (!new_node.isa<isl::schedule_node_band>()) {
    return sch;
  }
  auto new_band = new_node.as<isl::schedule_node_band>();
  new_band = new_band.set_permutable(permutable);
  if (new_band.n_member() != coincidence.size()) {
    return sch;
  }
  for (int i = 0; i < static_cast<int>(new_band.n_member()); ++i) {
    auto it = coincidence.find(i);
    CHECK(it != coincidence.end());
    new_band = new_band.member_set_coincident(i, it->second);
  }
  return new_band.get_schedule();
}

long ComputeSchedule::CollectReductionExtent(const isl::union_pw_aff &aff) {
  long reduce_extent = -1;
  auto pw_aff_list = aff.get_pw_aff_list();
  for (size_t i = 0; i < pw_aff_list.size(); ++i) {
    auto pw = pw_aff_list.get_at(i);
    auto id = pw.domain().get_tuple_id();
    if (scop_info_.analysis_result_.GetReduceTensorInfoMap().find(id) ==
        scop_info_.analysis_result_.GetReduceTensorInfoMap().end()) {
      continue;
    }
    isl::union_pw_aff single_aff = isl::union_pw_aff(pw);
    auto extent = single_aff.max_val().get_num_si() + 1;
    if (reduce_extent == -1) {
      reduce_extent = extent;
    } else if (reduce_extent != extent) {  // conflict reduction extent, exit.
      return -1;
    }
  }
  return reduce_extent;
};

std::unordered_set<std::string> ComputeSchedule::CollectSwapIds(const isl::union_pw_aff &aff,
                                                                const long reduce_extent) {
  std::unordered_set<std::string> ids;
  auto pw_aff_list = aff.get_pw_aff_list();
  for (size_t i = 0; i < pw_aff_list.size(); ++i) {
    auto pw = pw_aff_list.get_at(i);
    auto id = pw.domain().get_tuple_id();
    if (scop_info_.analysis_result_.GetReduceTensorInfoMap().find(id) !=
          scop_info_.analysis_result_.GetReduceTensorInfoMap().end() ||
        scop_info_.analysis_result_.IsReduceInitStmt(id)) {
      continue;
    }
    isl::union_pw_aff single_aff = isl::union_pw_aff(pw);
    auto extent = single_aff.max_val().get_num_si() + 1;
    if (extent != reduce_extent) {
      ids.insert(id.get_name());
    }
  }
  return ids;
};

isl::union_pw_aff ComputeSchedule::GenerateNewAffine(const isl::union_pw_aff &swap_out,
                                                     const isl::union_pw_aff &swap_in,
                                                     std::unordered_set<std::string> swap_ids) {
  isl::union_pw_aff new_aff;

  // keep reduction stmts in original aff
  isl::pw_aff_list pw_out = swap_out.get_pw_aff_list();
  for (size_t i = 0; i < pw_out.size(); ++i) {
    isl::pw_aff pw = pw_out.get_at(i);
    auto id_name = pw.domain().get_tuple_name();
    if (swap_ids.find(id_name) == swap_ids.end()) {
      if (new_aff.is_null()) {
        new_aff = isl::union_pw_aff(pw);
      } else {
        new_aff = new_aff.union_add(isl::union_pw_aff(pw));
      }
    }
  }

  // swap in element-wise stmt in other aff
  isl::pw_aff_list pw_in = swap_in.get_pw_aff_list();
  for (size_t i = 0; i < pw_in.size(); ++i) {
    isl::pw_aff pw = pw_in.get_at(i);
    auto id_name = pw.domain().get_tuple_name();
    if (swap_ids.find(id_name) != swap_ids.end()) {
      if (new_aff.is_null()) {
        new_aff = isl::union_pw_aff(pw);
      } else {
        new_aff = new_aff.union_add(isl::union_pw_aff(pw));
      }
    }
  }
  return new_aff;
};

isl::schedule ComputeSchedule::Run(isl::schedule sch) {
  if (scop_info_.user_config_.GetModScheduleShift()) {
    pass_info_.dependences_ = ModDependences(pass_info_.dependences_);
  }
  pass_info_.constraints_ = MakeScheduleConstraints(sch, pass_info_);
  SetIslOptions();
  auto computed_sch = pass_info_.constraints_.compute_schedule();
  if (scop_info_.user_config_.GetTarget() == TARGET_CUDA) {
    computed_sch = PermuteOuterBand(computed_sch);
  }
  return computed_sch;
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
