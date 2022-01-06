/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
  status = isl_options_set_schedule_treat_coalescing(ctx, 0);
  CHECK(status == isl_stat_ok);

  if (scop_info_.user_config_.GetEnableScheduleOuterCoincidence()) {
    status = isl_options_set_schedule_outer_coincidence(ctx, 1);
    CHECK(status == isl_stat_ok);
  }

  if (scop_info_.user_config_.GetDisableWholeComponent()) {
    status = isl_options_set_schedule_whole_component(ctx, 0);
    CHECK(status == isl_stat_ok);
  } else {
    status = isl_options_set_schedule_whole_component(ctx, 1);
    CHECK(status == isl_stat_ok);
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
  return computed_sch;
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
