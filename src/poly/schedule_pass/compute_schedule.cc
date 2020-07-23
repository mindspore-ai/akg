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

  if (scop_info_.user_config_.GetComputeReschedule()) {
    status = isl_options_set_schedule_whole_component(ctx, 0);
    CHECK(status == isl_stat_ok);
  } else {
    status = isl_options_set_schedule_maximize_coincidence(ctx, 0);
    CHECK(status == isl_stat_ok);
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

isl::schedule ComputeSchedule::Run(isl::schedule sch) {
  if (scop_info_.user_config_.GetModScheduleShift()) {
    pass_info_.dependences_ = ModDependences(pass_info_.dependences_);
  }
  pass_info_.constraints_ = MakeScheduleConstraints(sch, pass_info_);
  SetIslOptions();
  return pass_info_.constraints_.compute_schedule();
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
