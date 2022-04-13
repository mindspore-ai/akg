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

#include "poly/isl_util.h"

#ifdef AKG_USE_MLS
#include "poly/mls.h"
#endif

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

  if (scop_info_.user_config_.GetEnableScheduleMaximizeCoincidence()) {
    status = isl_options_set_schedule_maximize_coincidence(ctx, 1);
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

#ifdef AKG_USE_MLS
mls::bin::Hints ComputeSchedule::ExtractDirectivesFromAKG(void) {
  mls::bin::Hints hints;

  ForTypeMap directives = scop_info_.analysis_result_.GetForTypeMap();
  std::map<std::string, std::vector<int>> serials_dir;
  std::map<std::string, std::vector<int>> vectorials_dir;
  std::map<std::string, std::vector<int>> parallels_dir;
  for (const auto &[stmt, vloop_directive] : directives) {
    std::string stmt_string = stmt.get_name();
    for (uint i = 0; i < vloop_directive.size(); ++i) {
      switch (vloop_directive[i]) {
        case ForType::Serial:
          break;
        case ForType::Invariant:
          LOG(INFO) << stmt_string << "invariant_for";
          serials_dir[stmt_string].push_back(i);
          break;
        case ForType::Parallel:
          LOG(INFO) << stmt_string << "parallel";
          parallels_dir[stmt_string].push_back(i);
          break;
        case ForType::Vectorized:
        case ForType::Swizzled:  // treat "Swizzled" like "Vectorized" for the moment
          LOG(INFO) << stmt_string << "vectorized";
          vectorials_dir[stmt_string].push_back(i);
          break;
        case ForType::Unrolled:
          LOG(WARNING) << stmt_string << "Do not treat ForType::Unrolled as a directives";
          break;
      }
    }
  }

  hints.SetSerials(serials_dir);
  hints.SetVectorials(vectorials_dir);
  return hints;
}
#endif

isl::schedule ComputeSchedule::Run(isl::schedule sch) {
  if (scop_info_.user_config_.GetModScheduleShift()) {
    pass_info_.dependences_ = ModDependences(pass_info_.dependences_);
  }
  pass_info_.constraints_ = MakeScheduleConstraints(sch, pass_info_);

#ifdef AKG_USE_MLS
  const bool enable_mlsched = MLSchedShouldBeUsed(scop_info_);
  bool enable_isl = !enable_mlsched;

  isl::schedule result;
  if (enable_mlsched) {
    mls::bin::Options options = MLSchedOptionsInit(scop_info_);
    if (options.ShouldLogInternalDebugging()) {
      LOG(INFO) << "MLSched v." << mls::bin::VersionString() << std::endl;
      LOG(INFO) << options.String() << std::endl;
    }

    const std::string &kernel_name = scop_info_.user_config_.GetKernelName();
    mls::bin::Scop scop(sch.get(), pass_info_.dependences_.get(), ExtractDirectivesFromAKG(), options, kernel_name);
    const bool mlsched_success = scop.ComputeSchedule();
    if (options.ShouldLogInternalDebugging()) {
      LOG(INFO) << scop.String(options) << std::endl;
    }

    if (mlsched_success) {
      result = isl::manage(scop.ToIslSchedule(sch.ctx().get()));
    } else {
      enable_isl = true;
    }
  }

  // Schedule with isl if MLSched is disabled or cannot return a schedule
  if (!enable_mlsched || enable_isl) {
    SetIslOptions();
    result = pass_info_.constraints_.compute_schedule();
  }
#else
  SetIslOptions();
  result = pass_info_.constraints_.compute_schedule();
#endif

  return result;
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
