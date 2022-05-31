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
#include "poly/schedule_tree_util.h"
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

isl::schedule ComputeSchedule::AdjustInplaceAssignOrder(const isl::schedule &sch) {
  isl::schedule_node outer_band_node = GetOuterBand(sch.get_root());

  auto inplace_assign_nodes = scop_info_.analysis_result_.GetInplaceAssignNodes();
  if (!outer_band_node.isa<isl::schedule_node_set>() || inplace_assign_nodes.empty()) {
    return sch;
  }

  std::unordered_set<std::string> inplace_assign_stmts;
  for (auto &op : inplace_assign_nodes) {
    for (auto &stmt : scop_info_.analysis_result_.GetStatementMap()) {
      if (op == stmt.second) {
        inplace_assign_stmts.emplace(stmt.first.get_name());
        break;
      }
    }
  }

  auto set_node = outer_band_node.as<isl::schedule_node_set>();
  auto after_set_node = outer_band_node;
  std::vector<size_t> new_pos;
  std::vector<size_t> after_pos;
  for (size_t i = 0; i < set_node.n_children(); ++i) {
    auto child_node = set_node.child(i);
    isl::union_set filter_set = child_node.as<isl::schedule_node_filter>().filter();
    bool is_inplace_assign = false;
    filter_set.foreach_set([inplace_assign_stmts, &is_inplace_assign](isl::set s) {
      if (inplace_assign_stmts.count(s.get_tuple_name()) != 0) {
        is_inplace_assign = true;
        return;
      }
    });

    if (is_inplace_assign) {
      after_pos.push_back(i);
    } else {
      new_pos.push_back(i);
    }
  }

  new_pos.insert(new_pos.end(), after_pos.begin(), after_pos.end());
  auto domain_node = isl::schedule_node::from_domain(sch.get_domain());
  auto new_node = ReConstructChildScheduleTree(domain_node, sch.get_root(), outer_band_node);
  auto empty_node = isl::schedule_node();
  new_node = ReConstructSetOrSequenceNode(GetOuterBand(new_node), outer_band_node, empty_node, new_pos);
  return new_node.get_schedule();
}

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
    mls::bin::Options options = MLSchedOptionsInit(pass_info_, scop_info_);
    if (options.ShouldLogInternalDebugging()) {
      LOG(INFO) << "MLSched v." << mls::bin::VersionString() << std::endl;
      LOG(INFO) << options.String() << std::endl;
    }

    const std::string &kernel_name = scop_info_.user_config_.GetKernelName();
    const mls::bin::Hints hints = ExtractDirectivesFromAKG(scop_info_);
    const isl::union_map reads = UnwrappedAccesses(scop_info_.analysis_result_.GetReads());
    const isl::union_map writes = UnwrappedAccesses(scop_info_.analysis_result_.GetWrites());
    isl_union_map *const dependences = pass_info_.dependences_.get();
    mls::bin::Scop scop(sch.get(), dependences, reads.get(), writes.get(), hints, options, kernel_name.c_str());
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

  result = AdjustInplaceAssignOrder(result);
  return result;
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
