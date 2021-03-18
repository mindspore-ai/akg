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
#include "init_schedule.h"

namespace akg {
namespace ir {
namespace poly {

void InitSchedule::RemoveUninitializedCopyin(isl::union_map &copy_in, const Binds &binds) {
  isl::union_set copyin_range = copy_in.range();
  auto ForeachSetFn = [&copy_in, &binds](const isl::set &set) {
    std::string tensor_name = set.get_tuple_name();
    bool defined = false;
    for (auto bind : binds) {
      if (bind.first->op->name == tensor_name) {
        defined = true;
      }
    }
    if (!defined) {
      copy_in = copy_in.subtract_range(set);
      LOG(WARNING) << "remove uninitialized copyin, tensor name=" << tensor_name << ", access=" << set;
    }
  };
  copyin_range.foreach_set(ForeachSetFn);
}

void InitSchedule::ModDependencesBeforeGroup(const isl::schedule &schedule) {
  if (!scop_info_.mmu_info_.IsSpecGemm()) {
    if (scop_info_.user_config_.GetRemoveSelfDependence()) {
      pass_info_.dependences_ = RemoveReduceOpSelfDependence(scop_info_, pass_info_);
    }

    if (scop_info_.user_config_.GetForceRemoveSelfDependence()) {
      pass_info_.dependences_ = RemoveSelfDependence(pass_info_);
    }
  }

  if (scop_info_.user_config_.GetRemoveInvariantDependence()) {
    pass_info_.dependences_ = RemoveInvariantDependence(schedule, pass_info_, scop_info_);
  }
}

void InitSchedule::ComputeCopyIn(const isl::schedule &schedule) {
  auto reads = scop_info_.analysis_result_.GetReads().domain_factor_domain();
  auto writes = scop_info_.analysis_result_.GetWrites().domain_factor_domain();
  auto uai = isl::union_access_info(reads);
  uai = uai.set_kill(writes);
  uai = uai.set_may_source(writes);
  uai = uai.set_schedule(schedule);
  auto flow = uai.compute_flow();
  auto mayNoSource = flow.get_may_no_source();
  scop_info_.analysis_result_.RecordCopyin(scop_info_.analysis_result_.GetReads().intersect_range(mayNoSource.range()));
}

/*
 * Force dependences between multiple liveout operations. This function should
 * only be called when there exist multiple liveouts. As the fusion heuristic
 * of the isl scheduler only fuses operations that depend on each other, one
 * may have to use this function to introduce additional dependences between
 * liveouts that are originally independent.
 *
 * Note that introducing dependences between two operations never violates the
 * semantics of the program; it is thus safe to call this function and unnecessary
 * to check the correctness. However, redundant fake dependences may hamper the
 * tilabily and parallelism, one should introduce as few dependences as possible.
 *
 * This function first traverses the sets in "liveouts", and obtains the sample
 * points of each set, from which two union_sets are constructed. This is to
 * guarantee the minimum of the introduced dependences. The dependence "dep"
 * contructed from this two union_sets are then added to the dependences of the
 * program.
 *
 * TODO: prohibit forcely introducing dependences when the intermediate
 * operations of different liveouts intersect?
 */
void InitSchedule::ForceDepBetweenLiveouts(const isl::union_set liveouts) {
  auto list = liveouts.get_set_list();
  auto n = liveouts.n_set();
  for (unsigned i = 1; i < n; i++) {
    auto pnt1 = list.get_at(0).sample_point();
    auto pnt2 = list.get_at(i).sample_point();
    auto dep = isl::union_map::from_domain_and_range(isl::union_set(pnt1), isl::union_set(pnt2));
    pass_info_.force_dependences_ = pass_info_.force_dependences_.unite(dep);
  }
}

/*
 * Remove all the self-dependences if they are not depended by other statements.
 */
isl::union_map InitSchedule::RemoveLeafSelfDependence(const isl::union_map &dependences) {
  isl::union_map preserved = isl::union_map::empty(dependences.get_space());
  isl::union_map removed = isl::union_map::empty(dependences.get_space());
  dependences.foreach_map([&preserved, &removed](const isl::map &m) -> void {
    if (m.domain().get_tuple_id() != m.range().get_tuple_id()) {
      preserved = preserved.add_map(m);
    } else {
      removed = removed.add_map(m);
    }
  });
  removed.foreach_map([&preserved](const isl::map &m) -> void {
    if (!preserved.intersect_domain(isl::union_set(m.domain()).universe()).is_empty()) {
      preserved = preserved.add_map(m);
    }
  });
  return preserved;
}

isl::schedule InitSchedule::Run(isl::schedule sch) {
  ComputeCopyIn(sch);
  RemoveUninitializedCopyin(scop_info_.analysis_result_.GetCopyin(), scop_info_.user_config_.GetOriginBind());

  pass_info_.dependences_ =
    ComputeAllDependences(sch, scop_info_.analysis_result_.GetReads(), scop_info_.analysis_result_.GetWrites());
  /*
   * Collect all statements into a union_set that do not appear as a source of a dependence.
   * When union_set is not a set, i.e., there exist multiple liveouts, introduce dependences
   * between these liveouts by calling ForceDepBetweenLiveouts.
   */
  if (scop_info_.user_config_.GetTarget() == TARGET_CUDA) {
    pass_info_.force_dependences_ = isl::union_map::empty(sch.ctx());
    auto sinks = RemoveLeafSelfDependence(pass_info_.dependences_).domain();
    auto domain = sch.get_root().as<isl::schedule_node_domain>().get_domain();
    sinks = domain.subtract(sinks);
    if (!sinks.isa_set()) {
      ForceDepBetweenLiveouts(sinks);
      pass_info_.dependences_ = pass_info_.dependences_.unite(pass_info_.force_dependences_);
    }
  }

  pass_info_.orig_dependences_ = pass_info_.dependences_;

  if (scop_info_.user_config_.GetTarget() != TARGET_CUDA) {
    ModDependencesBeforeGroup(sch);
  }

  return sch;
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
