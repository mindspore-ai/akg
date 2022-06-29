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
  if (scop_info_.user_config_.GetEnableAtomicAdd()) {
    auto atomic_info = scop_info_.analysis_result_.GetReduceOutStmtIdToTensor();
    if (!atomic_info.empty()) {
      pass_info_.dependences_ = RemoveSelfDependence(pass_info_, atomic_info);
    }
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

isl::schedule InitSchedule::Run(isl::schedule sch) {
  ComputeCopyIn(sch);
  RemoveUninitializedCopyin(scop_info_.analysis_result_.GetCopyin(), scop_info_.user_config_.GetOriginBind());

  pass_info_.dependences_ =
    ComputeAllDependences(sch, scop_info_.analysis_result_.GetReads(), scop_info_.analysis_result_.GetWrites());
  auto tot_stmt = scop_info_.analysis_result_.GetTensorOfTensorStmt();
  if (!tot_stmt.empty()) {
    if (scop_info_.user_config_.GetTarget() == TARGET_CUDA ||
        (scop_info_.user_config_.GetTarget() == TARGET_CPU && scop_info_.analysis_result_.GetRemoveSelfDependence())) {
      pass_info_.dependences_ = RemoveSelfDependence(pass_info_, tot_stmt);
    }
  }

  pass_info_.orig_dependences_ = pass_info_.dependences_;

  if (scop_info_.user_config_.GetTarget() == TARGET_CCE) {
    ModDependencesBeforeGroup(sch);
  }

  scop_info_.origin_schedule_ = sch;
  return sch;
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
