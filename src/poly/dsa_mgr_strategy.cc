/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "poly/dsa_mgr_strategy.h"

#include "poly/schedule_pass/group.h"
#include "poly/schedule_pass/tile_outer_band.h"
#include "poly/schedule_pass/memory_manager.h"
#include "poly/schedule_pass/sink_c0.h"
#include "poly/schedule_pass/sink_last_axis.h"
#include "poly/schedule_pass/keep_outer_band_order.h"

#include "poly/schedule_pass/split_outer_band.h"
#include "poly/schedule_pass/transfer_stmt.h"
#include "poly/schedule_pass/reset_coincidence_of_reduce.h"
#include "poly/schedule_pass/set_all_coincidence.h"
#include "poly/schedule_pass/reschedule.h"
#include "poly/schedule_pass/reorder_inner_band.h"
#include "poly/schedule_pass/change_marknode_position.h"
#include "poly/schedule_pass/insert_node_for_allocc.h"
#include "poly/schedule_pass/label_realize_out_position.h"
#include "poly/schedule_pass/mark_fuse_op.h"
#include "poly/schedule_pass/reorder_mark_nodes.h"
#include "poly/schedule_pass/compute_transfer_copyin.h"
#include "poly/schedule_pass/compute_inner_band_dependency.h"
#include "poly/schedule_pass/mark_outer_most.h"
#include "poly/schedule_pass/analyze_schedule.h"

namespace akg {
namespace ir {
namespace poly {

void DsaMgrStrategy::RegisterTilingPasses() { RegisterPass(std::make_shared<TileOuterBand>(pass_info_, scop_info_)); }

void DsaMgrStrategy::RegisterMemPromPasses() {
  RegisterPass(std::make_shared<LabelRealizeOutPosition>());
  if (scop_info_.mmu_info_.IsSpecGemm() || scop_info_.mmu_info_.IsGemm() ||
      scop_info_.mmu_info_.IsConvBackpropFilter()) {
    RegisterPass(std::make_shared<InsertNodeForAllocC>());
  }
  RegisterPass(std::make_shared<MemoryManager>(scop_info_));
  if (!scop_info_.mmu_info_.IsSpecGemm()) {
    RegisterPass(std::make_shared<TransferStmt>(scop_info_, pass_info_));
  }
}

void DsaMgrStrategy::RegisterSchedulePasses() {
#ifdef AKG_USE_MLS
  const bool enable_mlsched = MLSchedShouldBeUsed(scop_info_);
#else
  const bool enable_mlsched = false;
#endif
  if (!enable_mlsched && !scop_info_.user_config_.GetDisableGroup()) {
    RegisterPass(std::make_shared<GroupStatements>(pass_info_));
  }
  RegisterPass(std::make_shared<ComputeSchedule>(pass_info_, scop_info_));
  if (scop_info_.user_config_.GetReorderSchedule()) {
    RegisterPass(std::make_shared<SinkC0>());
  }
  if (scop_info_.user_config_.GetSinkLastAxis()) {
    RegisterPass(std::make_shared<SinkLastAxis>(pass_info_));
  }
  if (scop_info_.user_config_.GetKeepOuterBandOrder()) {
    RegisterPass(std::make_shared<KeepOuterBandOrder>(scop_info_));
  }
  if (!enable_mlsched) {
    RegisterPass(std::make_shared<UnGroupStatements>(pass_info_));
  }
}

void DsaMgrStrategy::RegisterPasses() {
  passes_.clear();
  RegisterNormalizationPasses();
  RegisterConstrainedScheduling();

  RegisterSchedulePasses();

  if (scop_info_.user_config_.GetOuterBandNeedSplit() && !scop_info_.mmu_info_.IsSpecGemm()) {
    RegisterPass(std::make_shared<SplitOuterBand>());
  }
  RegisterPass(std::make_shared<ComputeInnerBandDependency>(scop_info_));
  if (!scop_info_.mmu_info_.IsSpecGemm() && (scop_info_.mmu_info_.IsConv() || scop_info_.mmu_info_.IsGemm())) {
    RegisterPass(std::make_shared<ComputeTransferCopyin>(scop_info_, pass_info_));
  }
  RegisterPass(std::make_shared<AnalyzeSchedule>(scop_info_));
  RegisterTilingPasses();
  if (scop_info_.user_config_.GetIsTuning()) {
    return;
  }
  RegisterPass(std::make_shared<ResetCoincidenceOfReduce>(scop_info_, pass_info_));
  if (scop_info_.user_config_.GetPragmaSetAllCoincident()) {
    RegisterPass(std::make_shared<SetAllCoincidence>());
  }
  if (scop_info_.user_config_.GetEnableReschedule() &&
      (!scop_info_.user_config_.GetIsDynamic() || !scop_info_.mmu_info_.IsConv())) {
    RegisterPass(std::make_shared<Reschedule>(scop_info_, pass_info_));
  }
  RegisterPass(std::make_shared<ReorderInnerBand>(scop_info_.analysis_result_.GetCondVarsMap()));
  RegisterPass(std::make_shared<ChangeMarkNodePosition>(scop_info_.analysis_result_.ExtractWithStmtId()));
  RegisterMemPromPasses();
  RegisterPass(std::make_shared<ReorderMarkNodes>());
  RegisterPass(std::make_shared<MarkFuseOp>(scop_info_));
  if (!scop_info_.mmu_info_.IsSpecGemm()) {
    RegisterPass(std::make_shared<MarkOuterMost>(scop_info_));
  }
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
