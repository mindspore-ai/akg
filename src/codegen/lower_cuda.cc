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

/*
 * The following function will run step by step in function `Lower`, the order is:
 * [CudaLowerBegin -> CudaLowerStageTuning ->
 *  CudaLowerPoly -> CudaLowerBeforeFlattern -> CudaLowerFlattern ->
 *  CudaLowerBeforeRewrite -> CudaLowerRewrite -> CudaLowerBeforeLowerFunc -> CudaLowerDone].
 * > If tuning is active in `CudaLowerStageTuning`, the lower stage in the back will be dropped.
 *
 * - If a extra pass is needed, please insert to the right stage function.
 *   > For example: a new pass `NewPass` is needed between pass `RemoveFakeOp` and `RewriteForTensorCore`.
 *   > You can insert `NewPass` in `CudaLowerBeforeFlattern`(`RemoveFakeOp` and `RewriteForTensorCore` are in
 *     `CudaLowerBeforeFlattern`).
 *
 * - If an early stop is required with a return of Stage, please set `early_stop` in `LowerImpl`.
 *   > For example: a early stop with result of `VectorizeLoop` is desered.
 *   > You can set condition in `CudaLowerBeforeRewrite`(stage which `VectorizeLoop` is in) to early stop,
 *   > and set return value as `{stmt, true}`.
 */

#include <algorithm>
#include <utility>
#include <vector>

#include "ir_pass.h"
#include "codegen/lower.h"
#include "codegen/stage_lower.h"
#include "composite/utils/util.h"

namespace akg {
StageResult CudaLowerBegin(Stmt &, LowerData &data) {
  Stmt stmt = LowerInitWithSchedule(data);

  if (!data->polyhedral) {
    g_attrs.Set(kEnablePolySch, air::make_const(Int(32), false));
    return {stmt, false};
  }
  stmt = NEXT_PASS(ReplaceSeparator, stmt);

  Map<Tensor, Tensor> multi_output_mapping;
  UpdateMultiValueFuncBinds(data->binds_0, data->args, multi_output_mapping);
  stmt = NEXT_PASS(RewriteMultiValueFunc, stmt, multi_output_mapping);

  Map<Tensor, Tensor> replace;
  RenameBinds(data->binds_0, data->config, data->args, data->arg_list_0, replace);
  stmt = NEXT_PASS(RenameRealize, stmt, data->binds_0, replace);

  if (g_attrs.GetBool(kEnableElementwiseFlatten, true)) {
    Array<NodeRef> arg_list_tmp;
    Map<Tensor, Buffer> binds_tmp;
    GetFlattenedBinds(data->args, data->binds_0, data->config, arg_list_tmp, binds_tmp, false);
    Stmt stmt_tmp = NEXT_PASS(ElementwiseFlatten, stmt, data->binds_0, binds_tmp);
    if (stmt_tmp.get() != stmt.get()) {
      stmt = stmt_tmp;
      data->arg_list_0 = arg_list_tmp;
      data->binds_0 = binds_tmp;
    }
  }

  if (g_attrs.GetBool(kEnableFuseAxis, false)) {
    Array<NodeRef> fuse_axis_res = NEXT_PASS(FuseAxis, stmt, data->arg_list_0, data->binds_0);
    CHECK_EQ(fuse_axis_res.size(), 3);
    stmt = air::Downcast<Stmt>(fuse_axis_res[0]);
    data->arg_list_0 = air::Downcast<Array<NodeRef>>(fuse_axis_res[1]);
    data->binds_0 = air::Downcast<Map<Tensor, Buffer>>(fuse_axis_res[2]);
  }
  PassMgr::SetArgs(data->arg_list_0);
  stmt = NEXT_PASS(AddAttrForLayoutOp, stmt, data->sch);
  stmt = NEXT_PASS(RewriteTensorIndex, stmt);
  return {stmt, false};
}

StageResult CudaLowerStageTuning(Stmt &stmt, LowerData &data) {
  // Stage1 is about Tuning things.
  return LowerStageTuning(stmt, data);
}

StageResult CudaLowerPoly(Stmt &stmt, LowerData &data) {
  // Stage2 is about Polyheral.
  return LowerPoly(stmt, data);
}

StageResult CudaLowerBeforeFlattern(Stmt &stmt, LowerData &data) {
  if (data->polyhedral) {
    stmt = NEXT_PASS(LowerWith, stmt);
    if (!g_csr.empty()) {
      stmt = NEXT_PASS(RestoreCsrLoop, stmt, data->binds_0, true);
    }
  }
  stmt = NEXT_PASS(ReconstructLayout, stmt);
  stmt = NEXT_PASS(RemoveFakeOp, stmt);
  stmt = NEXT_PASS(RewriteForTensorCore, stmt, data->sch, data->binds_0);
  return {stmt, false};
}

StageResult CudaLowerFlattern(Stmt &stmt, LowerData &data) {
  // Keep Stage4 do flatten only.
  return LowerFlattern(stmt, data);
}

StageResult CudaLowerBeforeRewrite(Stmt &stmt, LowerData &data) {
  stmt = NEXT_PASS_IF(!data->simple_mode, LoopPartition, stmt, data->config->partition_const_loop);
  stmt = NEXT_PASS_IF(data->config->disable_vectorize, SkipVectorize, stmt);
  stmt = NEXT_PASS_IF(!data->config->disable_vectorize, VectorizeLoop, stmt);
  stmt = NEXT_PASS(InjectVirtualThread, stmt);
  stmt = NEXT_PASS_IF(data->polyhedral, InjectTransferBufferScope, stmt);
  stmt = NEXT_PASS(InjectDoubleBuffer, stmt, data->config->double_buffer_split_loop,
                   g_attrs.GetBool(kEnableDoubleBuffer, false));
  return {stmt, false};
}

StageResult CudaLowerRewrite(Stmt &stmt, LowerData &data) {
  // Keep Stage6 do Storage Rewrite only.
  stmt = NEXT_PASS(StorageRewrite, stmt);
  return {stmt, false};
}

StageResult CudaLowerBeforeLowerFunc(Stmt &stmt, LowerData &data) {
  Target target_platform = Target::Create(data->target);
  stmt =
    NEXT_PASS_IF(target_platform->device_type == kDLGPU && data->polyhedral && g_attrs.GetBool(kEnableSwizzleGPU, true),
                 SwizzleGPU, stmt, g_attrs);
  stmt = NEXT_PASS(UnrollLoop, stmt, data->config->auto_unroll_max_step, data->config->auto_unroll_max_depth,
                   data->config->auto_unroll_max_extent, data->config->unroll_explicit);

  stmt = NEXT_PASS(Simplify, stmt);
  stmt = NEXT_PASS(RemoveNoOp, stmt);
  stmt = NEXT_PASS_IF(data->config->instrument_bound_checkers, InstrumentBoundCheckers, stmt);
  stmt = NEXT_PASS_IF(!data->config->disable_select_rewriting, RewriteUnsafeSelect, stmt);
  stmt = NEXT_PASS_IF(BuildConfig::Current()->detect_global_barrier, ThreadSyncStmt, stmt, "global");
  stmt = NEXT_PASS_IF(!g_attrs.GetBool(kEnablePolySch, false), ThreadSyncStmt, stmt, "shared");
  stmt = NEXT_PASS(ThreadSyncStmt, stmt, "warp");
  stmt = NEXT_PASS(InferFragmentStmt, stmt);
  stmt = NEXT_PASS(LowerThreadAllreduceStmt, stmt, target_platform->thread_warp_size);
  return {stmt, false};
}

StageResult CudaLowerDone(Stmt &stmt, LowerData &data) { return LowerDone(stmt, data); }

namespace lower {
// The order of following register will affect the stage order, please make sure it is right.
REG_STAGE_LOWER("cuda", StageType::Begin, "BEGIN", CudaLowerBegin);
REG_STAGE_LOWER("cuda", StageType::Tuning, "TUNING", CudaLowerStageTuning);
REG_STAGE_LOWER("cuda", StageType::Poly, "POLY", CudaLowerPoly);
REG_STAGE_LOWER("cuda", StageType::BeforeFlattern, "BEFORE_FLATTERN", CudaLowerBeforeFlattern);
REG_STAGE_LOWER("cuda", StageType::Flattern, "FLATTERN", CudaLowerFlattern);
REG_STAGE_LOWER("cuda", StageType::BeforeRewrite, "BEFORE_REWRITE", CudaLowerBeforeRewrite);
REG_STAGE_LOWER("cuda", StageType::Rewrite, "REWRITE", CudaLowerRewrite);
REG_STAGE_LOWER("cuda", StageType::BeforeLowerFunc, "BEFORE_LOWERFUNC", CudaLowerBeforeLowerFunc);
REG_STAGE_LOWER("cuda", StageType::End, "END", CudaLowerDone);

NodeRef CudaLowerImpl(const LowerData &data, bool get_stmt) {
  if (data->target == "cuda" && (data->tuning || g_attrs.GetInt(kHelpTiling, -1) > help_tiling_level["None"])) {
    return StageLower(data).RunTo(StageType::Tuning).Node();
  }
  return StageLower(data).RunTo(get_stmt ? StageType::BeforeLowerFunc : StageType::End).Node();
}
REG_IMPL_LOWER("cuda", CudaLowerImpl);
}  // namespace lower
}  // namespace akg
