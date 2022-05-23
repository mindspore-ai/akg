/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
 * [LLVMLowerBegin -> LLVMLowerStageTuning ->
 *  LLVMLowerPoly -> LLVMLowerBeforeFlattern -> LLVMLowerFlattern -> CudaLowerDone].
 *
 * > If tuning is active in `LLVMLowerStageTuning`, the lower stage in the back will be dropped.
 *
 */

#include <algorithm>
#include <utility>
#include <vector>

#include "build_module.h"
#include "ir_pass.h"
#include "codegen/lower.h"
#include "codegen/stage_lower.h"
#include "composite/utils/util.h"
#include "pass/utils.h"

namespace akg {
StageResult LLVMLowerBegin(Stmt &, LowerData &data) {
  Stmt stmt = LowerInitWithSchedule(data);

  if (!data->polyhedral) {
    g_attrs.Set(kEnablePolySch, air::make_const(Int(32), false));
    return {stmt, false};
  }
  stmt = NEXT_PASS(ReplaceSeparator, stmt);
  stmt = NEXT_PASS(RewriteMultiValueFunc, stmt);

  Map<Tensor, Tensor> replace;
  RenameBinds(data->binds_0, data->config, data->args, data->arg_list_0, replace);
  stmt = NEXT_PASS(RenameRealize, stmt, data->binds_0, replace);

  Array<NodeRef> arg_list_tmp;
  Map<Tensor, Buffer> binds_tmp;
  GetFlattenedBinds(data->args, data->binds_0, data->config, arg_list_tmp, binds_tmp, false);
  Stmt stmt_tmp = NEXT_PASS(ElementwiseFlatten, stmt, data->binds_0, binds_tmp);
  if (stmt_tmp.get() != stmt.get()) {
    stmt = stmt_tmp;
    data->arg_list_0 = arg_list_tmp;
    data->binds_0 = binds_tmp;
  }
  if (AttrExists(data->sch, "fuse_axis_extern")) {
    stmt = NEXT_PASS(FuseAxisExternOp, stmt, data->sch);
  }
  PassMgr::SetArgs(data->arg_list_0);
  stmt = NEXT_PASS(AddAttrForLayoutOp, stmt, data->sch, false);
  stmt = NEXT_PASS(AddAttrForConvolutionsOp, stmt);
  stmt = NEXT_PASS(RewriteTensorIndex, stmt);
  return {stmt, false};
}

StageResult LLVMLowerStageTuning(Stmt &stmt, LowerData &data) { return LowerStageTuning(stmt, data); }

StageResult LLVMLowerPoly(Stmt &stmt, LowerData &data) { return LowerPoly(stmt, data); }

StageResult LLVMLowerBeforeFlattern(Stmt &stmt, LowerData &data) {
  if (data->polyhedral) {
    stmt = NEXT_PASS(LowerWith, stmt);
    if (!g_csr.empty()) {
      stmt = NEXT_PASS(RestoreCsrLoop, stmt, data->binds_0, false);
    }
    stmt = NEXT_PASS(RealizeCompress, stmt);
    stmt = NEXT_PASS(AdjustParallelLoop, stmt);
    stmt = NEXT_PASS(ReductionFactor, stmt, data->binds_0);
  }
  return {stmt, false};
}

StageResult LLVMLowerFlattern(Stmt &stmt, LowerData &data) { return LowerFlattern(stmt, data); }

StageResult LLVMBeforeLowerFunc(Stmt &stmt, LowerData &data) {
  stmt = NEXT_PASS_IF(!data->simple_mode, LoopPartition, stmt, data->config->partition_const_loop);
  stmt = NEXT_PASS_IF(data->config->disable_vectorize, SkipVectorize, stmt);
  stmt = NEXT_PASS_IF(!data->config->disable_vectorize, VectorizeLoop, stmt);
  stmt = NEXT_PASS(UnrollLoop, stmt, data->config->auto_unroll_max_step, data->config->auto_unroll_max_depth,
                   data->config->auto_unroll_max_extent, data->config->unroll_explicit);
  return {stmt, false};
}

StageResult LLVMLowerDone(Stmt &stmt, LowerData &data) { return LowerDone(stmt, data); }

namespace lower {
// The order of following register will affect the stage order, please make sure it is right.
REG_STAGE_LOWER("llvm", StageType::Begin, "BEGIN", LLVMLowerBegin);
REG_STAGE_LOWER("llvm", StageType::Tuning, "TUNING", LLVMLowerStageTuning);
REG_STAGE_LOWER("llvm", StageType::Poly, "POLY", LLVMLowerPoly);
REG_STAGE_LOWER("llvm", StageType::BeforeFlattern, "BEFORE_FLATTERN", LLVMLowerBeforeFlattern);
REG_STAGE_LOWER("llvm", StageType::Flattern, "FLATTERN", LLVMLowerFlattern);
REG_STAGE_LOWER("llvm", StageType::BeforeLowerFunc, "BeforeLowerFunc", LLVMBeforeLowerFunc);
REG_STAGE_LOWER("llvm", StageType::End, "END", LLVMLowerDone);

NodeRef LLVMLowerImpl(const LowerData &data, bool get_stmt) { return StageLower(data).RunTo().Node(); }
REG_IMPL_LOWER("llvm", LLVMLowerImpl);
}  // namespace lower
}  // namespace akg
