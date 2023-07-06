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
#include "composite/optimize/optimize.h"
#include <memory>
#include "composite/optimize/pass.h"

namespace akg {

#define ADD_PASS(pm, pass) (pm).Register(#pass, pass)

Stmt Optimize(Stmt &s, BuildInfo &info) {
  auto pm = TranslatePassMgr(&info);
  ADD_PASS(pm, BroadcastInserter);
  if (info.opt.target == "aicore") {
    ADD_PASS(pm, ReshapeTensor);
  }
  if (info.opt.target == "aicore") {
    ADD_PASS(pm, TransDataRewriter);
  }
  if (info.opt.target == "aicore") {
    ADD_PASS(pm, OpsCombine);
  }
  ADD_PASS(pm, AxisAttrNormalize);
  ADD_PASS(pm, ElimReshapeBackward);
  ADD_PASS(pm, ElimReshapeForward);
  if (info.opt.fold_dim) {
    ADD_PASS(pm, FoldDimension);
  }
  ADD_PASS(pm, InplaceAssignOpt);
  if (info.opt.target == "aicore") {
    ADD_PASS(pm, TypeCastInserter);
  }
  ADD_PASS(pm, RenameMatmul);
  if (info.opt.target == "cuda") {
    ADD_PASS(pm, DeleteCast);
  }
  if (info.opt.target == "aicore") {
    ADD_PASS(pm, PeelDimension);
  }
  ADD_PASS(pm, ComplexExpander);
  if (info.opt.target == "aicore") {
    ADD_PASS(pm, CleanZeroAligner);
  }
  ADD_PASS(pm, AddAttrsForOp);
  ADD_PASS(pm, BroadcastForSSA);
  if (info.opt.target == "cuda") {
    ADD_PASS(pm, LogicalOrToAdd);
  }
  return pm.Run(s);
}

Stmt OptimizeForTBE(const Stmt &s, BuildInfo &info) {
  auto pm = TranslatePassMgr(&info, "composite_tbe");
  ADD_PASS(pm, ReshapeTensor);
  ADD_PASS(pm, AxisAttrNormalize);
  ADD_PASS(pm, ElimReshapeBackward);
  ADD_PASS(pm, ElimReshapeForward);
  ADD_PASS(pm, FoldDimension);
  ADD_PASS(pm, TypeCastInserter);
  ADD_PASS(pm, ElimInplaceAssign);
  return pm.Run(s);
}
}  // namespace akg
