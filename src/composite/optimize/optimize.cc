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
#include "composite/optimize/optimize.h"
#include <memory>
#include "composite/optimize/rename_matmul.h"
#include "composite/optimize/reshape_tensor.h"
#include "composite/optimize/elim_reshape.h"
#include "composite/optimize/inplace_assign_mutator.h"
#include "composite/optimize/broadcast_inserter.h"
#include "composite/optimize/axis_attr_normalize.h"
#include "composite/optimize/fold_dimension.h"
#include "composite/optimize/typecast_inserter.h"
#include "composite/optimize/ops_combine.h"
#include "composite/optimize/intrin_rewriter.h"
#include "composite/optimize/peel_dimension.h"
#include "composite/optimize/complex_expander.h"
#include "composite/optimize/delete_cast.h"
#include "composite/optimize/transdata_rewriter.h"
#include "composite/optimize/clean_zero_align.h"

namespace akg {
Stmt Optimize(Stmt &s, BuildInfo &info) {
  auto pm = CompositeOptPassMgr(info);
  // insert broadcast
  pm.RegisterPass(std::make_shared<BroadcastInserter>());
  // reshape optimize
  if (info.opt.target == "aicore") {
    pm.RegisterPass(std::make_shared<ReshapeTensor>());
  }
  // rewrite the TransData op
  if (info.opt.target == "aicore") {
    pm.RegisterPass(std::make_shared<TransDataRewriter>());
  }
  // ops combine
  if (info.opt.target == "aicore") {
    pm.RegisterPass(std::make_shared<OpsCombine>(pm.info_));
  }
  // normalize axis attr
  pm.RegisterPass(std::make_shared<AxisAttrNormalize>());
  // elim reshape backward
  pm.RegisterPass(std::make_shared<ElimReshapeBackward>(pm.info_));
  // elim reshape forward
  pm.RegisterPass(std::make_shared<ElimReshapeForward>(pm.info_));
  // fold dimension for multi-dim shape
  if (info.opt.fold_dim) {
    pm.RegisterPass(std::make_shared<FoldDimension>(pm.info_));
  }
  // inplace_assign
  pm.RegisterPass(std::make_shared<InplaceAssignOpt>(pm.info_));
  // insert cast for equal(int32) in ascend
  if (info.opt.target == "aicore") {
    pm.RegisterPass(std::make_shared<TypeCastInserter>());
  }
  // rename MatMul to BatchMatMul
  pm.RegisterPass(std::make_shared<RenameMatmul>());
  // delete cast for MatMul fusion
  if (info.opt.target == "cuda") {
    pm.RegisterPass(std::make_shared<DeleteCast>());
  }
  if (info.opt.target == "aicore") {
    // intrin rewrite
    pm.RegisterPass(std::make_shared<IntrinRewriter>());
    // peel dimension on given axes
    pm.RegisterPass(std::make_shared<PeelDimension>(pm.info_));
  }
  // expand complex op
  pm.RegisterPass(std::make_shared<ComplexExpander>());
  if (info.opt.target == "aicore") {
    pm.RegisterPass(std::make_shared<CleanZeroAligner>(pm.info_));
  }
  s = pm.Run(s);
  return s;
}

}  // namespace akg
