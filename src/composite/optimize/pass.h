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
#ifndef COMPOSITE_PASS_H_
#define COMPOSITE_PASS_H_

#include "composite/utils/util.h"

namespace akg {

// insert broadcast
Stmt BroadcastInserter(const Stmt &s, BuildInfo *info);

// normalize axis attr
Stmt AxisAttrNormalize(const Stmt &s, BuildInfo *info);

// align memory clear size for ascend
Stmt CleanZeroAligner(const Stmt &s, BuildInfo *info);

// insert cast for equal(int32) in ascend
Stmt TypeCastInserter(const Stmt &s, BuildInfo *info);

// intrin rewrite
Stmt IntrinRewriter(const Stmt &s, BuildInfo *info);

// elim reshape forward
Stmt ElimReshapeForward(const Stmt &s, BuildInfo *info);

// elim reshape backward
Stmt ElimReshapeBackward(const Stmt &s, BuildInfo *info);

// fold dimension for multi-dim shape
Stmt FoldDimension(const Stmt &s, BuildInfo *info);

// reshape optimize
Stmt ReshapeTensor(const Stmt &s, BuildInfo *info);

// inplace_assign
Stmt InplaceAssignOpt(const Stmt &s, BuildInfo *info);

// delete cast for MatMul fusion
Stmt DeleteCast(const Stmt &s, BuildInfo *info);

// ops combine
Stmt OpsCombine(const Stmt &s, BuildInfo *info);

// peel dimension on given axes
Stmt PeelDimension(const Stmt &s, BuildInfo *info);

// rename MatMul to BatchMatMul
Stmt RenameMatmul(const Stmt &s, BuildInfo *info);

// rewrite the TransData op
Stmt TransDataRewriter(const Stmt &s, BuildInfo *info);

// expand complex op
Stmt ComplexExpander(const Stmt &s, BuildInfo *info);

// add attrs for op
Stmt AddAttrsForOp(const Stmt &s, BuildInfo *info);

// add broadct for ssa problem
Stmt BroadcastForSSA(const Stmt &s, BuildInfo *info);

}  // namespace akg
#endif  // COMPOSITE_PASS_H_
