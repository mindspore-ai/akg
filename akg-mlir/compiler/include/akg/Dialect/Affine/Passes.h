/**
 * Copyright 2023-2025 Huawei Technologies Co., Ltd
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

#ifndef COMPILER_INCLUDE_AKG_DIALECT_AFFINE_PASSES_H_
#define COMPILER_INCLUDE_AKG_DIALECT_AFFINE_PASSES_H_

#include "akg/Dialect/Affine/Analysis/AutoTiling.h"
#include "akg/Dialect/Affine/Analysis/Axis.h"
#include "akg/Dialect/Affine/Analysis/Config.h"
#include "akg/Dialect/Affine/Analysis/Model.h"
#include "akg/Dialect/Affine/Transforms/AKGLoopParallelize.h"
#include "akg/Dialect/Affine/Transforms/AKGLoopTiling.h"
#include "akg/Dialect/Affine/Transforms/AKGLoopUnroll.h"
#include "akg/Dialect/Affine/Transforms/AffineDataCopyGeneration.h"
#include "akg/Dialect/Affine/Transforms/AffineHandleBoundaryIfExtract.h"
#include "akg/Dialect/Affine/Transforms/AffineHandleBoundaryIfRestore.h"
#include "akg/Dialect/Affine/Transforms/AffineIteratorConversion.h"
#include "akg/Dialect/Affine/Transforms/AffineLoopReorder.h"
#include "akg/Dialect/Affine/Transforms/AffineReductionAnnotation.h"
#include "akg/Dialect/Affine/Transforms/AKGLoopFusion.h"
#include "akg/Dialect/Affine/Transforms/AffineMemoryPromotion.h"
#include "akg/Dialect/Affine/Transforms/AffineTailBlockTiling.h"
#include "akg/Dialect/Affine/Transforms/ExtractIfOp.h"
#include "akg/Dialect/Affine/Transforms/FixDynamicIndexing.h"
#include "akg/Dialect/Affine/Transforms/ForceConvertAffineForToAffineParallel.h"
#include "akg/Dialect/Affine/Transforms/GenerateSingleAffineParallel.h"
#include "akg/Dialect/Affine/Transforms/MergeFusionOp.h"
#include "akg/Dialect/Affine/Transforms/RemoveRedundantLoops.h"
#include "akg/Dialect/Affine/Transforms/ReplaceUnknownDimsToOutputDim.h"
#include "akg/Dialect/Affine/Transforms/SimplifyShape.h"
#include "akg/Dialect/Affine/Transforms/UnifyShape.h"
#include "akg/Dialect/Affine/Transforms/WorkaroundFixReduceInitialization.h"

namespace mlir {

/// Generate the code for registering transforms passes.
#ifndef GEN_PASS_REGISTRATION
#define GEN_PASS_REGISTRATION
#include "akg/Dialect/Affine/Passes.h.inc"
#endif
}  // namespace mlir

#endif  // COMPILER_INCLUDE_AKG_DIALECT_AFFINE_PASSES_H_
