/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#ifndef MFUSION_DIALECT_MUSE_TRANSFORMS_PASSES_H
#define MFUSION_DIALECT_MUSE_TRANSFORMS_PASSES_H

#include "mfusion/Dialect/Muse/Transforms/Decompose/Decompose.h"
#include "mfusion/Dialect/Muse/Transforms/Cluster/DVMCluster.h"
#include "mfusion/Dialect/Muse/Transforms/Recompose/Recompose.h"
#include "mfusion/Dialect/Muse/Transforms/Fusion/FuseAddRmsNorm.h"
#include "mfusion/Dialect/Muse/Transforms/Fusion/FuseMatMul.h"
#include "mfusion/Dialect/Muse/Transforms/Fusion/GeluFusion.h"

namespace mlir {

/// Generate the code for registering transforms passes.
#ifndef GEN_PASS_REGISTRATION
#define GEN_PASS_REGISTRATION
#include "mfusion/Dialect/Muse/Transforms/Passes.h.inc"
#endif
}  // namespace mlir

#endif  // MFUSION_DIALECT_MUSE_TRANSFORMS_PASSES_H
