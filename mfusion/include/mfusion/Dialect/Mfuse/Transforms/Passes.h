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

#ifndef MFUSION_DIALECT_MFUSE_TRANSFORMS_PASSES_H
#define MFUSION_DIALECT_MFUSE_TRANSFORMS_PASSES_H

#include "mfusion/Dialect/Mfuse/MfuseDialect.h"
#include "mfusion/Dialect/Mfuse/Transforms/Decompose/Decompose.h"
#include "mfusion/Dialect/Mfuse/Transforms/Cluster/DVMCluster.h"
#include "mfusion/Dialect/Mfuse/Transforms/Cluster/AKGCluster.h"
#include "mfusion/Dialect/Mfuse/Transforms/Recompose/Recompose.h"
#include "mfusion/Dialect/Mfuse/Transforms/Fusion/FusionPasses.h"
#include "mfusion/Dialect/Mfuse/Transforms/Outlining/OutlineMfuseFusedSubgraphs.h"
#include "mfusion/Dialect/Mfuse/Transforms/Outlining/CopyFusedSubgraphs.h"
#include "mfusion/Dialect/Mfuse/Transforms/ConvertDvmSubgraphToMfuseDvmCall.h"
#include "mfusion/Dialect/Mfuse/Transforms/Split/Split.h"

namespace mlir {

/// Generate the code for registering transforms passes.
#ifndef GEN_PASS_REGISTRATION
#define GEN_PASS_REGISTRATION
#include "mfusion/Dialect/Mfuse/Transforms/Passes.h.inc"
#endif
}  // namespace mlir

#endif  // MFUSION_DIALECT_MFUSE_TRANSFORMS_PASSES_H
