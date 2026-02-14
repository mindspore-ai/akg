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

#ifndef MFUSION_DIALECT_MFUSE_TRANSFORMS_FUSION_FUSION_PASSES_H
#define MFUSION_DIALECT_MFUSE_TRANSFORMS_FUSION_FUSION_PASSES_H

#include "mfusion/Dialect/Mfuse/Transforms/Fusion/MfuseFusion.h"
#include "mfusion/Dialect/Mfuse/Transforms/Fusion/Passes/FuseAddRmsNorm.h"
#include "mfusion/Dialect/Mfuse/Transforms/Fusion/Passes/FuseGelu.h"
#include "mfusion/Dialect/Mfuse/Transforms/Fusion/Passes/FuseSwiGlu.h"
#include "mfusion/Dialect/Mfuse/Transforms/Fusion/Passes/MatMul/FuseMatMulCast.h"
#include "mfusion/Dialect/Mfuse/Transforms/Fusion/Passes/MatMul/FuseMatmulReshapeBiasAdd.h"
#include "mfusion/Dialect/Mfuse/Transforms/Fusion/Passes/MatMul/FuseMatmulTransposeWeight.h"
#include "mfusion/Dialect/Mfuse/Transforms/Fusion/Passes/MatMul/FuseMatmulUnsqueezeSqueeze.h"
#include "mfusion/Dialect/Mfuse/Transforms/Fusion/Passes/MatMul/FuseBatchMatMul.h"
#include "mfusion/Dialect/Mfuse/Transforms/Fusion/Passes/MatMul/FuseBatchMatMulToMul.h"
#include "mfusion/Dialect/Mfuse/Transforms/Fusion/Passes/MatMul/FuseMatMulReshape.h"

#endif  // MFUSION_DIALECT_MFUSE_TRANSFORMS_FUSION_FUSION_PASSES_H
