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

#ifndef MFUSION_CONVERSION_MFUSETOTORCH_UTILS_H
#define MFUSION_CONVERSION_MFUSETOTORCH_UTILS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"

namespace mlir {

namespace TorchD = mlir::torch::Torch;

/// Materializes constant value as a Torch scalar
/// Looks through UnrealizedConversionCastOp; if the value is a rank-0 float/int constant, creates
/// ConstantFloatOp/ConstantIntOp; otherwise returns the value as-is for the Torch op.
Value materializeConstValueToTorchScalar(Operation *op, Value value, ConversionPatternRewriter &rewriter);

}  // namespace mlir

#endif  // MFUSION_CONVERSION_MFUSETOTORCH_UTILS_H
