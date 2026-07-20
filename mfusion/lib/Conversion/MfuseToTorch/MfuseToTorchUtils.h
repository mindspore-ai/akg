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

#ifndef MFUSION_CONVERSION_MFUSE_TO_TORCH_MFUSE_TO_TORCH_UTILS_H
#define MFUSION_CONVERSION_MFUSE_TO_TORCH_MFUSE_TO_TORCH_UTILS_H

#include "llvm/ADT/StringRef.h"
#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class Location;
class Operation;
class Type;
class TypeConverter;

namespace mfuse {

bool isDvmKernelGenerator(llvm::StringRef kernelGenerator);

// Returns true for the copied DVM subgraph function that is converted to Torch/FX
// payload after the original outlined DVM function has been serialized.
bool isInsideDvmCopiedSubgraph(Operation *op);

FailureOr<Value> buildSwapLastTwoDimsPermute(Location loc, Value v, ConversionPatternRewriter &rewriter);

/// True when \p inputType can legally broadcast as aten.addmm's `input` onto a 2D
/// matmul result (1D bias / [1,N] / same-shape residual), and element types match.
/// Dynamic shapes → false.
bool isLegalAddmmInputShape(Type mmOutType, Type inputType);

/// Infer static 2D matmul product type from operands and transpose flags.
FailureOr<RankedTensorType> infer2DMatmulOutType(Type selfTy, Type otherTy, bool trans1, bool trans2);

/// Build torch.aten.addmm(input, mat1, mat2, beta=1, alpha=1), materializing
/// trans_x1/trans_x2 as permutes on mat1/mat2 when needed.
/// Requires mat1/mat2/input to already be torch ValueTensorType (e.g. adaptor
/// operands or successfully remapped values); otherwise returns failure.
FailureOr<Value> createAtenAddmm(Location loc, Type resultType, Value mat1, Value mat2, Value input, bool trans1,
                                 bool trans2, ConversionPatternRewriter &rewriter);

/// Shared match state for folding mfuse.aclnn.mm + mfuse.add → aten.addmm.
struct AclnnMmAddFoldMatch {
  AclnnMmOp mmOp;
  AddOp addOp;
  Value input;  ///< Bias / residual (mfuse value, not necessarily remapped).
};

bool matchAclnnMmAddFoldFromAdd(AddOp addOp, AclnnMmAddFoldMatch &out);
bool matchAclnnMmAddFoldFromMm(AclnnMmOp mmOp, AclnnMmAddFoldMatch &out);

/// Shape + dtype legality for a matched mm+add fold (ignores torch remapping).
/// Takes a non-const match because ODS getters (getOut / getResult) are non-const.
bool isAclnnMmAddFoldLegal(AclnnMmAddFoldMatch &match);

/// Emit aten.addmm and replace add / erase mm. \p mat1/\p mat2/\p remappedInput must be
/// Torch ValueTensorType. Returns failure if type conversion or createAtenAddmm fails
/// (IR unchanged aside from rewriter rollback).
LogicalResult rewriteAclnnMmAddToAtenAddmm(AclnnMmAddFoldMatch &match, const TypeConverter &converter, Value mat1,
                                           Value mat2, Value remappedInput, ConversionPatternRewriter &rewriter);

}  // namespace mfuse
}  // namespace mlir

#endif  // MFUSION_CONVERSION_MFUSE_TO_TORCH_MFUSE_TO_TORCH_UTILS_H
