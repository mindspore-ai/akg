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
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class Location;
class Operation;

namespace mfuse {

bool isDvmKernelGenerator(llvm::StringRef kernelGenerator);

// Returns true for the copied DVM subgraph function that is converted to Torch/FX
// payload after the original outlined DVM function has been serialized.
bool isInsideDvmCopiedSubgraph(Operation *op);

FailureOr<Value> buildSwapLastTwoDimsPermute(Location loc, Value v, ConversionPatternRewriter &rewriter);

}  // namespace mfuse
}  // namespace mlir

#endif  // MFUSION_CONVERSION_MFUSE_TO_TORCH_MFUSE_TO_TORCH_UTILS_H
