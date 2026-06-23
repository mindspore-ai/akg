/**
 * Copyright 2023-2026 Huawei Technologies Co., Ltd
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

#ifndef AKG_ANALYSIS_LOOPTILING_H_
#define AKG_ANALYSIS_LOOPTILING_H_

#include <vector>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "akg/Utils/SmallVectorSize.h"

using llvm::DenseMap;
using llvm::SmallVector;

namespace mlir {
namespace autotiling {

struct TilingMetadata {
  std::vector<SmallVector<unsigned, kSmallVectorSizeSix>> bandTileSizes;
  std::vector<SmallVector<int, kSmallVectorSizeSix>> bandConstraintMaxs;

  bool empty() const { return bandTileSizes.empty() && bandConstraintMaxs.empty(); }
  void clear() {
    bandTileSizes.clear();
    bandConstraintMaxs.clear();
  }
};

// Create tiling functions for a kernel
// originalKernel: the input IR function
// builder: OpBuilder for creating operations
// out: output map to store created tiling functions (key: tiling strategy index, value: FuncOp)
// isStaticShape: if true, create empty tiling function (equivalent to bands.empty())
LogicalResult createTilingFunctions(func::FuncOp originalKernel, OpBuilder &builder,
                                    DenseMap<int64_t, func::FuncOp> &out, bool isStaticShape = false);
LogicalResult createTilingFunctions(func::FuncOp originalKernel, OpBuilder &builder,
                                    DenseMap<int64_t, func::FuncOp> &out, bool isStaticShape, TilingMetadata *metadata);

// Get the selected tiling strategy index (default: 0)
int64_t getSelectedTilingStrategyIndex();

// Apply tiling to a kernel using tile sizes from a memref
// originalKernel: the input IR function to be tiled (tileSizesMemref is the last argument)
// builder: OpBuilder for creating operations
// isStaticShape: if true, calculate tile sizes using auto-tiling and create tileSizeValues directly
LogicalResult applyTilingFromTilingFunc(func::FuncOp originalKernel, OpBuilder &builder, bool isStaticShape = false);
LogicalResult applyTilingFromTilingFunc(func::FuncOp originalKernel, OpBuilder &builder, bool isStaticShape,
                                        const TilingMetadata *metadata);

}  // namespace autotiling
}  // namespace mlir

#endif  // AKG_ANALYSIS_LOOPTILING_H_
