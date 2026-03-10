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

#ifndef MFUSION_DIALECT_MFUSE_TRANSFORMS_SPLIT_FUSEOPSPLITTER_H
#define MFUSION_DIALECT_MFUSE_TRANSFORMS_SPLIT_FUSEOPSPLITTER_H

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <string>
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"
#include "mfusion/Analysis/Split/Area.h"
#include "mfusion/Analysis/Split/SplitModel.h"
#include "mfusion/Dialect/Mfuse/Transforms/Split/SplitSchemer.h"

/// FuseOpSplitter splits fuse operations
class FuseOpSplitter {
 public:
  FuseOpSplitter() = default;
  virtual ~FuseOpSplitter() = default;

  /// Try to split a specific operation
  bool trySplit(mlir::mfuse::FusedOp op, const std::string &kernelGenerator = "DVM");

  /// Get the split scheme
  virtual mlir::mfuse::split::SplitSchemerPtr getSplitSchema(const std::string &kernelGenerator);
};

#endif  // MFUSION_DIALECT_MFUSE_TRANSFORMS_SPLIT_FUSEOPSPLITTER_H
