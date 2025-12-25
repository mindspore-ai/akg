/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "akg/Dialect/NPUVector/IR/NPUVector.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/VectorInterfaces.h"

using namespace mlir;  // NOLINT(build/namespaces)
using namespace mlir::npuvector;  // NOLINT(build/namespaces)

//===----------------------------------------------------------------------===//
// TransferReadOp
//===----------------------------------------------------------------------===//

void TransferReadOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  if (llvm::isa<MemRefType>(getShapedType()))
    effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable(), SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// TransferWriteOp
//===----------------------------------------------------------------------===//

void TransferWriteOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  if (llvm::isa<MemRefType>(getShapedType()))
    effects.emplace_back(MemoryEffects::Write::get(), &getSourceMutable(), SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "akg/Dialect/NPUVector/IR/NPUVectorOps.cpp.inc"

// Include builder implementations
#define GET_OP_IMPL
// NOLINTNEXTLINE(build/include)
#include "akg/Dialect/NPUVector/IR/NPUVectorOps.cpp.inc"

