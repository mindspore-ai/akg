/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Operation.h"

#include "akg/Dialect/Linalg/IR/LinalgExtOps.h"
#include "akg/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "akg/Dialect/Linalg/Transforms/Bufferize.h"

namespace mlir {
#define GEN_PASS_DEF_LINALGEXTBUFFERIZE
#include "akg/Dialect/Linalg/Passes.h.inc"
}  // namespace mlir

using namespace mlir;
using namespace bufferization;

namespace {
/// Converts Linalg operations that work on tensor-type operands or results to
/// work on buffers.
struct LinalgExtBufferizePass : public impl::LinalgExtBufferizeBase<LinalgExtBufferizePass> {
  void runOnOperation() override {
    BufferizationOptions options = getPartialBufferizationOptions();
    options.bufferAlignment = 0;
    options.opFilter.allowDialect<linalg::LinalgDialect, linalgExt::LinalgExtDialect>();

    if (failed(bufferizeOp(getOperation(), options))) {
      signalPassFailure();
    }
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect, memref::MemRefDialect, tensor::TensorDialect,
                    linalg::LinalgDialect, linalgExt::LinalgExtDialect>();

    linalg::registerBufferizableOpInterfaceExternalModels(registry);
    linalgExt::registerBufferizableOpInterfaceExternalModels(registry);
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createLinalgExtBufferizePass() {
  return std::make_unique<LinalgExtBufferizePass>();
}
