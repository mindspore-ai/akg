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

namespace {
using mlir::DialectRegistry;
using mlir::failed;
using mlir::OperationPass;
/// Converts Linalg operations that work on tensor-type operands or results to
/// work on buffers.
struct LinalgExtBufferizePass : public mlir::impl::LinalgExtBufferizeBase<LinalgExtBufferizePass> {
  void runOnOperation() override {
    mlir::bufferization::BufferizationOptions options = mlir::bufferization::getPartialBufferizationOptions();
    options.bufferAlignment = 0;
    options.opFilter.allowDialect<mlir::linalg::LinalgDialect, mlir::linalgExt::LinalgExtDialect>();

    if (failed(mlir::bufferization::bufferizeOp(getOperation(), options))) {
      signalPassFailure();
    }
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::bufferization::BufferizationDialect, mlir::memref::MemRefDialect, mlir::tensor::TensorDialect,
                    mlir::linalg::LinalgDialect, mlir::linalgExt::LinalgExtDialect>();

    mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::linalgExt::registerBufferizableOpInterfaceExternalModels(registry);
  }
};
}  // namespace

namespace mlir {
std::unique_ptr<OperationPass<func::FuncOp>> createLinalgExtBufferizePass() {
  return std::make_unique<LinalgExtBufferizePass>();
}
}  // namespace mlir
