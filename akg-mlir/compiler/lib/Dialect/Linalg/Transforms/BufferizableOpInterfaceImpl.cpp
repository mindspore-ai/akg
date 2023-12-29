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

#include "akg/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "akg/Dialect/Linalg/IR/LinalgExtOps.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::linalgExt;
using namespace mlir::bufferization;

namespace {
// Bufferization of linalg.generic.
template <typename OpTy>
struct LinalgExtOpBufferizationTait
    : public DstBufferizableOpInterfaceExternalModel<LinalgExtOpBufferizationTait<OpTy>, OpTy> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand, const AnalysisState &state) const {
    // Operand is read if it is used in the computation.
    auto genericOp = cast<linalg::LinalgOp>(op);
    return genericOp.payloadUsesValueFromOperand(&opOperand);
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand, const AnalysisState &state) const {
    // Operand is written to if it has an aliasing OpResult.
    auto bufferizableOp = cast<BufferizableOpInterface>(op);
    return !bufferizableOp.getAliasingOpResult(opOperand, state).empty();
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter, const BufferizationOptions &options) const {
    return bufferizeDestinationStyleOpInterface(rewriter, cast<DestinationStyleOpInterface>(op), options);
  }
};

// registers the `BufferizableOpInterface` with each of LinalgExt Ops.
template <typename... Ops>
struct LinalgExtOpInterfaceHelper {
  static void registerOpInterface(MLIRContext *ctx) {
    (Ops::template attachInterface<LinalgExtOpBufferizationTait<Ops>>(*ctx), ...);
  }
};
}  // namespace

void mlir::linalgExt::registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, linalgExt::LinalgExtDialect *dialect) {
    // Register all LinalgExt ops.
    LinalgExtOpInterfaceHelper<
#define GET_OP_LIST
#include "akg/Dialect/Linalg/IR/LinalgExtOps.cpp.inc"
      >::registerOpInterface(ctx);
  });
}

