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
#include "akg/Transforms/AKGOperatorIdentify.h"

#include <iostream>
#include "akg/Utils/AnalysisCommon.hpp"
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Builders.h"

namespace mlir {
#define GEN_PASS_DECL_AKGOPERATORIDENTIFY
#define GEN_PASS_DEF_AKGOPERATORIDENTIFY
#include "akg/Transforms/Passes.h.inc"
}  // namespace mlir

using namespace mlir;

namespace {
struct AKGOperatorIdentify : public impl::AKGOperatorIdentifyBase<AKGOperatorIdentify> {
  AKGOperatorIdentify() {}
  void runOnOperation() override;
};
}  // namespace

static bool elementwiseMatch(Operation *operation) {
  assert(operation->getNumResults() == 1 && "All TOSA elementwise ops should only return a single result.");

  auto resultTy = operation->getResult(0).getType().dyn_cast<ShapedType>();
  // Input indexing maps may be broadcasted.
  for (Value operand : operation->getOperands()) {
    ShapedType type = operand.getType().cast<ShapedType>();
    if (type.getShape() != resultTy.getShape()) {
      return false;
    }
  }
  return true;
}

static OperatorTemplate getOperatorTemplate(Operation *op) {
  OperatorTemplate opTemplate = OperatorTemplate::Default;
  if (TosaOperatorType::isTosaElementwiseOp(op) || MindOperatorType::isMindElementwiseOp(op)) {
    if (elementwiseMatch(op)) {
      opTemplate = OperatorTemplate::Elementwise;
    } else {
      opTemplate = OperatorTemplate::Broadcast;
    }
  } else if (isa<tosa::TileOp>(op)) {
    opTemplate = OperatorTemplate::Broadcast;
  } else if (isa<tosa::TransposeOp>(op)) {
    opTemplate = OperatorTemplate::Transpose;
  } else if (isa<tosa::ReshapeOp>(op)) {
    opTemplate = OperatorTemplate::Reshape;
  } else if (TosaOperatorType::isTosaReduceOp(op) || MindOperatorType::isMindReduceOp(op)) {
    opTemplate = OperatorTemplate::Reduce;
  } else if (isa<tosa::MatMulOp>(op)) {
    opTemplate = OperatorTemplate::Matmul;
  } else if (isa<tosa::Conv2DOp, tosa::Conv3DOp, tosa::DepthwiseConv2DOp>(op)) {
    opTemplate = OperatorTemplate::Conv;
  }
  return opTemplate;
}

void AKGOperatorIdentify::runOnOperation() {
  FunctionOpInterface funcOp = getOperation();
  OperatorTemplate curOpTemplate = OperatorTemplate::Default;
  funcOp.walk([&](Operation *op) {
    OperatorTemplate opTemplate = getOperatorTemplate(op);
    if (opTemplate > curOpTemplate) {
      curOpTemplate = opTemplate;
    }
  });

  auto iter = operatorTemplateMap.find((int)curOpTemplate);
  if (iter == operatorTemplateMap.end()) {
    return;
  }

  OpBuilder builder(funcOp.getContext());
  Attribute opType = builder.getStringAttr(iter->second);
  funcOp->setAttr(kOperatorTypeStr, opType);
}

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createAKGOperatorIdentifyPass() {
  return std::make_unique<AKGOperatorIdentify>();
}