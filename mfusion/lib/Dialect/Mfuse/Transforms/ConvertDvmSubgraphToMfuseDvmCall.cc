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

#include "mfusion/Dialect/Mfuse/Transforms/ConvertDvmSubgraphToMfuseDvmCall.h"
#include "mfusion/Dialect/Mfuse/Transforms/Outlining/FusionAttributes.h"

#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mfusion/Dialect/Dvm/IR/Dvm.h"
#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"
#include "mfusion/Dialect/Mfuse/Transforms/Passes.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTDVMSUBGRAPHTOMFUSEDVMCALL
#include "mfusion/Dialect/Mfuse/Transforms/Passes.h.inc"

namespace mfuse {
namespace {

static bool isDvmOutlinedFunc(func::FuncOp func) {
  if (!func || !func->hasAttr(mfusion_attrs::kOutlined)) {
    return false;
  }
  auto fusionTypeAttr = func->getAttrOfType<StringAttr>(mfusion_attrs::kFusionType);
  return fusionTypeAttr && fusionTypeAttr.getValue() == "dvm";
}

static bool isDynamicType(Type type) {
  auto shapedType = type.dyn_cast<ShapedType>();
  return shapedType && shapedType.hasRank() && !shapedType.hasStaticShape();
}

static bool isDynamicSubgraph(func::FuncOp func) {
  if (!func) {
    return false;
  }

  auto funcType = func.getFunctionType();
  for (Type type : funcType.getInputs()) {
    if (isDynamicType(type)) {
      return true;
    }
  }
  for (Type type : funcType.getResults()) {
    if (isDynamicType(type)) {
      return true;
    }
  }

  auto walkResult = func.walk([&](Operation *op) {
    for (Type type : op->getOperandTypes()) {
      if (isDynamicType(type)) {
        return WalkResult::interrupt();
      }
    }
    for (Type type : op->getResultTypes()) {
      if (isDynamicType(type)) {
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return walkResult.wasInterrupted();
}

static std::string serializeDvmFuncToModuleString(func::FuncOp funcOp) {
  auto module = ModuleOp::create(funcOp.getLoc());
  OpBuilder builder(module.getBodyRegion());

  auto cloned = funcOp.clone();
  cloned.setName("entry");
  builder.insert(cloned);

  std::string result;
  llvm::raw_string_ostream os(result);
  module.print(os);
  os.flush();
  return result;
}

static StringAttr getCopiedSubgraphNameAttr(func::FuncOp funcOp) {
  auto attr = funcOp->getAttrOfType<StringAttr>(mfusion_attrs::kCopiedSubgraph);
  if (!attr) {
    funcOp->emitError("Missing required attribute '")
        << mfusion_attrs::kCopiedSubgraph << "' on outlined function. "
        << "Ensure copy-fused-subgraphs pass runs before this pass.";
    return nullptr;
  }
  return attr;
}

struct ConvertDvmSubgraphToMfuseDvmCallPass
    : public impl::ConvertDvmSubgraphToMfuseDvmCallBase<ConvertDvmSubgraphToMfuseDvmCallPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SymbolTable symbolTable(module);

    SmallVector<func::FuncOp> outlinedFuncs;
    for (auto func : module.getOps<func::FuncOp>()) {
      if (isDvmOutlinedFunc(func)) {
        outlinedFuncs.push_back(func);
      }
    }

    for (auto func : outlinedFuncs) {
      auto subgraphAttr = getCopiedSubgraphNameAttr(func);
      if (!subgraphAttr) {
        return signalPassFailure();
      }
      std::string subgraphMlir = serializeDvmFuncToModuleString(func);
      auto subgraphMlirAttr = StringAttr::get(module.getContext(), subgraphMlir);

      auto uses = symbolTable.getSymbolUses(func, module);
      if (uses) {
        for (auto use : *uses) {
          auto callOp = dyn_cast<func::CallOp>(use.getUser());
          if (!callOp) continue;
          OpBuilder builder(callOp);
          OperationState state(callOp.getLoc(), "mfuse.dvm_call");
          state.addOperands(callOp.getOperands());
          state.addTypes(callOp.getResultTypes());
          state.addAttribute("subgraph_mlir", subgraphMlirAttr);
          state.addAttribute("subgraph", subgraphAttr);
          state.addAttribute("is_dynamic", builder.getBoolAttr(isDynamicSubgraph(func)));
          auto *newOp = builder.create(state);
          callOp.replaceAllUsesWith(newOp->getResults());
          callOp.erase();
        }
      }

      func.erase();
    }
  }
};

}  // namespace
}  // namespace mfuse

std::unique_ptr<Pass> createConvertDvmSubgraphToMfuseDvmCallPass() {
  return std::make_unique<mfuse::ConvertDvmSubgraphToMfuseDvmCallPass>();
}

}  // namespace mlir
