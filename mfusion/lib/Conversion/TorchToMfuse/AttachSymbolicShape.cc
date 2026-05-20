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

#include <algorithm>
#include <iterator>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"

#include "mfusion/Analysis/SymbolicShape/SymEngineAnalysis.h"
#include "mfusion/Conversion/Passes.h"
#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"
#include "mfusion/Dialect/Mfuse/IR/MfuseDialect.h"
#include "mfusion/Dialect/Mfuse/Support/SymbolAttrUtils.h"

namespace {
namespace TorchD = mlir::torch::Torch;

mlir::Attribute dropSymbolicShapeEncoding(mlir::RankedTensorType type) {
  auto encoding = type.getEncoding();
  if (!encoding) {
    return {};
  }
  if (mlir::mfuse::SymbolAttrUtils::isSymbolicShapeEncoding(encoding)) {
    return {};
  }

  auto dict = mlir::dyn_cast<mlir::DictionaryAttr>(encoding);
  if (!dict) {
    return encoding;
  }

  mlir::MLIRContext *ctx = type.getContext();
  auto symKey = mlir::StringAttr::get(ctx, mlir::mfuse::SymbolAttrUtils::kSymShapeKey);
  auto baseKey = mlir::StringAttr::get(ctx, mlir::mfuse::SymbolAttrUtils::kBaseEncodingKey);
  llvm::SmallVector<mlir::NamedAttribute> entries;
  entries.reserve(dict.getValue().size());
  std::copy_if(dict.getValue().begin(), dict.getValue().end(), std::back_inserter(entries),
               [symKey](const mlir::NamedAttribute &entry) { return entry.getName() != symKey; });

  if (entries.empty()) {
    return {};
  }
  if (entries.size() == 1 && entries.front().getName() == baseKey) {
    return entries.front().getValue();
  }
  return mlir::DictionaryAttr::get(ctx, entries);
}

// Set a builtin/Mfuse tensor value to an already canonicalized type, e.g.
// tensor<1x?xf32, #mfuse.symshape<["1", "6"]>> becomes tensor<1x6xf32>.
bool setRankedTensorValueType(mlir::Value value, mlir::RankedTensorType newType) {
  if (auto result = mlir::dyn_cast<mlir::OpResult>(value)) {
    if (result.getType() != newType) {
      result.setType(newType);
    }
    return true;
  }
  if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    if (arg.getType() != newType) {
      arg.setType(newType);
    }
    return true;
  }
  return false;
}

bool hasFuncCallers(mlir::func::FuncOp func) {
  auto module = func->getParentOfType<mlir::ModuleOp>();
  if (!module) {
    return false;
  }

  bool hasCaller = false;
  module.walk([&](mlir::func::CallOp callOp) {
    if (callOp.getCallee() == func.getSymName()) {
      hasCaller = true;
    }
  });
  return hasCaller;
}

unsigned countReturnOps(mlir::func::FuncOp func) {
  unsigned count = 0;
  func.walk([&](mlir::func::ReturnOp) { ++count; });
  return count;
}

bool canRefineFunctionSignature(mlir::func::FuncOp func) {
  // Conservative first version: only refine standalone single-return function
  // signatures. If a function has func.call users, every call site's
  // operand/result types must be updated together with the callee signature.
  // TODO(mfusion): Support called functions by propagating signature
  // refinements to all func.call users.
  return func && !func.isExternal() && !hasFuncCallers(func) && countReturnOps(func) == 1;
}

mlir::func::FuncOp getEntryFunctionForArgument(mlir::BlockArgument arg) {
  auto func = mlir::dyn_cast_or_null<mlir::func::FuncOp>(arg.getOwner()->getParentOp());
  if (!func || func.getBody().empty() || arg.getOwner() != &func.getBody().front()) {
    return {};
  }
  return func;
}

bool hasIneligibleReturnUser(mlir::Value value) {
  for (mlir::Operation *user : value.getUsers()) {
    auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(user);
    if (!returnOp) {
      continue;
    }
    auto func = returnOp->getParentOfType<mlir::func::FuncOp>();
    if (!canRefineFunctionSignature(func)) {
      return true;
    }
  }
  return false;
}

// Keep the entry block argument type and the func.func input type in sync when
// a bound function input is refined.
//
// Before:
//   func.func @main(%arg0: !torch.vtensor<[1,?],si64>) {
//     torch.bind_symbolic_shape %arg0, [], affine_map<() -> (1, 6)>
//   }
//
// After:
//   func.func @main(%arg0: !torch.vtensor<[1,6],si64>) {
//     ...
//   }
//
// If the printed function signature stays at !torch.vtensor<[1,?],...> while
// the argument uses !torch.vtensor<[1,6],...>, later mfuse-to-torch roundtrip can
// leave a type-changing unrealized_conversion_cast at the function boundary.
bool updateFunctionArgumentType(mlir::BlockArgument arg, mlir::Type newType) {
  auto func = getEntryFunctionForArgument(arg);
  if (!canRefineFunctionSignature(func)) {
    return false;
  }

  auto funcType = func.getFunctionType();
  llvm::SmallVector<mlir::Type> inputs(funcType.getInputs().begin(), funcType.getInputs().end());
  inputs[static_cast<size_t>(arg.getArgNumber())] = newType;
  func.setFunctionType(mlir::FunctionType::get(func.getContext(), inputs, funcType.getResults()));
  return true;
}

// Generic value type setter for Torch values. For eligible function entry
// arguments, this also updates the printed func.func signature.
bool setTorchValueType(mlir::Value value, mlir::Type newType) {
  if (auto result = mlir::dyn_cast<mlir::OpResult>(value)) {
    if (result.getType() != newType) {
      if (hasIneligibleReturnUser(result)) {
        return false;
      }
      result.setType(newType);
    }
    return true;
  }
  if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    if (arg.getType() != newType) {
      if (getEntryFunctionForArgument(arg) && !updateFunctionArgumentType(arg, newType)) {
        return false;
      }
      arg.setType(newType);
    }
    return true;
  }
  return false;
}

// Canonicalize a builtin RankedTensorType according to the symbolic shape exprs
// already resolved from torch.bind_symbolic_shape.
//
// Torch dialect can legally express a "pseudo dynamic" shape as:
//
//   !torch.vtensor<[1,1,?,37],i1>
//   torch.bind_symbolic_shape ..., affine_map<() -> (1, 1, 6, 37)>
//
// Mfuse IR is stricter: a dynamic dimension may not be paired with a constant
// symbolic expression. At the Torch-to-Mfuse boundary this must become:
//
//   tensor<1x1x6x37xi1>
//
// Real dynamic dimensions are preserved with an encoding:
//
//   affine_map<()[s0] -> (1, s0, 37)>
//     -> tensor<1x?x37xf32, #mfuse.symshape<["1", "s0", "37"]>>
mlir::FailureOr<mlir::RankedTensorType> canonicalizeTypeWithSymbolicExprs(
  mlir::Operation *diagOp, mlir::RankedTensorType ranked, llvm::ArrayRef<mlir::mfuse::SymbolAttrUtils::SymExpr> exprs,
  mlir::OpBuilder &builder, mfusion::SymEngineAnalysis &analysis) {
  if (ranked.getRank() != static_cast<int64_t>(exprs.size())) {
    diagOp->emitError("bind_symbolic_shape result rank does not match ranked tensor type rank");
    return mlir::failure();
  }

  llvm::SmallVector<int64_t> newShape(ranked.getShape().begin(), ranked.getShape().end());
  bool hasDynamicExpr = false;
  for (int64_t i = 0; i < ranked.getRank(); ++i) {
    auto maybeInt = analysis.tryExtractInt64(exprs[static_cast<size_t>(i)]);
    if (mlir::succeeded(maybeInt)) {
      if (!mlir::ShapedType::isDynamic(newShape[static_cast<size_t>(i)]) &&
          newShape[static_cast<size_t>(i)] != *maybeInt) {
        diagOp->emitError() << "static tensor dimension " << i << " is " << newShape[static_cast<size_t>(i)]
                            << " but bind_symbolic_shape expression is " << *maybeInt;
        return mlir::failure();
      }
      newShape[static_cast<size_t>(i)] = *maybeInt;
      continue;
    }

    if (!mlir::ShapedType::isDynamic(newShape[static_cast<size_t>(i)])) {
      diagOp->emitError() << "static tensor dimension " << i << " is " << newShape[static_cast<size_t>(i)]
                          << " but bind_symbolic_shape expression is non-constant: "
                          << exprs[static_cast<size_t>(i)]->__str__();
      return mlir::failure();
    }
    hasDynamicExpr = true;
  }

  auto baseEncoding = hasDynamicExpr ? ranked.getEncoding() : dropSymbolicShapeEncoding(ranked);
  auto baseType = mlir::RankedTensorType::get(newShape, ranked.getElementType(), baseEncoding);
  if (!hasDynamicExpr) {
    return baseType;
  }
  return mlir::mfuse::SymbolAttrUtils::withSymbolicAttr(
    baseType, mlir::mfuse::SymbolAttrUtils::createSymbolicShapeAttr(builder, exprs));
}

// Mirror the same canonicalization onto Torch ValueTensorType. Torch types do
// not carry mfuse.symshape encodings, so only constant bind expressions refine
// '?' to a static size. Non-constant symbols remain '?' in Torch IR.
//
// This is needed for roundtrip cleanup. If only the Mfuse cast result is
// refined, the final pipeline can produce:
//
//   !torch.vtensor<[1,1,?,37],i1>
//     -> tensor<1x1x6x37xi1>
//     -> !torch.vtensor<[1,1,6,37],i1>
//
// The source and target Torch tensor types differ, so
// reconcile-unrealized-casts cannot erase the cast chain. Refining the bound
// Torch SSA value to !torch.vtensor<[1,1,6,37],i1> keeps both sides consistent.
mlir::FailureOr<TorchD::ValueTensorType> canonicalizeTorchTypeWithSymbolicExprs(
  mlir::Operation *diagOp, TorchD::ValueTensorType type, llvm::ArrayRef<mlir::mfuse::SymbolAttrUtils::SymExpr> exprs,
  mfusion::SymEngineAnalysis &analysis) {
  if (!type.hasSizes()) {
    return mlir::failure();
  }
  auto sizes = type.getSizes();
  if (static_cast<int64_t>(sizes.size()) != static_cast<int64_t>(exprs.size())) {
    diagOp->emitError("bind_symbolic_shape result rank does not match torch tensor type rank");
    return mlir::failure();
  }

  llvm::SmallVector<int64_t> newSizes(sizes.begin(), sizes.end());
  for (auto [index, expr] : llvm::enumerate(exprs)) {
    auto maybeInt = analysis.tryExtractInt64(expr);
    int64_t currentSize = newSizes[index];
    if (mlir::succeeded(maybeInt)) {
      if (currentSize != TorchD::kUnknownSize && currentSize != *maybeInt) {
        diagOp->emitError() << "static tensor dimension " << index << " is " << currentSize
                            << " but bind_symbolic_shape expression is " << *maybeInt;
        return mlir::failure();
      }
      newSizes[index] = *maybeInt;
      continue;
    }

    if (currentSize != TorchD::kUnknownSize) {
      diagOp->emitError() << "static tensor dimension " << index << " is " << currentSize
                          << " but bind_symbolic_shape expression is non-constant: " << expr->__str__();
      return mlir::failure();
    }
  }

  auto newType = mlir::dyn_cast<TorchD::ValueTensorType>(
    type.getWithSizesAndDtype(llvm::ArrayRef<int64_t>(newSizes), type.getOptionalDtype()));
  if (!newType) {
    diagOp->emitError("failed to build canonical torch tensor type from bind_symbolic_shape");
    return mlir::failure();
  }
  return newType;
}

// Canonicalize the Torch value declared by this bind op when the bind
// expression resolves a '?' dimension to a constant:
//
//   before:
//     %arg15: !torch.vtensor<[1,1,?,37],i1>
//     torch.bind_symbolic_shape %arg15, [],
//       affine_map<() -> (1, 1, 6, 37)>
//
//   after:
//     %arg15: !torch.vtensor<[1,1,6,37],i1>
//
// True dynamic bindings remain dynamic:
//
//   %s1 = torch.symbolic_int "s1" {min_val = 2, max_val = ...}
//   torch.bind_symbolic_shape %arg3, [%s1],
//     affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],si64>
//
//   after:
//     %arg3: !torch.vtensor<[?],si64>
mlir::LogicalResult canonicalizeTorchValueWithSymbolicExprs(mlir::Operation *diagOp, mlir::Value value,
                                                            llvm::ArrayRef<mlir::mfuse::SymbolAttrUtils::SymExpr> exprs,
                                                            mfusion::SymEngineAnalysis &analysis) {
  auto tensorType = mlir::dyn_cast<TorchD::ValueTensorType>(value.getType());
  if (!tensorType || !tensorType.hasSizes()) {
    return mlir::success();
  }
  auto canonicalType = canonicalizeTorchTypeWithSymbolicExprs(diagOp, tensorType, exprs, analysis);
  if (mlir::failed(canonicalType)) {
    return mlir::failure();
  }
  (void)setTorchValueType(value, *canonicalType);
  return mlir::success();
}

// Returns true if newType only refines unknown Torch tensor dimensions in
// oldType to concrete sizes, with rank and dtype unchanged.
bool isRefinedTorchTensorType(mlir::Type oldType, mlir::Type newType) {
  auto oldTensorType = mlir::dyn_cast<TorchD::ValueTensorType>(oldType);
  auto newTensorType = mlir::dyn_cast<TorchD::ValueTensorType>(newType);
  if (!oldTensorType || !newTensorType || !oldTensorType.hasSizes() || !newTensorType.hasSizes()) {
    return false;
  }
  if (oldTensorType.getOptionalDtype() != newTensorType.getOptionalDtype()) {
    return false;
  }
  auto oldSizes = oldTensorType.getSizes();
  auto newSizes = newTensorType.getSizes();
  if (oldSizes.size() != newSizes.size()) {
    return false;
  }
  for (auto [oldSize, newSize] : llvm::zip(oldSizes, newSizes)) {
    if (oldSize != newSize && oldSize != TorchD::kUnknownSize) {
      return false;
    }
  }
  return true;
}

// Function result types are not tied to a distinct SSA value. If the single
// return operand was refined from a pseudo dynamic Torch tensor, refine the
// function result as well:
//
//   func.func @main(...) -> !torch.vtensor<[1,1,?,37],f32>
//   return %0 : !torch.vtensor<[1,1,6,37],f32>
//
// becomes:
//
//   func.func @main(...) -> !torch.vtensor<[1,1,6,37],f32>
//
// Only monotonic refinements from '?' to a concrete size are accepted here.
void refineFunctionResultTypes(mlir::func::FuncOp func) {
  if (func.isExternal() || hasFuncCallers(func)) {
    return;
  }
  llvm::SmallVector<mlir::func::ReturnOp> returnOps;
  func.walk([&](mlir::func::ReturnOp returnOp) { returnOps.push_back(returnOp); });
  if (returnOps.size() != 1) {
    // A function can have multiple return terminators through control flow:
    //
    //   cf.cond_br %cond, ^then, ^else
    // ^then:
    //   return %a : !torch.vtensor<[1,1,6,37],f32>
    // ^else:
    //   return %b : !torch.vtensor<[1,1,?,37],f32>
    //
    // Refining the function result from only one return could make the other
    // return invalid. Keep the signature unchanged unless there is exactly one
    // return terminator.
    // TODO(mfusion): Support this after checking that all return operands
    // allow the same monotonic result type refinement.
    return;
  }

  auto funcType = func.getFunctionType();
  auto returnOp = returnOps.front();
  if (returnOp.getNumOperands() != funcType.getNumResults()) {
    return;
  }

  llvm::SmallVector<mlir::Type> results(funcType.getResults().begin(), funcType.getResults().end());
  bool updated = false;
  for (auto [index, operand] : llvm::enumerate(returnOp.getOperands())) {
    if (isRefinedTorchTensorType(results[index], operand.getType())) {
      results[index] = operand.getType();
      updated = true;
    }
  }
  if (updated) {
    func.setFunctionType(mlir::FunctionType::get(func.getContext(), funcType.getInputs(), results));
  }
}

mlir::LogicalResult attachToCastResults(mlir::Operation *diagOp, mlir::Operation *castOp,
                                        llvm::ArrayRef<mlir::mfuse::SymbolAttrUtils::SymExpr> exprs,
                                        mlir::OpBuilder &builder, mfusion::SymEngineAnalysis &analysis) {
  for (mlir::Value result : castOp->getResults()) {
    auto ranked = mlir::dyn_cast<mlir::RankedTensorType>(result.getType());
    if (!ranked) {
      continue;
    }
    if (ranked.getRank() != static_cast<int64_t>(exprs.size())) {
      continue;
    }
    auto canonicalType = canonicalizeTypeWithSymbolicExprs(diagOp, ranked, exprs, builder, analysis);
    if (mlir::failed(canonicalType)) {
      return mlir::failure();
    }
    result.setType(*canonicalType);
  }
  return mlir::success();
}

struct ConvertTorchSymbolToMfusePass
    : public mlir::PassWrapper<ConvertTorchSymbolToMfusePass, mlir::OperationPass<mlir::ModuleOp>> {
  ConvertTorchSymbolToMfusePass() = default;
  ConvertTorchSymbolToMfusePass(const ConvertTorchSymbolToMfusePass &pass) : PassWrapper(pass) {}

  Option<bool> removeAllBindOps{*this, "remove_all_bind_ops",
                                llvm::cl::desc("Remove all torch.bind_symbolic_shape ops"), llvm::cl::init(true)};

  mlir::StringRef getArgument() const final { return "convert-torch-symbol-to-mfuse"; }
  mlir::StringRef getDescription() const final {
    return "Attach symbolic shape attributes after Torch-to-Mfuse conversion";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<TorchD::TorchDialect>();
    registry.insert<mlir::mfuse::MfuseDialect>();
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext *ctx = &getContext();

    mfusion::SymEngineAnalysis analysis;
    handleSymbolicIntOps(module, ctx);
    if (mlir::failed(handleBindSymbolicShapeOps(module, ctx, analysis))) {
      signalPassFailure();
    }
  }

 private:
  void handleSymbolicIntOps(mlir::ModuleOp module, mlir::MLIRContext *ctx) {
    llvm::SmallVector<TorchD::SymbolicIntOp> symIntOps;
    module.walk([&](TorchD::SymbolicIntOp op) { symIntOps.push_back(op); });

    llvm::DenseMap<mlir::func::FuncOp, llvm::SmallVector<mlir::NamedAttribute>> funcSymInfo;
    for (auto symOp : symIntOps) {
      auto func = symOp->getParentOfType<mlir::func::FuncOp>();
      if (!func) {
        continue;
      }
      auto nameAttr = mlir::StringAttr::get(ctx, symOp.getSymbolName());
      auto infoAttr = mlir::mfuse::SymbolInfoAttr::get(ctx, symOp.getMinVal(), symOp.getMaxVal());
      funcSymInfo[func].emplace_back(nameAttr, infoAttr);
    }

    for (auto &entry : funcSymInfo) {
      auto func = entry.first;
      auto existing = func->getAttrOfType<mlir::DictionaryAttr>("mfuse.syminfo");
      llvm::SmallVector<mlir::NamedAttribute> merged;
      if (existing) {
        merged.assign(existing.getValue().begin(), existing.getValue().end());
      }

      for (const auto &attr : entry.second) {
        auto it = std::find_if(merged.begin(), merged.end(),
                               [&](const mlir::NamedAttribute &item) { return item.getName() == attr.getName(); });
        if (it == merged.end()) {
          merged.push_back(attr);
        } else if (it->getValue() != attr.getValue()) {
          *it = attr;
        }
      }

      func->setAttr("mfuse.syminfo", mlir::DictionaryAttr::get(ctx, merged));
    }
  }

  mlir::LogicalResult handleBindSymbolicShapeOps(mlir::ModuleOp module, mlir::MLIRContext *ctx,
                                                 mfusion::SymEngineAnalysis &analysis) {
    llvm::SmallVector<TorchD::BindSymbolicShapeOp> bindOps;
    module.walk([&](TorchD::BindSymbolicShapeOp op) { bindOps.push_back(op); });
    mlir::OpBuilder builder(ctx);

    // Resolver for Torch dialect symbolic integers.
    mfusion::SymEngineAnalysis::SymbolNameResolver torchResolver = [](mlir::Value v) -> mlir::FailureOr<std::string> {
      if (auto symIntOp = v.getDefiningOp<TorchD::SymbolicIntOp>()) {
        return symIntOp.getSymbolName().str();
      }
      return mlir::failure();
    };

    llvm::SmallVector<mlir::Operation *> eraseOps;
    for (auto bindOp : bindOps) {
      auto affineMap = bindOp.getShapeExpressions().getValue();
      auto exprs = analysis.applyAffineMap(affineMap, bindOp.getShapeSymbols(), torchResolver);
      if (mlir::failed(exprs)) {
        bindOp.emitError("unsupported symbolic dim source; expected torch.symbolic_int values");
        return mlir::failure();
      }

      // Keep the bound Torch value type consistent with its bind_symbolic_shape fact.
      if (mlir::failed(canonicalizeTorchValueWithSymbolicExprs(bindOp, bindOp.getOperand(), *exprs, analysis))) {
        return mlir::failure();
      }

      // Case 1: Torch tensor used by casts to RankedTensorType.
      for (mlir::Operation *user : bindOp.getOperand().getUsers()) {
        auto castOp = llvm::dyn_cast<mlir::UnrealizedConversionCastOp>(user);
        if (!castOp) {
          continue;
        }
        if (mlir::failed(attachToCastResults(bindOp, castOp, *exprs, builder, analysis))) {
          return mlir::failure();
        }
      }

      // Case 2: Mfuse op result cast to Torch tensor, then bound by bind_symbolic_shape.
      // after this case, the bind_symbolic_shape and the unrealized_conversion_cast is no longer needed.
      // unrealized_conversion_cast will be removed by the post conversion pipeline.
      bool remove_op = false;
      if (auto castOp = bindOp.getOperand().getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
        for (mlir::Value input : castOp.getInputs()) {
          auto ranked = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
          if (!ranked || ranked.getRank() != static_cast<int64_t>(exprs->size())) {
            continue;
          }
          auto canonicalType = canonicalizeTypeWithSymbolicExprs(bindOp, ranked, *exprs, builder, analysis);
          if (mlir::failed(canonicalType)) {
            return mlir::failure();
          }
          if (setRankedTensorValueType(input, *canonicalType)) {
            remove_op = true;
          }
        }
      }

      if (removeAllBindOps || remove_op) {
        eraseOps.push_back(bindOp);
      }
    }

    for (mlir::Operation *op : eraseOps) {
      op->erase();
    }
    module.walk([&](mlir::func::FuncOp func) { refineFunctionResultTypes(func); });
    return mlir::success();
  }
};
}  // namespace

namespace mlir {
std::unique_ptr<Pass> createConvertTorchSymbolToMfusePass() {
  return std::make_unique<ConvertTorchSymbolToMfusePass>();
}
}  // namespace mlir
