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

#include "akg/Dialect/Affine/Transforms/TilingFunc.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <unordered_set>

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/IR/OpImplementation.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // ADD: tensor dialect ops

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

// HACC dialect
#include "bishengir/Dialect/HACC/IR/HACC.h"

#define DEBUG_TYPE "tiling-func"

namespace mlir {
#define GEN_PASS_DECL_TILINGFUNC  // FIX: use correct macro names
#define GEN_PASS_DEF_TILINGFUNC
#include "akg/Dialect/Affine/Passes.h.inc"
}  // namespace mlir

namespace mlir::affine {

namespace mockattr {
static constexpr const char *kEnableAutoMarkBufferSize = "enable_auto_mark_buffer_size";
static constexpr const char *kBlockDim = "hacc.block_dim";
static constexpr const char *kFusionKind = "hfusion.fusion_kind";
static constexpr const char *kFusionKindPureElemwise = "PURE_ELEMWISE";
}  // namespace mockattr

namespace {

using hacc::HACCFuncType;
using hacc::HACCFuncTypeAttr;
using hacc::HostFuncType;
using hacc::HostFuncTypeAttr;
using hacc::InputIdxAttr;
using hacc::KernelArgType;
using hacc::KernelArgTypeAttr;
using hacc::OutputIdxAttr;
using hacc::TilingFunctionAttr;

struct AutoTilingOptions {
  unsigned blockDim = 40;
};

struct KernelInfo {
  func::FuncOp originalKernel;
  std::string baseKernelName;
  unsigned blockDim = 40;
};

struct TilingInfo {
  func::FuncOp hostTilingFunc;
  void setHostTilingFunc(func::FuncOp f) { hostTilingFunc = f; }
  func::FuncOp getHostTilingFunc() const { return hostTilingFunc; }
};

class TilingBase {
 public:
  explicit TilingBase(func::FuncOp f)
      : originalKernel_(f),
        module_(f ? f->getParentOfType<ModuleOp>() : ModuleOp()),
        kernelInfo_(std::make_unique<KernelInfo>()),
        tilingInfo_(),
        tilingKernel_() {
    if (kernelInfo_) {
      kernelInfo_->originalKernel = f;
    }
  }
  virtual ~TilingBase() = default;

  LogicalResult runOnOperation(OpBuilder &builder) {
    if (failed(runPreTilingProcedure(builder))) return failure();
    if (failed(runTilingProcedure(builder))) return failure();
    if (failed(runPostTilingProcedure(builder))) return failure();
    return success();
  }

  static void setAutoTilingOptions(const AutoTilingOptions &opt) { options_ = opt; }

 protected:
  LogicalResult runPreTilingProcedure(OpBuilder &) {
    kernelInfo_->baseKernelName = originalKernel_.getSymName().str();
    kernelInfo_->blockDim = options_.blockDim;

    auto *ctx = originalKernel_.getContext();
    originalKernel_->setAttr("hacc.function_kind", HACCFuncTypeAttr::get(ctx, HACCFuncType::HOST));

    return success();
  }

  LogicalResult runTilingProcedure(OpBuilder &builder) {
    if (failed(createHostTilingFunction(builder))) return failure();
    if (failed(initTilingKernel(builder))) return failure();
    if (failed(applyTilingImpl(builder))) return failure();
    if (failed(fixCallSitesAndCaller(builder))) return failure();
    return success();
  }
  LogicalResult runPostTilingProcedure(OpBuilder &) { return success(); }

  LogicalResult createHostTilingFunction(OpBuilder &builder) {
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPoint(originalKernel_);

    constexpr int64_t kDummyTiling[] = {0, 12280, 13, 1, 49120};
    constexpr unsigned kN = static_cast<unsigned>(sizeof(kDummyTiling) / sizeof(kDummyTiling[0]));
    static_assert(kN >= 1, "host tiling function must return at least 1 result");

    auto origTy = originalKernel_.getFunctionType();
    if (origTy.getNumResults() != 1) {
      originalKernel_.emitError("expect exactly 1 result before rewriting");
      return failure();
    }

    SmallVector<Type> argTypes;
    argTypes.reserve(origTy.getNumInputs() + 1);
    for (Type ty : origTy.getInputs()) {
      if (auto memrefType = dyn_cast<MemRefType>(ty)) {
        ty = RankedTensorType::get(memrefType.getShape(), memrefType.getElementType());
      }
      argTypes.push_back(ty);
    }

    Type outTy = origTy.getResult(0);
    if (auto memrefType = dyn_cast<MemRefType>(outTy)) {
      outTy = RankedTensorType::get(memrefType.getShape(), memrefType.getElementType());
    }
    argTypes.push_back(outTy);

    SmallVector<Type> resTypes(kN, builder.getI64Type());

    std::string name = kernelInfo_->baseKernelName + "_single_outlined_0_0_tiling_function";
    auto funcTy = FunctionType::get(builder.getContext(), argTypes, resTypes);
    auto host = builder.create<func::FuncOp>(originalKernel_.getLoc(), name, funcTy);
    host.addEntryBlock();

    host->setAttr("hacc.function_kind", HACCFuncTypeAttr::get(builder.getContext(), HACCFuncType::HOST));
    host->setAttr("hacc.host_func_type", HostFuncTypeAttr::get(builder.getContext(), HostFuncType::kTilingFunction));

    unsigned nInputs = origTy.getNumInputs();
    auto *ctx = builder.getContext();
    for (unsigned i = 0; i < nInputs; ++i) {
      host.setArgAttr(i, "hacc.arg_type", KernelArgTypeAttr::get(ctx, KernelArgType::kInput));
      host.setArgAttr(i, "hacc.input_idx", InputIdxAttr::get(ctx, i));
    }
    host.setArgAttr(nInputs, "hacc.arg_type", KernelArgTypeAttr::get(ctx, KernelArgType::kOutput));
    host.setArgAttr(nInputs, "hacc.output_idx", OutputIdxAttr::get(ctx, 0));

    host.setResultAttr(0, "hacc.arg_type", KernelArgTypeAttr::get(ctx, KernelArgType::kSyncBlockLock));
    for (unsigned i = 1; i < kN; ++i) {
      host.setResultAttr(i, "hacc.arg_type", KernelArgTypeAttr::get(ctx, KernelArgType::kTilingKey));
    }
    builder.setInsertionPointToEnd(&host.getBody().front());
    SmallVector<Value> cst;
    cst.reserve(kN);
    for (unsigned i = 0; i < kN; ++i) {
      cst.push_back(builder.create<arith::ConstantIntOp>(host.getLoc(), kDummyTiling[i], 64));
    }
    builder.create<func::ReturnOp>(host.getLoc(), cst);

    tilingInfo_.setHostTilingFunc(host);
    return success();
  }

  LogicalResult collectDeviceSignature(func::FuncOp orig, SmallVector<Type> &devInputs, Type &outTy,
                                       SmallVector<Type> &devResults) {
    auto origTy = orig.getFunctionType();
    if (origTy.getNumResults() != 1) {
      orig.emitError("expect exactly 1 result");
      return failure();
    }
    devInputs.clear();
    devInputs.reserve(origTy.getNumInputs() + 1);

    for (Type ty : origTy.getInputs()) {
      if (auto memrefType = dyn_cast<MemRefType>(ty)) {
        ty = RankedTensorType::get(memrefType.getShape(), memrefType.getElementType());
      }
      devInputs.push_back(ty);
    }

    outTy = origTy.getResult(0);
    if (auto memrefType = dyn_cast<MemRefType>(outTy)) {
      outTy = RankedTensorType::get(memrefType.getShape(), memrefType.getElementType());
    }
    devInputs.push_back(outTy);

    devResults.clear();
    devResults.push_back(outTy);
    return success();
  }

  void setHaccIOArgAttrs(func::FuncOp f, unsigned nInputs, OpBuilder &builder, bool isOutputOnLastArg) {
    auto *ctx = builder.getContext();
    for (unsigned i = 0; i < nInputs; ++i) {
      f.setArgAttr(i, "hacc.arg_type", KernelArgTypeAttr::get(ctx, KernelArgType::kInput));
      f.setArgAttr(i, "hacc.input_idx", InputIdxAttr::get(ctx, i));
    }
    if (isOutputOnLastArg) {
      f.setArgAttr(nInputs, "hacc.arg_type", KernelArgTypeAttr::get(ctx, KernelArgType::kOutput));
      f.setArgAttr(nInputs, "hacc.output_idx", OutputIdxAttr::get(ctx, 0));
    }
  }

  func::FuncOp createAndAnnotateDeviceFunc(OpBuilder &builder, Location loc, StringRef name, FunctionType devTy,
                                           FunctionType origTy, unsigned blockDim, func::FuncOp hostTiling) {
    auto deviceFunc = builder.create<func::FuncOp>(loc, name, devTy);
    unsigned nInputs = origTy.getNumInputs();

    setHaccIOArgAttrs(deviceFunc, nInputs, builder, /*isOutputOnLastArg=*/true);

    deviceFunc->setAttr(mockattr::kEnableAutoMarkBufferSize, builder.getUnitAttr());

    deviceFunc->setAttr("hacc.function_kind", HACCFuncTypeAttr::get(builder.getContext(), HACCFuncType::DEVICE));
    deviceFunc->setAttr(mockattr::kFusionKind,
                        StringAttr::get(builder.getContext(), mockattr::kFusionKindPureElemwise));
    deviceFunc->setAttr(mockattr::kBlockDim, builder.getI64IntegerAttr(blockDim));

    deviceFunc->setAttr("hacc.entry", builder.getUnitAttr());

    if (hostTiling) {
      deviceFunc->setAttr(
        "hacc.tiling_function",
        TilingFunctionAttr::get(builder.getContext(), FlatSymbolRefAttr::get(hostTiling.getSymNameAttr())));
    }
    return deviceFunc;
  }

  static void replaceForInit(AffineForOp forOp, unsigned idx, Value newInit) {
    auto inits = SmallVector<Value>(forOp.getInits().begin(), forOp.getInits().end());
    if (idx >= inits.size()) return;
    if (inits[idx] == newInit) return;

    unsigned start = forOp.getLowerBoundOperands().size() + forOp.getUpperBoundOperands().size();
    inits[idx] = newInit;
    forOp.getOperation()->setOperands(start, static_cast<unsigned>(inits.size()), inits);

    Block &body = forOp.getRegion().front();
    unsigned iterArgStart = body.getNumArguments() - static_cast<unsigned>(inits.size());
    BlockArgument iterArg = body.getArgument(iterArgStart + idx);
    iterArg.replaceAllUsesWith(newInit);
  }

  static void retargetAllForInitsToOut(Value returned, Value outArg) {
    if (!returned || !outArg || returned == outArg) return;

    SmallVector<Value, 8> worklist;
    worklist.push_back(returned);

    llvm::SmallPtrSet<void *, 16> seen;

    while (!worklist.empty()) {
      Value v = worklist.pop_back_val();
      if (!seen.insert(v.getAsOpaquePointer()).second) continue;

      if (auto res = llvm::dyn_cast<OpResult>(v)) {
        Operation *def = res.getDefiningOp();
        if (!def) continue;

        if (auto forOp = llvm::dyn_cast<AffineForOp>(def)) {
          auto results = forOp.getResults();
          for (unsigned i = 0, e = results.size(); i < e; ++i) {
            if (results[i] == v) {
              replaceForInit(forOp, i, outArg);
              Value init = forOp.getInits()[i];
              worklist.push_back(init);
              break;
            }
          }
        } else if (auto ins = llvm::dyn_cast<tensor::InsertSliceOp>(def)) {
          worklist.push_back(ins.getDest());
          worklist.push_back(ins.getSource());
        } else if (auto addf = llvm::dyn_cast<arith::AddFOp>(def)) {
          worklist.push_back(addf.getLhs());
          worklist.push_back(addf.getRhs());
        } else if (auto exf = llvm::dyn_cast<arith::ExtFOp>(def)) {
          worklist.push_back(exf.getIn());
        } else if (auto exsi = llvm::dyn_cast<arith::ExtSIOp>(def)) {
          worklist.push_back(exsi.getIn());
        } else if (auto slice = llvm::dyn_cast<tensor::ExtractSliceOp>(def)) {
          worklist.push_back(slice.getSource());
        } else if (llvm::isa<tensor::EmptyOp>(def)) {
          // no-op
        } else {
          for (Value opnd : def->getOperands()) worklist.push_back(opnd);
        }
      }
    }
  }

  Value cloneKernelBodyToDeviceFunc(func::FuncOp originalKernel, func::FuncOp deviceFunc, unsigned nInputs,
                                    Value outArg) {
    Value returned;
    if (originalKernel.empty()) return returned;

    IRMapping map;
    Block &oldEntry = originalKernel.front();

    unsigned argToMap = std::min<unsigned>(oldEntry.getNumArguments(), nInputs);
    for (unsigned i = 0; i < argToMap; ++i) {
      map.map(oldEntry.getArgument(i), deviceFunc.getBody().front().getArgument(i));
    }
    if (oldEntry.getNumArguments() > nInputs) {
      map.map(oldEntry.getArgument(nInputs), outArg);
    }

    func::ReturnOp oldRet = nullptr;
    SmallVector<Operation *> toClone;
    toClone.reserve(oldEntry.getOperations().size());
    for (Operation &op : oldEntry) {
      if (auto r = llvm::dyn_cast<func::ReturnOp>(op)) {
        oldRet = r;
        continue;
      }
      toClone.push_back(&op);
    }

    OpBuilder b = OpBuilder::atBlockEnd(&deviceFunc.getBody().front());
    for (Operation *op : toClone) b.clone(*op, map);

    if (oldRet && oldRet.getNumOperands() == 1) {
      Value mapped = map.lookupOrNull(oldRet.getOperand(0));
      if (mapped) returned = mapped;
    }
    return returned;
  }

  void retargetAffineForInitsIfNeeded(func::FuncOp deviceFunc, Value returned, Value outArg) {
    if (!outArg) return;

    if (!returned) {
      deviceFunc.walk([&](func::ReturnOp ret) {
        if (ret.getNumOperands() == 1) {
          retargetAllForInitsToOut(ret.getOperand(0), outArg);
        }
      });
      return;
    }

    if (returned == outArg) return;
    retargetAllForInitsToOut(returned, outArg);
  }

  LogicalResult initTilingKernel(OpBuilder &builder) {
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPoint(originalKernel_);

    SmallVector<Type> devInputs, devResults;
    Type outTy = nullptr;
    if (failed(collectDeviceSignature(originalKernel_, devInputs, outTy, devResults))) return failure();

    std::string name = kernelInfo_->baseKernelName + "_single_outlined_0_0_0";
    auto devTy = FunctionType::get(builder.getContext(), devInputs, devResults);
    auto origTy = originalKernel_.getFunctionType();
    auto deviceFunc = createAndAnnotateDeviceFunc(builder, originalKernel_.getLoc(), name, devTy, origTy,
                                                  kernelInfo_->blockDim, tilingInfo_.getHostTilingFunc());

    Block *entry = deviceFunc.addEntryBlock();
    OpBuilder b = OpBuilder::atBlockEnd(entry);
    unsigned nInputs = origTy.getNumInputs();
    Value outArg = entry->getArgument(nInputs);

    Value returned = cloneKernelBodyToDeviceFunc(originalKernel_, deviceFunc, nInputs, outArg);
    if (!returned) returned = outArg;

    retargetAffineForInitsIfNeeded(deviceFunc, returned, outArg);

    b.create<func::ReturnOp>(deviceFunc.getLoc(), ValueRange{outArg});
    tilingKernel_ = deviceFunc;

    if (failed(createOrGetGetTilingStructSizeFunction(builder, deviceFunc))) return failure();
    return success();
  }

  virtual LogicalResult applyTilingImpl(OpBuilder &) { return success(); }

  LogicalResult fixCallSitesAndCaller(OpBuilder &builder) {
    assert(tilingKernel_);

    auto oldTy = originalKernel_.getFunctionType();
    unsigned nInputs = oldTy.getNumInputs();
    Type outTy = oldTy.getResult(0);

    SmallVector<Type> newInputs(oldTy.getInputs().begin(), oldTy.getInputs().end());
    newInputs.push_back(outTy);
    SmallVector<Type> newResults = {outTy};

    originalKernel_.setFunctionType(FunctionType::get(builder.getContext(), newInputs, newResults));

    setHaccIOArgAttrs(originalKernel_, nInputs, builder, /*isOutputOnLastArg=*/true);

    while (!originalKernel_.getBody().empty()) {
      originalKernel_.getBody().front().erase();
    }

    Block *entry = originalKernel_.addEntryBlock();
    OpBuilder b = OpBuilder::atBlockEnd(entry);
    Location loc = originalKernel_.getLoc();

    SmallVector<Value> passArgs(entry->args_begin(), entry->args_end());
    auto call = b.create<func::CallOp>(loc, tilingKernel_.getSymName(), TypeRange{outTy}, passArgs);
    b.create<func::ReturnOp>(loc, call.getResults());

    return success();
  }

  LogicalResult createOrGetGetTilingStructSizeFunction(OpBuilder &builder, func::FuncOp deviceFunc) {
    ModuleOp module = deviceFunc->getParentOfType<ModuleOp>();
    if (!module) {
      deviceFunc.emitError("cannot find parent ModuleOp for device function");
      return failure();
    }

    std::string base = deviceFunc.getSymName().str();
    if (base.size() >= 2 && base.compare(base.size() - 2, 2, "_0") == 0) {
      base = base.substr(0, base.size() - 2);
    }

    std::string hostName = base + "_get_tiling_struct_size_function";

    if (auto sym = SymbolTable::lookupSymbolIn(module, StringAttr::get(module.getContext(), hostName))) {
      if (auto f = dyn_cast<func::FuncOp>(sym)) {
        auto expectedTy = FunctionType::get(module.getContext(), TypeRange{}, TypeRange{builder.getI64Type()});
        if (f.getFunctionType() != expectedTy) {
          f.emitError("found existing _get_tiling_struct_size_function with mismatched type");
          return failure();
        }
        auto *ctx = module.getContext();
        f->setAttr("hacc.function_kind", HACCFuncTypeAttr::get(ctx, HACCFuncType::HOST));
        f->setAttr("hacc.host_func_type", HostFuncTypeAttr::get(ctx, HostFuncType::kInferSyncBlockLockNumFunction));
        return success();
      }
    }

    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToStart(module.getBody());

    auto funcTy = FunctionType::get(module.getContext(), TypeRange{}, TypeRange{builder.getI64Type()});
    auto host = builder.create<func::FuncOp>(deviceFunc.getLoc(), hostName, funcTy);
    host.setVisibility(SymbolTable::Visibility::Public);

    auto *ctx = builder.getContext();
    host->setAttr("hacc.function_kind", HACCFuncTypeAttr::get(ctx, HACCFuncType::HOST));
    host->setAttr("hacc.host_func_type", HostFuncTypeAttr::get(ctx, HostFuncType::kInferSyncBlockLockNumFunction));

    Block *entry = host.addEntryBlock();
    OpBuilder b = OpBuilder::atBlockEnd(entry);
    auto zero = b.create<arith::ConstantIntOp>(host.getLoc(), 0, 64);
    b.create<func::ReturnOp>(host.getLoc(), ValueRange{zero});
    return success();
  }

 protected:
  func::FuncOp originalKernel_;
  ModuleOp module_;
  std::unique_ptr<KernelInfo> kernelInfo_;
  TilingInfo tilingInfo_;
  func::FuncOp tilingKernel_;
  static AutoTilingOptions options_;
};

AutoTilingOptions TilingBase::options_;

class PureElemwiseTiling : public TilingBase {
 public:
  using TilingBase::TilingBase;
  LogicalResult applyTilingImpl(OpBuilder &) override { return success(); }
};

struct TilingFunc : public mlir::impl::TilingFuncBase<TilingFunc> {
  TilingFunc() = default;
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (!module) {
      signalPassFailure();
      return;
    }

    SmallVector<func::FuncOp> kernels;
    module.walk([&](func::FuncOp f) {
      StringRef name = f.getSymName();
      if (f.empty()) return;
      kernels.push_back(f);
    });

    AutoTilingOptions opts;
    TilingBase::setAutoTilingOptions(opts);

    OpBuilder builder(module.getContext());
    for (func::FuncOp k : kernels) {
      PureElemwiseTiling sch(k);
      if (failed(sch.runOnOperation(builder))) {
        k.emitError("auto-tiling failed");
        signalPassFailure();
        return;
      }
    }
  }
};

}  // namespace
}  // namespace mlir::affine

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> mlir::affine::createTilingFuncPass() {
  return std::make_unique<TilingFunc>();
}
