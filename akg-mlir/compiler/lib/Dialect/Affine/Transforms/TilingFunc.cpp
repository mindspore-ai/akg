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

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "tiling-func"

namespace mlir {
#define GEN_PASS_DECL_TILINGFUNC
#define GEN_PASS_DEF_TILINGFUNC
#include "akg/Dialect/Affine/Passes.h.inc"
}  // namespace mlir

namespace mlir::affine {

namespace mockattr {
static constexpr const char *kFunctionKind = "hacc.function_kind";
static constexpr const char *kHostFuncType = "hacc.host_func_type";
static constexpr const char *kEnableAutoMarkBufferSize = "enable_auto_mark_buffer_size";
static constexpr const char *kBlockDim = "hacc.block_dim";
static constexpr const char *kTilingFunction = "hacc.tiling_function";
static constexpr const char *kFusionKind = "hfusion.fusion_kind";
static constexpr const char *kDevice = "DEVICE";
static constexpr const char *kHost = "HOST";
static constexpr const char *kHostTilingFunction = "tiling_function";
static constexpr const char *kFusionKindPureElemwise = "PURE_ELEMWISE";
}  // namespace mockattr

namespace {

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

    SmallVector<Type> resTypes;
    resTypes.reserve(kN);
    for (unsigned i = 0; i < kN; ++i) resTypes.push_back(builder.getI64Type());

    std::string name = kernelInfo_->baseKernelName + "_single_outlined_0_0_tiling_function";
    auto funcTy = FunctionType::get(builder.getContext(), argTypes, resTypes);
    auto host = builder.create<func::FuncOp>(originalKernel_.getLoc(), name, funcTy);
    host.addEntryBlock();

    host->setAttr(mockattr::kFunctionKind, StringAttr::get(builder.getContext(), mockattr::kHost));
    host->setAttr(mockattr::kHostFuncType, StringAttr::get(builder.getContext(), mockattr::kHostTilingFunction));

    unsigned nInputs = origTy.getNumInputs();
    for (unsigned i = 0; i < nInputs; ++i) {
      host.setArgAttr(i, "hacc.arg_type", StringAttr::get(builder.getContext(), "input"));
      host.setArgAttr(i, "hacc.input_idx", builder.getI64IntegerAttr(i));
    }
    host.setArgAttr(nInputs, "hacc.arg_type", StringAttr::get(builder.getContext(), "output"));
    host.setArgAttr(nInputs, "hacc.output_idx", builder.getI64IntegerAttr(0));

    host.setResultAttr(0, "hacc.arg_type", StringAttr::get(builder.getContext(), "tiling_key"));
    for (unsigned i = 1; i < kN; ++i) {
      host.setResultAttr(i, "hacc.arg_type", StringAttr::get(builder.getContext(), "tiling_data"));
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
    for (unsigned i = 0; i < nInputs; ++i) {
      f.setArgAttr(i, "hacc.arg_type", StringAttr::get(builder.getContext(), "input"));
      f.setArgAttr(i, "hacc.input_idx", builder.getI64IntegerAttr(i));
    }
    if (isOutputOnLastArg) {
      f.setArgAttr(nInputs, "hacc.arg_type", StringAttr::get(builder.getContext(), "output"));
      f.setArgAttr(nInputs, "hacc.output_idx", builder.getI64IntegerAttr(0));
    }
  }

  func::FuncOp createAndAnnotateDeviceFunc(OpBuilder &builder, Location loc, StringRef name, FunctionType devTy,
                                           FunctionType origTy, unsigned blockDim, func::FuncOp hostTiling) {
    auto deviceFunc = builder.create<func::FuncOp>(loc, name, devTy);
    unsigned nInputs = origTy.getNumInputs();

    setHaccIOArgAttrs(deviceFunc, nInputs, builder, /*isOutputOnLastArg=*/true);

    deviceFunc->setAttr(mockattr::kEnableAutoMarkBufferSize, builder.getUnitAttr());
    deviceFunc->setAttr(mockattr::kFunctionKind, StringAttr::get(builder.getContext(), mockattr::kDevice));
    deviceFunc->setAttr(mockattr::kFusionKind,
                        StringAttr::get(builder.getContext(), mockattr::kFusionKindPureElemwise));
    deviceFunc->setAttr(mockattr::kBlockDim, builder.getI64IntegerAttr(blockDim));
    deviceFunc->setAttr("hacc.entry", builder.getUnitAttr());
    if (hostTiling) {
      deviceFunc->setAttr(mockattr::kTilingFunction, FlatSymbolRefAttr::get(hostTiling.getSymNameAttr()));
    }
    return deviceFunc;
  }

  static Value getForInitViaOperands(AffineForOp forOp, unsigned idx) {
    unsigned start = forOp.getLowerBoundOperands().size() + forOp.getUpperBoundOperands().size();
    return forOp->getOperand(start + idx);
  }

  static void replaceForInit(AffineForOp forOp, unsigned idx, Value newInit) {
    unsigned numIter = forOp.getNumIterOperands();
    if (idx >= numIter) return;

    SmallVector<Value, 4> inits;
    inits.reserve(numIter);
    for (unsigned i = 0; i < numIter; ++i) {
      inits.push_back(i == idx ? newInit : getForInitViaOperands(forOp, i));
    }
    unsigned start = forOp.getLowerBoundOperands().size() + forOp.getUpperBoundOperands().size();
    forOp.getOperation()->setOperands(start, static_cast<unsigned>(inits.size()), inits);
  }

  static void processBlockArgumentIfFromForIter(Value cur, SmallVector<Value, 16> &wl) {
    if (auto barg = dyn_cast<BlockArgument>(cur)) {
      Block *owner = barg.getOwner();
      if (owner && owner->getParentOp()) {
        if (auto forOp = dyn_cast<AffineForOp>(owner->getParentOp())) {
          unsigned numIters = forOp.getNumIterOperands();
          unsigned firstIterIndex = owner->getNumArguments() - numIters;
          if (barg.getArgNumber() >= firstIterIndex) {
            unsigned iterIdx = barg.getArgNumber() - firstIterIndex;
            Value init = getForInitViaOperands(forOp, iterIdx);
            wl.push_back(init);
            if (!forOp.getRegion().empty()) {
              for (Operation &op : forOp.getRegion().front()) {
                if (auto y = dyn_cast<AffineYieldOp>(op)) {
                  if (iterIdx < y.getNumOperands()) wl.push_back(y.getOperand(iterIdx));
                }
              }
            }
          }
        }
      }
    }
  }

  static bool isValueUpdatedByInsertSlice(Value v) {
    if (!v) return false;
    SmallVector<Value, 16> wl;
    llvm::SmallPtrSet<void *, 32> seen;
    wl.push_back(v);

    while (!wl.empty()) {
      Value cur = wl.pop_back_val();
      if (!seen.insert(cur.getAsOpaquePointer()).second) continue;

      if (auto res = dyn_cast<OpResult>(cur)) {
        Operation *def = res.getDefiningOp();
        if (!def) continue;

        if (isa<tensor::InsertSliceOp>(def)) {
          return true;
        }
        if (auto forOp = dyn_cast<AffineForOp>(def)) {
          wl.append(def->operand_begin(), def->operand_end());
          if (!forOp.getRegion().empty()) {
            for (Operation &op : forOp.getRegion().front()) {
              if (auto y = dyn_cast<AffineYieldOp>(op)) {
                wl.append(y->operand_begin(), y->operand_end());
              }
            }
          }
          continue;
        }
        if (auto slice = dyn_cast<tensor::ExtractSliceOp>(def)) {
          wl.push_back(slice.getSource());
          continue;
        }
        wl.append(def->operand_begin(), def->operand_end());
        continue;
      }

      processBlockArgumentIfFromForIter(cur, wl);
    }
    return false;
  }

  static bool forResultUpdatedByInsertSlice(AffineForOp forOp, unsigned resultIdx) {
    if (resultIdx >= forOp.getNumResults()) return false;
    Value res = forOp.getResult(resultIdx);
    return isValueUpdatedByInsertSlice(res);
  }

  static std::optional<std::pair<AffineForOp, unsigned>> isResultDirectlyFromAffineFor(Value v) {
    if (!v) return std::nullopt;
    if (auto res = dyn_cast<OpResult>(v)) {
      if (auto forOp = dyn_cast<AffineForOp>(res.getOwner())) {
        auto results = forOp.getResults();
        for (unsigned i = 0, e = results.size(); i < e; ++i) {
          if (results[i] == v) return std::make_pair(forOp, i);
        }
      }
    }
    return std::nullopt;
  }

  void maybeRetargetForInitForDirectReturn(Value returned, Value outArg) {
    if (!returned || !outArg) return;
    auto info = isResultDirectlyFromAffineFor(returned);
    if (!info) return;

    AffineForOp forOp = info->first;
    unsigned idx = info->second;

    Value init = getForInitViaOperands(forOp, idx);
    auto initRes = dyn_cast_or_null<OpResult>(init);
    if (!initRes) return;
    Operation *def = initRes.getDefiningOp();
    if (!def || !isa<tensor::EmptyOp>(def)) return;

    if (!forResultUpdatedByInsertSlice(forOp, idx)) return;

    replaceForInit(forOp, idx, outArg);
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
      if (auto r = dyn_cast<func::ReturnOp>(op)) {
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

    maybeRetargetForInitForDirectReturn(returned, outArg);

    Value finalReturn = returned ? returned : outArg;
    b.create<func::ReturnOp>(deviceFunc.getLoc(), ValueRange{finalReturn});
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
    if (base.size() >= 2 && base.substr(base.size() - 2) == "_0") {
      base = base.substr(0, base.size() - 2);
    }

    std::string hostName = base + "_get_tiling_struct_size_function";
    if (auto sym = SymbolTable::lookupSymbolIn(module, StringAttr::get(module.getContext(), hostName))) {
      if (isa<func::FuncOp>(sym)) return success();
    }

    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToStart(module.getBody());
    auto funcTy = FunctionType::get(module.getContext(), TypeRange{}, TypeRange{builder.getI64Type()});
    auto host = builder.create<func::FuncOp>(deviceFunc.getLoc(), hostName, funcTy);
    host.setVisibility(SymbolTable::Visibility::Public);
    host->setAttr(mockattr::kFunctionKind, StringAttr::get(builder.getContext(), mockattr::kHost));

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
      if (auto kind = f->getAttrOfType<StringAttr>(mockattr::kFunctionKind);
          !kind || kind.getValue() == mockattr::kDevice) {
        kernels.push_back(f);
      }
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
