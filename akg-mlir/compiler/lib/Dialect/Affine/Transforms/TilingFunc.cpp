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

// HACC dialect enums/attrs
#include "bishengir/Dialect/HACC/IR/HACC.h"

#define DEBUG_TYPE "tiling-func"

namespace mlir {
#define GEN_PASS_DECL_TILINGFUNC
#define GEN_PASS_DEF_TILINGFUNC
#include "akg/Dialect/Affine/Passes.h.inc"
}  // namespace mlir

namespace mlir::affine {

namespace mockattr {
static constexpr const char *kEnableAutoMarkBufferSize = "enable_auto_mark_buffer_size";
static constexpr const char *kFusionKind = "hfusion.fusion_kind";
static constexpr const char *kFusionKindPureElemwise = "PURE_ELEMWISE";
}  // namespace mockattr

namespace {

using hacc::BlockDimAttr;
using hacc::HACCFuncType;
using hacc::HACCFuncTypeAttr;
using hacc::HACCToLLVMIRTranslateAttr;
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
    if (auto *ctx = originalKernel_.getContext()) {
      ctx->getOrLoadDialect<hacc::HACCDialect>();
    }

    kernelInfo_->baseKernelName = originalKernel_.getSymName().str();
    kernelInfo_->blockDim = options_.blockDim;

    // annotate original as HOST (enum attr)
    auto *ctx = originalKernel_.getContext();
    originalKernel_->setAttr(HACCFuncTypeAttr::name, HACCFuncTypeAttr::get(ctx, HACCFuncType::HOST));
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

    SmallVector<Type> argTypes;
    argTypes.reserve(origTy.getNumInputs() + origTy.getNumResults());
    for (Type ty : origTy.getInputs()) {
      if (auto memrefType = dyn_cast<MemRefType>(ty)) {
        ty = RankedTensorType::get(memrefType.getShape(), memrefType.getElementType());
      }
      argTypes.push_back(ty);
    }
    for (Type ty : origTy.getResults()) {
      if (auto memrefType = dyn_cast<MemRefType>(ty)) {
        ty = RankedTensorType::get(memrefType.getShape(), memrefType.getElementType());
      }
      argTypes.push_back(ty);
    }

    SmallVector<Type> resTypes;
    resTypes.reserve(kN);
    for (unsigned i = 0; i < kN; ++i) resTypes.push_back(builder.getI64Type());

    std::string name = kernelInfo_->baseKernelName + "_single_outlined_0_0_tiling_function";
    auto funcTy = FunctionType::get(builder.getContext(), argTypes, resTypes);
    auto host = builder.create<func::FuncOp>(originalKernel_.getLoc(), name, funcTy);
    host.addEntryBlock();

    host->setAttr(HACCFuncTypeAttr::name, HACCFuncTypeAttr::get(builder.getContext(), HACCFuncType::HOST));
    host->setAttr(HostFuncTypeAttr::name, HostFuncTypeAttr::get(builder.getContext(), HostFuncType::kTilingFunction));

    unsigned nInputs = origTy.getNumInputs();
    unsigned nResults = origTy.getNumResults();

    auto *ctx = builder.getContext();

    for (unsigned i = 0; i < nInputs; ++i) {
      host.setArgAttr(i, KernelArgTypeAttr::name, KernelArgTypeAttr::get(ctx, KernelArgType::kInput));
      host.setArgAttr(i, InputIdxAttr::name, InputIdxAttr::get(ctx, i));
    }
    for (unsigned i = 0; i < nResults; ++i) {
      unsigned argIdx = nInputs + i;
      host.setArgAttr(argIdx, KernelArgTypeAttr::name, KernelArgTypeAttr::get(ctx, KernelArgType::kOutput));
      host.setArgAttr(argIdx, OutputIdxAttr::name, OutputIdxAttr::get(ctx, i));
    }

    host.setResultAttr(0, KernelArgTypeAttr::name, KernelArgTypeAttr::get(ctx, KernelArgType::kTilingKey));
    for (unsigned i = 1; i < kN; ++i) {
      host.setResultAttr(i, KernelArgTypeAttr::name, KernelArgTypeAttr::get(ctx, KernelArgType::kTilingData));
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

  LogicalResult collectDeviceSignature(func::FuncOp orig, SmallVector<Type> &devInputs, SmallVector<Type> &devResults) {
    auto origTy = orig.getFunctionType();

    devInputs.clear();
    devInputs.reserve(origTy.getNumInputs() + origTy.getNumResults());

    for (Type ty : origTy.getInputs()) {
      if (auto memrefType = dyn_cast<MemRefType>(ty)) {
        ty = RankedTensorType::get(memrefType.getShape(), memrefType.getElementType());
      }
      devInputs.push_back(ty);
    }
    for (Type ty : origTy.getResults()) {
      if (auto memrefType = dyn_cast<MemRefType>(ty)) {
        ty = RankedTensorType::get(memrefType.getShape(), memrefType.getElementType());
      }
      devInputs.push_back(ty);
    }

    devResults.clear();
    for (Type ty : origTy.getResults()) {
      if (auto memrefType = dyn_cast<MemRefType>(ty)) {
        ty = RankedTensorType::get(memrefType.getShape(), memrefType.getElementType());
      }
      devResults.push_back(ty);
    }
    return success();
  }

  void setHaccIOArgAttrs(func::FuncOp f, unsigned nInputs, unsigned nOutputs, OpBuilder &builder) {
    auto *ctx = builder.getContext();
    for (unsigned i = 0; i < nInputs; ++i) {
      f.setArgAttr(i, KernelArgTypeAttr::name, KernelArgTypeAttr::get(ctx, KernelArgType::kInput));
      f.setArgAttr(i, InputIdxAttr::name, InputIdxAttr::get(ctx, i));
    }
    for (unsigned i = 0; i < nOutputs; ++i) {
      unsigned argIdx = nInputs + i;
      f.setArgAttr(argIdx, KernelArgTypeAttr::name, KernelArgTypeAttr::get(ctx, KernelArgType::kOutput));
      f.setArgAttr(argIdx, OutputIdxAttr::name, OutputIdxAttr::get(ctx, i));
    }
  }

  func::FuncOp createAndAnnotateDeviceFunc(OpBuilder &builder, Location loc, StringRef name, FunctionType devTy,
                                           FunctionType origTy, unsigned blockDim, func::FuncOp hostTiling) {
    auto deviceFunc = builder.create<func::FuncOp>(loc, name, devTy);
    unsigned nInputs = origTy.getNumInputs();
    unsigned nOutputs = origTy.getNumResults();

    setHaccIOArgAttrs(deviceFunc, nInputs, nOutputs, builder);

    deviceFunc->setAttr(mockattr::kEnableAutoMarkBufferSize, builder.getUnitAttr());
    deviceFunc->setAttr(HACCFuncTypeAttr::name, HACCFuncTypeAttr::get(builder.getContext(), HACCFuncType::DEVICE));
    deviceFunc->setAttr(mockattr::kFusionKind,
                        StringAttr::get(builder.getContext(), mockattr::kFusionKindPureElemwise));

    deviceFunc->setAttr(BlockDimAttr::name, builder.getI64IntegerAttr(blockDim));
    deviceFunc->setAttr(
      StringAttr::get(builder.getContext(), stringifyHACCToLLVMIRTranslateAttr(HACCToLLVMIRTranslateAttr::ENTRY)),
      builder.getUnitAttr());
    if (hostTiling) {
      deviceFunc->setAttr(
        TilingFunctionAttr::name,
        TilingFunctionAttr::get(builder.getContext(), FlatSymbolRefAttr::get(hostTiling.getSymNameAttr())));
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

  void maybeRetargetInitAtTopmostEmpty(Value returned, Value outArg) {
    if (!returned || !outArg) return;

    auto target = findTopmostEmptyInitForAndIter(returned);
    if (!target) return;

    AffineForOp forOp = target->first;
    unsigned iterIdx = target->second;

    if (!forResultUpdatedByInsertSlice(forOp, iterIdx)) return;

    Value init = getForInitViaOperands(forOp, iterIdx);
    Value initUnwrapped = unwrapViewLikeProducers(init);
    Operation *def = initUnwrapped.getDefiningOp();
    if (!def || !isa<tensor::EmptyOp>(def)) return;

    replaceForInit(forOp, iterIdx, outArg);
  }

  static std::optional<std::pair<AffineForOp, unsigned>> findTopmostEmptyInitForAndIter(Value fromReturned) {
    Value start = unwrapViewLikeProducers(fromReturned);
    auto info = isResultDirectlyFromAffineFor(start);
    if (!info) return std::nullopt;

    AffineForOp curFor = info->first;
    unsigned curResIdx = info->second;
    unsigned curIterIdx = curResIdx;

    std::optional<std::pair<AffineForOp, unsigned>> topmost;

    llvm::SmallDenseSet<std::pair<void *, unsigned>, 8> seen;

    while (true) {
      if (!forResultUpdatedByInsertSlice(curFor, curResIdx)) break;

      Value init = getForInitViaOperands(curFor, curIterIdx);
      Value initUnwrapped = unwrapViewLikeProducers(init);

      if (Operation *def = initUnwrapped.getDefiningOp()) {
        if (isa<tensor::EmptyOp>(def)) {
          topmost = std::make_pair(curFor, curIterIdx);
          break;
        }
      }

      auto up = isResultDirectlyFromAffineFor(initUnwrapped);
      if (!up) break;

      auto key = std::make_pair(up->first.getOperation(), up->second);
      if (seen.contains(key)) break;
      seen.insert(key);

      curFor = up->first;
      curResIdx = up->second;
      curIterIdx = curResIdx;
    }

    return topmost;
  }

  static Value unwrapViewLikeProducers(Value v) {
    Value cur = v;
    while (true) {
      Operation *def = cur ? cur.getDefiningOp() : nullptr;
      if (!def) break;
      if (isa<tensor::CastOp>(def)) {
        cur = def->getOperand(0);
        continue;
      }
      if (auto ext = dyn_cast<tensor::ExtractSliceOp>(def)) {
        cur = ext.getSource();
        continue;
      }
      break;
    }
    return cur;
  }

  SmallVector<Value> cloneKernelBodyToDeviceFunc(func::FuncOp originalKernel, func::FuncOp deviceFunc, unsigned nInputs,
                                                 unsigned nOutputs, ArrayRef<Value> outArgs) {
    SmallVector<Value> returnedValues;
    if (originalKernel.empty()) return returnedValues;

    IRMapping map;
    Block &oldEntry = originalKernel.front();

    for (unsigned i = 0; i < std::min<unsigned>(oldEntry.getNumArguments(), nInputs); ++i) {
      map.map(oldEntry.getArgument(i), deviceFunc.getBody().front().getArgument(i));
    }

    if (oldEntry.getNumArguments() > nInputs && !outArgs.empty()) {
      map.map(oldEntry.getArgument(nInputs), outArgs[0]);
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

    if (oldRet) {
      unsigned numRet = oldRet.getNumOperands();
      returnedValues.resize(numRet);
      for (unsigned i = 0; i < numRet; ++i) {
        Value mapped = map.lookupOrNull(oldRet.getOperand(i));
        if (mapped) returnedValues[i] = mapped;
      }
    }
    return returnedValues;
  }

  LogicalResult initTilingKernel(OpBuilder &builder) {
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPoint(originalKernel_);

    SmallVector<Type> devInputs, devResults;
    if (failed(collectDeviceSignature(originalKernel_, devInputs, devResults))) return failure();

    std::string name = kernelInfo_->baseKernelName + "_single_outlined_0_0_0";
    auto devTy = FunctionType::get(builder.getContext(), devInputs, devResults);
    auto origTy = originalKernel_.getFunctionType();
    auto deviceFunc = createAndAnnotateDeviceFunc(builder, originalKernel_.getLoc(), name, devTy, origTy,
                                                  kernelInfo_->blockDim, tilingInfo_.getHostTilingFunc());

    Block *entry = deviceFunc.addEntryBlock();
    OpBuilder b = OpBuilder::atBlockEnd(entry);
    unsigned nInputs = origTy.getNumInputs();
    unsigned nOutputs = origTy.getNumResults();

    SmallVector<Value> outArgs;
    outArgs.reserve(nOutputs);
    for (unsigned i = 0; i < nOutputs; ++i) {
      outArgs.push_back(entry->getArgument(nInputs + i));
    }

    SmallVector<Value> returned = cloneKernelBodyToDeviceFunc(originalKernel_, deviceFunc, nInputs, nOutputs, outArgs);

    SmallVector<Value> finalReturns;
    finalReturns.reserve(nOutputs);
    for (unsigned i = 0; i < nOutputs; ++i) {
      Value ret = (i < returned.size()) ? returned[i] : Value();
      Value outArg = (i < outArgs.size()) ? outArgs[i] : Value();
      if (ret) {
        maybeRetargetInitAtTopmostEmpty(ret, outArg);
      }
      finalReturns.push_back(ret ? ret : outArg);
    }

    b.create<func::ReturnOp>(deviceFunc.getLoc(), finalReturns);
    tilingKernel_ = deviceFunc;

    if (failed(createOrGetGetTilingStructSizeFunction(builder, deviceFunc))) return failure();
    return success();
  }

  virtual LogicalResult applyTilingImpl(OpBuilder &) { return success(); }

  LogicalResult fixCallSitesAndCaller(OpBuilder &builder) {
    assert(tilingKernel_);

    auto oldTy = originalKernel_.getFunctionType();
    unsigned nInputs = oldTy.getNumInputs();
    unsigned nOutputs = oldTy.getNumResults();

    SmallVector<Type> newInputs;
    newInputs.reserve(oldTy.getNumInputs() + oldTy.getNumResults());
    {
      auto in = oldTy.getInputs();
      std::copy(in.begin(), in.end(), std::back_inserter(newInputs));
    }
    {
      auto outs = oldTy.getResults();
      std::copy(outs.begin(), outs.end(), std::back_inserter(newInputs));
    }

    SmallVector<Type> newResults;
    newResults.reserve(oldTy.getNumResults());
    {
      auto outs = oldTy.getResults();
      std::copy(outs.begin(), outs.end(), std::back_inserter(newResults));
    }

    originalKernel_.setFunctionType(FunctionType::get(builder.getContext(), newInputs, newResults));

    setHaccIOArgAttrs(originalKernel_, nInputs, nOutputs, builder);

    while (!originalKernel_.getBody().empty()) {
      originalKernel_.getBody().front().erase();
    }

    Block *entry = originalKernel_.addEntryBlock();
    OpBuilder b = OpBuilder::atBlockEnd(entry);
    Location loc = originalKernel_.getLoc();

    SmallVector<Value> passArgs(entry->args_begin(), entry->args_end());
    auto call = b.create<func::CallOp>(loc, tilingKernel_.getSymName(), TypeRange(newResults), passArgs);
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
    auto *ctx = builder.getContext();
    host->setAttr(HACCFuncTypeAttr::name, HACCFuncTypeAttr::get(ctx, HACCFuncType::HOST));
    host->setAttr(HostFuncTypeAttr::name, HostFuncTypeAttr::get(ctx, HostFuncType::kGetTilingStructSizeFunction));

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
      if (auto kind = f->getAttrOfType<StringAttr>(HACCFuncTypeAttr::name); !kind || kind.getValue() == "DEVICE") {
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
