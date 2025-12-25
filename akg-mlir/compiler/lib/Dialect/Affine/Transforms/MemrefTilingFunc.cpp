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

#include "akg/Dialect/Affine/Transforms/MemrefTilingFunc.h"

#include <algorithm>
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

#define DEBUG_TYPE "memref-tiling-func"

namespace mlir {
#define GEN_PASS_DECL_MEMREFTILINGFUNC
#define GEN_PASS_DEF_MEMREFTILINGFUNC
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
    if (kernelInfo_) kernelInfo_->originalKernel = f;
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
    auto *ctx = originalKernel_.getContext();
    ctx->getOrLoadDialect<hacc::HACCDialect>();

    kernelInfo_->baseKernelName = originalKernel_.getSymName().str();
    kernelInfo_->blockDim = options_.blockDim;

    originalKernel_->setAttr(HACCFuncTypeAttr::name, HACCFuncTypeAttr::get(ctx, HACCFuncType::HOST));
    return success();
  }

  LogicalResult runTilingProcedure(OpBuilder &builder) {
    if (failed(createHostTilingFunction(builder))) return failure();
    SmallVector<int64_t> tilingkeys = {0};

    bool allOk = std::all_of(tilingkeys.begin(), tilingkeys.end(),
                             [&](int64_t key) { return succeeded(initTilingKernel(key, builder)); });
    if (!allOk) return failure();

    if (failed(applyTilingImpl(builder))) return failure();
    if (failed(fixCallSitesAndCaller(builder))) return failure();
    return success();
  }

  LogicalResult runPostTilingProcedure(OpBuilder &) { return success(); }

  // ---------------- Host tiling function ----------------
  LogicalResult createHostTilingFunction(OpBuilder &builder) {
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPoint(originalKernel_);

    constexpr int64_t kDummyTiling[] = {5, 12280, 13, 1, 49120};
    constexpr unsigned kN = static_cast<unsigned>(sizeof(kDummyTiling) / sizeof(kDummyTiling[0]));
    static_assert(kN >= 1, "host tiling function must return at least 1 result");

    auto origTy = originalKernel_.getFunctionType();

    SmallVector<Type> argTypes;
    argTypes.reserve(origTy.getNumInputs());
    std::copy(origTy.getInputs().begin(), origTy.getInputs().end(), std::back_inserter(argTypes));

    SmallVector<Type> resTypes(kN, builder.getI64Type());

    std::string name = kernelInfo_->baseKernelName + "_single_outlined_0_0_tiling_function";
    auto funcTy = FunctionType::get(builder.getContext(), argTypes, resTypes);
    auto host = builder.create<func::FuncOp>(originalKernel_.getLoc(), name, funcTy);
    host.addEntryBlock();

    auto *ctx = builder.getContext();
    host->setAttr(HACCFuncTypeAttr::name, HACCFuncTypeAttr::get(ctx, HACCFuncType::HOST));
    host->setAttr(HostFuncTypeAttr::name, HostFuncTypeAttr::get(ctx, HostFuncType::kTilingFunction));

    if (std::optional<ArrayAttr> maybeArray = originalKernel_.getArgAttrs()) {
      ArrayAttr arr = *maybeArray;
      unsigned copyN = std::min<unsigned>(arr.size(), host.getNumArguments());

      for (unsigned i = 0; i < copyN; ++i) {
        if (auto dict = arr[i].dyn_cast_or_null<DictionaryAttr>()) {
          SmallVector<NamedAttribute, 4> attrs;
          attrs.reserve(dict.size());
          std::copy(dict.begin(), dict.end(), std::back_inserter(attrs));
          std::for_each(attrs.begin(), attrs.end(), [&](const NamedAttribute &namedAttr) {
            host.setArgAttr(i, namedAttr.getName(), namedAttr.getValue());
          });
        }
      }
    }

    host.setResultAttr(0, KernelArgTypeAttr::name, KernelArgTypeAttr::get(ctx, KernelArgType::kTilingKey));
    for (unsigned i = 1; i < kN; ++i) {
      host.setResultAttr(i, KernelArgTypeAttr::name, KernelArgTypeAttr::get(ctx, KernelArgType::kTilingData));
    }

    builder.setInsertionPointToEnd(&host.getBody().front());
    SmallVector<Value> cst;
    cst.reserve(kN);
    std::transform(std::begin(kDummyTiling), std::end(kDummyTiling), std::back_inserter(cst),
                   [&](int64_t v) { return builder.create<arith::ConstantIntOp>(host.getLoc(), v, 64); });
    builder.create<func::ReturnOp>(host.getLoc(), cst);

    tilingInfo_.setHostTilingFunc(host);
    return success();
  }

  LogicalResult collectDeviceSignature(func::FuncOp orig, SmallVector<Type> &devInputs, SmallVector<Type> &devResults) {
    auto origTy = orig.getFunctionType();

    devInputs.clear();
    devInputs.reserve(origTy.getNumInputs());
    std::copy(origTy.getInputs().begin(), origTy.getInputs().end(), std::back_inserter(devInputs));

    devResults.clear();
    devResults.reserve(origTy.getNumResults());
    std::copy(origTy.getResults().begin(), origTy.getResults().end(), std::back_inserter(devResults));

    return success();
  }

  void copyHaccIOAttrsFrom(func::FuncOp orig, func::FuncOp dst) {
    if (std::optional<ArrayAttr> maybeArray = orig.getArgAttrs()) {
      ArrayAttr arr = *maybeArray;
      unsigned n = std::min<unsigned>(arr.size(), dst.getNumArguments());
      for (unsigned i = 0; i < n; ++i) {
        if (auto dict = arr[i].dyn_cast_or_null<DictionaryAttr>()) {
          SmallVector<NamedAttribute, 4> attrs;
          attrs.reserve(dict.size());
          std::copy(dict.begin(), dict.end(), std::back_inserter(attrs));
          std::for_each(attrs.begin(), attrs.end(), [&](const NamedAttribute &namedAttr) {
            dst.setArgAttr(i, namedAttr.getName(), namedAttr.getValue());
          });
        }
      }
    }
  }

  func::FuncOp createAndAnnotateDeviceFunc(OpBuilder &builder, Location loc, StringRef name, FunctionType devTy,
                                           FunctionType /*origTy*/, unsigned blockDim, func::FuncOp hostTiling) {
    auto deviceFunc = builder.create<func::FuncOp>(loc, name, devTy);

    copyHaccIOAttrsFrom(originalKernel_, deviceFunc);

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

  // ---------------- clone kernel body 到 device ----------------
  SmallVector<Value> cloneKernelBodyToDeviceFunc(func::FuncOp originalKernel, func::FuncOp deviceFunc) {
    SmallVector<Value> returnedValues;
    if (originalKernel.empty()) return returnedValues;

    IRMapping map;
    Block &oldEntry = originalKernel.front();
    Block &newEntry = deviceFunc.getBody().front();

    unsigned numArgs = std::min<unsigned>(oldEntry.getNumArguments(), newEntry.getNumArguments());
    for (unsigned i = 0; i < numArgs; ++i) map.map(oldEntry.getArgument(i), newEntry.getArgument(i));

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

    OpBuilder b = OpBuilder::atBlockEnd(&newEntry);
    std::for_each(toClone.begin(), toClone.end(),
                  [&](Operation *op) {
                    b.clone(*op, map);
                  });

    if (oldRet) {
      unsigned numRet = oldRet.getNumOperands();
      SmallVector<Value> retVals;
      retVals.reserve(numRet);
      returnedValues.resize(numRet);
      for (unsigned i = 0; i < numRet; ++i) {
        Value mapped = map.lookupOrNull(oldRet.getOperand(i));
        returnedValues[i] = mapped;
        if (mapped) retVals.push_back(mapped);
      }
      b.create<func::ReturnOp>(deviceFunc.getLoc(), retVals);
    } else {
      b.create<func::ReturnOp>(deviceFunc.getLoc(), ValueRange{});
    }
    return returnedValues;
  }

  // ---------------- init device kernel ----------------
  LogicalResult initTilingKernel(int64_t key, OpBuilder &builder) {
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPoint(originalKernel_);

    SmallVector<Type> devInputs, devResults;
    if (failed(collectDeviceSignature(originalKernel_, devInputs, devResults))) return failure();

    std::string keyStr = std::to_string(key);
    if (keyStr.size() == 1) {
      keyStr = "0" + keyStr;
    }

    std::string name = kernelInfo_->baseKernelName + "_" + keyStr;

    auto devTy = FunctionType::get(builder.getContext(), devInputs, devResults);
    auto origTy = originalKernel_.getFunctionType();

    auto deviceFunc = createAndAnnotateDeviceFunc(builder, originalKernel_.getLoc(), name, devTy, origTy,
                                                  kernelInfo_->blockDim, tilingInfo_.getHostTilingFunc());

    Block *entry = deviceFunc.addEntryBlock();
    (void)entry;

    (void)cloneKernelBodyToDeviceFunc(originalKernel_, deviceFunc);

    tilingKernel_ = deviceFunc;
    if (failed(createOrGetGetTilingStructSizeFunction(builder, deviceFunc))) return failure();
    return success();
  }

  virtual LogicalResult applyTilingImpl(OpBuilder &) { return success(); }

  LogicalResult fixCallSitesAndCaller(OpBuilder &builder) {
    assert(tilingKernel_);

    auto oldTy = originalKernel_.getFunctionType();

    while (!originalKernel_.getBody().empty()) originalKernel_.getBody().front().erase();

    Block *entry = originalKernel_.addEntryBlock();
    OpBuilder b = OpBuilder::atBlockEnd(entry);
    Location loc = originalKernel_.getLoc();

    SmallVector<Value> args(entry->args_begin(), entry->args_end());
    auto call = b.create<func::CallOp>(loc, tilingKernel_.getSymName(), TypeRange(oldTy.getResults()), args);
    b.create<func::ReturnOp>(loc, call.getResults());

    return success();
  }

  // ---------------- host util: get_tiling_struct_size ----------------
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

struct MemrefTilingFunc : public mlir::impl::MemrefTilingFuncBase<MemrefTilingFunc> {
  MemrefTilingFunc() = default;

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
        k.emitError("memref-based auto-tiling failed");
        signalPassFailure();
        return;
      }
    }
  }
};

}  // namespace
}  // namespace mlir::affine

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> mlir::affine::createMemrefTilingFuncPass() {
  return std::make_unique<MemrefTilingFunc>();
}
