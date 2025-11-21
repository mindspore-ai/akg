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

#include <memory>
#include <string>
#include <utility>
#include <iostream>
#include <algorithm>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
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
  [[maybe_unused]] bool enableManageHostResources = false;
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

    if (options_.enableManageHostResources) {
      (void)0;
    }
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
    constexpr unsigned kN = sizeof(kDummyTiling) / sizeof(kDummyTiling[0]);

    auto origTy = originalKernel_.getFunctionType();
    if (origTy.getNumResults() != 1) {
      originalKernel_.emitError() << "expect exactly 1 result before rewriting";
      return failure();
    }

    SmallVector<Type> argTypes(origTy.getInputs().begin(), origTy.getInputs().end());
    Type outTy = origTy.getResult(0);
    argTypes.push_back(outTy);

    SmallVector<Type> resTypes(kN, builder.getI64Type());

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
    for (unsigned i = 1; i < kN; ++i)
      host.setResultAttr(i, "hacc.arg_type", StringAttr::get(builder.getContext(), "tiling_data"));

    builder.setInsertionPointToEnd(&host.getBody().front());

    SmallVector<Value> cst;
    cst.reserve(kN);
    std::transform(std::begin(kDummyTiling), std::end(kDummyTiling), std::back_inserter(cst),
                   [&](int64_t v) { return builder.create<arith::ConstantIntOp>(host.getLoc(), v, 64); });

    builder.create<func::ReturnOp>(host.getLoc(), cst);

    tilingInfo_.setHostTilingFunc(host);
    return success();
  }

  LogicalResult initTilingKernel(OpBuilder &builder) {
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPoint(originalKernel_);

    auto origTy = originalKernel_.getFunctionType();
    if (origTy.getNumResults() != 1) {
      originalKernel_.emitError() << "expect exactly 1 result";
      return failure();
    }

    SmallVector<Type> devInputs(origTy.getInputs().begin(), origTy.getInputs().end());
    Type outTy = origTy.getResult(0);
    devInputs.push_back(outTy);

    SmallVector<Type> devResults;
    std::string name = kernelInfo_->baseKernelName + "_single_outlined_0_0_0";
    auto devTy = FunctionType::get(builder.getContext(), devInputs, devResults);
    auto deviceFunc = builder.create<func::FuncOp>(originalKernel_.getLoc(), name, devTy);

    unsigned nInputs = origTy.getNumInputs();
    for (unsigned i = 0; i < nInputs; ++i) {
      deviceFunc.setArgAttr(i, "hacc.arg_type", StringAttr::get(builder.getContext(), "input"));
      deviceFunc.setArgAttr(i, "hacc.input_idx", builder.getI64IntegerAttr(i));
    }
    deviceFunc.setArgAttr(nInputs, "hacc.arg_type", StringAttr::get(builder.getContext(), "output"));
    deviceFunc.setArgAttr(nInputs, "hacc.output_idx", builder.getI64IntegerAttr(0));

    deviceFunc->setAttr(mockattr::kEnableAutoMarkBufferSize, builder.getUnitAttr());
    deviceFunc->setAttr(mockattr::kFunctionKind, StringAttr::get(builder.getContext(), mockattr::kDevice));
    deviceFunc->setAttr(mockattr::kFusionKind,
                        StringAttr::get(builder.getContext(), mockattr::kFusionKindPureElemwise));
    deviceFunc->setAttr(mockattr::kBlockDim, builder.getI64IntegerAttr(kernelInfo_->blockDim));
    deviceFunc->setAttr("hacc.entry", builder.getUnitAttr());
    if (auto hostTiling = tilingInfo_.getHostTilingFunc())
      deviceFunc->setAttr(mockattr::kTilingFunction, FlatSymbolRefAttr::get(hostTiling.getSymNameAttr()));

    Block *entry = deviceFunc.addEntryBlock();
    OpBuilder b = OpBuilder::atBlockEnd(entry);
    Location loc = deviceFunc.getLoc();

    SmallVector<Value> inArgs;
    inArgs.reserve(nInputs);
    for (unsigned i = 0; i < nInputs; ++i) inArgs.push_back(entry->getArgument(i));
    Value outArg = entry->getArgument(nInputs);

    Value returned;
    {
      IRMapping map;
      if (!originalKernel_.empty()) {
        Block &oldEntry = originalKernel_.front();
        unsigned argToMap = std::min<unsigned>(oldEntry.getNumArguments(), nInputs);
        for (unsigned i = 0; i < argToMap; ++i) map.map(oldEntry.getArgument(i), inArgs[i]);
        if (oldEntry.getNumArguments() > nInputs) map.map(oldEntry.getArgument(nInputs), outArg);

        func::ReturnOp oldRet = nullptr;
        SmallVector<Operation *> toClone;
        for (Operation &op : oldEntry) {
          if (auto r = dyn_cast<func::ReturnOp>(op)) {
            oldRet = r;
            continue;
          }
          toClone.push_back(&op);
        }
        for (Operation *op : toClone) b.clone(*op, map);

        if (oldRet && oldRet.getNumOperands() == 1) returned = map.lookupOrDefault(oldRet.getOperand(0));
      }
    }
    if (!returned) returned = outArg;

    if (returned != outArg) b.create<memref::CopyOp>(loc, returned, outArg);

    b.create<func::ReturnOp>(loc);

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

    for (unsigned i = 0; i < nInputs; ++i) {
      originalKernel_.setArgAttr(i, "hacc.arg_type", StringAttr::get(builder.getContext(), "input"));
      originalKernel_.setArgAttr(i, "hacc.input_idx", builder.getI64IntegerAttr(i));
    }
    originalKernel_.setArgAttr(nInputs, "hacc.arg_type", StringAttr::get(builder.getContext(), "output"));
    originalKernel_.setArgAttr(nInputs, "hacc.output_idx", builder.getI64IntegerAttr(0));

    while (!originalKernel_.getBody().empty()) originalKernel_.getBody().front().erase();

    Block *entry = originalKernel_.addEntryBlock();
    OpBuilder b = OpBuilder::atBlockEnd(entry);
    Location loc = originalKernel_.getLoc();

    SmallVector<Value> passArgs(entry->args_begin(), entry->args_end());
    b.create<func::CallOp>(loc, tilingKernel_.getSymName(), TypeRange{}, passArgs);

    Value outArg = entry->getArgument(nInputs);
    b.create<func::ReturnOp>(loc, outArg);

    auto hostTiling = tilingInfo_.getHostTilingFunc();
    SmallVector<func::CallOp> callers;
    module_.walk([&](func::CallOp c) {
      if (c.getCallee() == kernelInfo_->baseKernelName) callers.push_back(c);
    });

    for (func::CallOp callOp : callers) {
      OpBuilder::InsertionGuard gg(builder);
      builder.setInsertionPoint(callOp);

      SmallVector<Value> operands(callOp.getOperands().begin(), callOp.getOperands().end());
      auto outMemRefTy = dyn_cast<MemRefType>(outTy);
      if (!outMemRefTy) {
        callOp.emitError() << "expect memref out";
        return failure();
      }
      Value outVal = builder.create<memref::AllocOp>(callOp.getLoc(), outMemRefTy);

      SmallVector<Value> tilArgs(operands);
      tilArgs.push_back(outVal);
      auto tilResTys = hostTiling.getFunctionType().getResults();
      auto tilCall = builder.create<func::CallOp>(callOp.getLoc(), hostTiling.getSymName(), tilResTys, tilArgs);

      SmallVector<Value> newCallArgs(operands);
      newCallArgs.push_back(outVal);
      newCallArgs.append(tilCall.getResults().begin(), tilCall.getResults().end());

      auto newCall = builder.create<func::CallOp>(callOp.getLoc(), originalKernel_.getSymName(),
                                                  originalKernel_.getFunctionType().getResults(), newCallArgs);

      callOp.replaceAllUsesWith(newCall.getResults());
      callOp.erase();
    }

    return success();
  }

  LogicalResult createOrGetGetTilingStructSizeFunction(OpBuilder &builder, func::FuncOp deviceFunc) {
    ModuleOp module = deviceFunc->getParentOfType<ModuleOp>();
    if (!module) {
      deviceFunc.emitError() << "cannot find parent ModuleOp for device function";
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

    auto funcTy =
      FunctionType::get(module.getContext(), /*inputs=*/TypeRange{}, /*results=*/TypeRange{builder.getI64Type()});
    auto host = builder.create<func::FuncOp>(deviceFunc.getLoc(), hostName, funcTy);
    host.setVisibility(SymbolTable::Visibility::Public);

    host->setAttr(mockattr::kFunctionKind, StringAttr::get(builder.getContext(), mockattr::kHost));

    Block *entry = host.addEntryBlock();
    OpBuilder b = OpBuilder::atBlockEnd(entry);
    auto zero = b.create<arith::ConstantIntOp>(host.getLoc(), /*value=*/0, /*width=*/64);
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
          !kind || kind.getValue() == mockattr::kDevice)
        kernels.push_back(f);
    });

    AutoTilingOptions opts;
    TilingBase::setAutoTilingOptions(opts);

    OpBuilder builder(module.getContext());
    for (func::FuncOp k : kernels) {
      PureElemwiseTiling sch(k);
      if (failed(sch.runOnOperation(builder))) {
        k.emitError() << "auto-tiling failed";
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
