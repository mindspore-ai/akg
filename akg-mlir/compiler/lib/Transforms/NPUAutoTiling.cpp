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

#include "akg/Transforms/NPUAutoTiling.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "akg/Analysis/LoopTiling.h"
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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

// HACC dialect enums/attrs
#include "bishengir/Dialect/HACC/IR/HACC.h"

#define DEBUG_TYPE "npu-auto-tiling"

namespace mlir {
#define GEN_PASS DECL_NPUAUTOTILING
#define GEN_PASS_DEF_NPUAUTOTILING
#include "akg/Transforms/Passes.h.inc"
}  // namespace mlir

namespace mlir {

namespace mockattr {
static constexpr const char *kEnableAutoMarkBufferSize = "enable_auto_mark_buffer_size";
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

static void setTilingKeyAndDataArgAttrs(func::FuncOp func, unsigned keyIdx, unsigned tilingDataIdx) {
  auto *ctx = func.getContext();
  auto katName = StringAttr::get(ctx, KernelArgTypeAttr::name);

  auto setArgKernelKind = [&](unsigned idx, KernelArgType kind) {
    DictionaryAttr old = func.getArgAttrDict(idx);
    SmallVector<NamedAttribute> nas;
    if (old) {
      nas.reserve(old.size() + 1);
      std::copy(old.begin(), old.end(), std::back_inserter(nas));
    }
    auto kat = KernelArgTypeAttr::get(ctx, kind);

    bool replaced = false;
    for (auto &na : nas) {
      if (na.getName() == katName) {
        na = NamedAttribute(katName, kat);
        replaced = true;
        break;
      }
    }
    if (!replaced) nas.emplace_back(katName, kat);

    func.setArgAttrs(idx, DictionaryAttr::get(ctx, nas));
  };

  setArgKernelKind(keyIdx, KernelArgType::kTilingKey);
  setArgKernelKind(tilingDataIdx, KernelArgType::kTilingStruct);
}

struct TilingInfo {
  func::FuncOp hostTilingFunc;
  DenseMap<int64_t, func::FuncOp> perCaseTilingFuncs;
  DenseMap<int64_t, func::FuncOp> tilingKey2Kernel;
  std::optional<int64_t> constTilingKey;
  bool isStaticShape = false;

  unsigned tilingStructSize = 0;

  void setHostTilingFunc(func::FuncOp f) { hostTilingFunc = f; }
  func::FuncOp getHostTilingFunc() const { return hostTilingFunc; }

  void setPerCaseTilingFunc(int64_t key, func::FuncOp f) { perCaseTilingFuncs[key] = f; }
  func::FuncOp getPerCaseTilingFunc(int64_t key) const {
    auto it = perCaseTilingFuncs.find(key);
    if (it == perCaseTilingFuncs.end()) return func::FuncOp();
    return it->second;
  }

  SmallVector<int64_t> getAllKeys() const {
    SmallVector<int64_t> keys;
    keys.reserve(perCaseTilingFuncs.size());
    std::transform(perCaseTilingFuncs.begin(), perCaseTilingFuncs.end(), std::back_inserter(keys),
                   [](const auto &it) { return it.first; });
    llvm::sort(keys.begin(), keys.end());
    return keys;
  }

  void recordKernelFunc(int64_t key, func::FuncOp func) { tilingKey2Kernel[key] = func; }

  DenseMap<int64_t, func::FuncOp> getTilingKey2KernelMap() const { return tilingKey2Kernel; }

  SmallVector<int64_t> getTilingCases() const { return getAllKeys(); }

  bool analyzeInputShapeStatic(func::FuncOp kernel) {
    for (auto arg : kernel.getArguments()) {
      if (auto memrefType = dyn_cast<MemRefType>(arg.getType())) {
        if (!memrefType.hasStaticShape()) {
          return false;
        }
      }
    }
    return true;
  }

  std::optional<int64_t> computeStaticTilingKey(func::FuncOp kernel) {
    for (auto arg : kernel.getArguments()) {
      if (auto memrefType = dyn_cast<MemRefType>(arg.getType())) {
        if (memrefType.hasStaticShape()) {
          return 0;
        }
      }
    }
    return std::nullopt;
  }

  LogicalResult trySimplifyTilingFunc(func::FuncOp originalKernel) {
    if (!hostTilingFunc) return success();

    isStaticShape = analyzeInputShapeStatic(originalKernel);

    if (isStaticShape) {
      if (auto staticKey = computeStaticTilingKey(originalKernel)) {
        constTilingKey = *staticKey;
        LLVM_DEBUG(llvm::dbgs() << "Static shape detected, tiling key = " << *staticKey << "\n");

        simplifyHostTilingFuncForStaticShape(*staticKey);
        return success();
      }
    }

    return trySimplifyDynamicTilingFunc();
  }

  void simplifyHostTilingFuncForStaticShape(int64_t selectedKey) {
    auto &body = hostTilingFunc.getBody();
    while (!body.empty()) {
      body.front().erase();
    }

    Block *entry = hostTilingFunc.addEntryBlock();
    OpBuilder b(entry, entry->begin());
    Location loc = hostTilingFunc.getLoc();

    unsigned numArgs = hostTilingFunc.getNumArguments();
    unsigned keyIdx = numArgs - 2;
    unsigned tilingDataIdx = numArgs - 1;

    SmallVector<Value> args(entry->args_begin(), entry->args_end());
    Value keyPtr = args[keyIdx];
    Value dataMem = args[tilingDataIdx];

    (void)keyPtr;
    (void)dataMem;

    func::FuncOp perCaseFunc = getPerCaseTilingFunc(selectedKey);
    if (perCaseFunc) {
      SmallVector<Value> callArgs(args.begin(), args.begin() + keyIdx + 2);
      b.create<func::CallOp>(loc, perCaseFunc.getSymName(), TypeRange{}, callArgs);
    }

    b.create<func::ReturnOp>(loc);
  }

  LogicalResult trySimplifyDynamicTilingFunc() {
    if (auto constKey = tryExtractConstantFromHostFunc()) {
      constTilingKey = *constKey;
      LLVM_DEBUG(llvm::dbgs() << "Constant tiling key extracted from host func: " << *constKey << "\n");
      return success();
    }

    return success();
  }

  std::optional<int64_t> tryExtractConstantFromHostFunc() {
    std::optional<int64_t> result;

    hostTilingFunc.walk([&](Operation *op) {
      if (auto returnOp = dyn_cast<func::ReturnOp>(op)) {
        if (returnOp.getNumOperands() > 0) {
          if (auto constVal = extractConstantIntValue(returnOp.getOperand(0))) {
            result = *constVal;
          }
        }
      }

      if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        if (auto constResult = analyzeIfOpForConst(ifOp)) {
          result = *constResult;
        }
      }
    });

    return result;
  }

  std::optional<int64_t> analyzeIfOpForConst(scf::IfOp ifOp) {
    if (ifOp.getNumResults() == 0) return std::nullopt;

    auto thenYield = dyn_cast<scf::YieldOp>(ifOp.thenBlock()->getTerminator());
    auto elseYield = dyn_cast<scf::YieldOp>(ifOp.elseBlock()->getTerminator());

    if (thenYield && elseYield && thenYield.getNumOperands() > 0 && elseYield.getNumOperands() > 0) {
      auto thenConst = extractConstantIntValue(thenYield.getOperand(0));
      auto elseConst = extractConstantIntValue(elseYield.getOperand(0));

      if (thenConst && elseConst && *thenConst == *elseConst) {
        return thenConst;
      }
    }

    return std::nullopt;
  }

  std::optional<int64_t> extractConstantIntValue(Value val) {
    if (auto constOp = val.getDefiningOp<arith::ConstantIntOp>()) {
      auto attr = constOp.getValue();
      if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
        return intAttr.getInt();
      }
    }

    if (auto constOp = val.getDefiningOp<arith::ConstantOp>()) {
      auto attr = constOp.getValue();
      if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
        return intAttr.getInt();
      }
    }

    if (auto constOp = val.getDefiningOp<arith::ConstantIndexOp>()) {
      return constOp.value();
    }

    return std::nullopt;
  }

  bool isTilingFullyStatic() const { return constTilingKey.has_value(); }

  void pruneUnusedTilingFuncs(int64_t selectedKey, ModuleOp module) {
    SmallVector<func::FuncOp> toErase;

    for (auto &it : perCaseTilingFuncs) {
      if (it.first != selectedKey) {
        toErase.push_back(it.second);
      }
    }

    for (auto &it : tilingKey2Kernel) {
      if (it.first != selectedKey) {
        toErase.push_back(it.second);
      }
    }

    for (auto func : toErase) func.erase();

    DenseMap<int64_t, func::FuncOp> newPerCaseMap;
    auto it = perCaseTilingFuncs.find(selectedKey);
    if (it != perCaseTilingFuncs.end()) newPerCaseMap[selectedKey] = it->second;
    perCaseTilingFuncs = std::move(newPerCaseMap);

    DenseMap<int64_t, func::FuncOp> newKernelMap;
    auto kit = tilingKey2Kernel.find(selectedKey);
    if (kit != tilingKey2Kernel.end()) newKernelMap[selectedKey] = kit->second;
    tilingKey2Kernel = std::move(newKernelMap);
  }
};

class TilingBase {
 public:
  explicit TilingBase(func::FuncOp f)
      : originalKernel_(f),
        module_(f ? f->getParentOfType<ModuleOp>() : ModuleOp()),
        kernelInfo_(std::make_unique<KernelInfo>()),
        tilingInfo_() {
    if (kernelInfo_) kernelInfo_->originalKernel = f;
  }

  virtual ~TilingBase() = default;

  LogicalResult runOnOperation(OpBuilder &builder) {
    if (failed(runPreTilingProcedure(builder))) return failure();
    if (failed(runTilingProcedure(builder))) return failure();
    if (failed(runPostTilingProcedure(builder))) return failure();
    if (failed(createOrGetGetTilingStructSizeFunction(builder))) return failure();
    return success();
  }

  static void setAutoTilingOptions(const AutoTilingOptions &opt) { options_ = opt; }

 protected:
  LogicalResult runPreTilingProcedure(OpBuilder &builder) {
    auto *ctx = originalKernel_.getContext();
    ctx->getOrLoadDialect<hacc::HACCDialect>();
    ctx->getOrLoadDialect<scf::SCFDialect>();
    ctx->getOrLoadDialect<arith::ArithDialect>();
    ctx->getOrLoadDialect<memref::MemRefDialect>();
    ctx->getOrLoadDialect<LLVM::LLVMDialect>();

    kernelInfo_->baseKernelName = originalKernel_.getSymName().str();
    kernelInfo_->blockDim = options_.blockDim;

    originalKernel_->setAttr(HACCFuncTypeAttr::name, HACCFuncTypeAttr::get(ctx, HACCFuncType::HOST));
    return success();
  }

  LogicalResult runTilingProcedure(OpBuilder &builder) {
    tilingInfo_.isStaticShape = tilingInfo_.analyzeInputShapeStatic(originalKernel_);

    llvm::DenseMap<int64_t, mlir::func::FuncOp> tilingFuncMap;
    if (failed(mlir::autotiling::createTilingFunctions(originalKernel_, builder, tilingFuncMap,
                                                       tilingInfo_.isStaticShape))) {
      return failure();
    }
    for (const auto &it : tilingFuncMap) {
      int64_t key = it.getFirst();
      mlir::func::FuncOp f = it.getSecond();
      tilingInfo_.setPerCaseTilingFunc(key, f);
    }

    if (failed(computeTilingStructSizeFromTilingFuncs(builder, tilingFuncMap))) return failure();

    if (failed(createHostTilingFunction(builder))) return failure();

    if (failed(tilingInfo_.trySimplifyTilingFunc(originalKernel_))) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to simplify host tiling func\n");
    } else {
      LLVM_DEBUG(llvm::dbgs() << "Successfully simplified host tiling func\n");
    }

    SmallVector<int64_t> tilingKeys;
    if (tilingInfo_.isTilingFullyStatic()) {
      int64_t selectedKey = *tilingInfo_.constTilingKey;
      LLVM_DEBUG(llvm::dbgs() << "Selected tiling key (constant folded): " << selectedKey << "\n");
      tilingKeys.push_back(selectedKey);
    } else {
      tilingKeys = tilingInfo_.getAllKeys();
    }

    if (tilingKeys.empty()) {
      originalKernel_.emitError("no per-case tiling functions recorded in TilingInfo");
      return failure();
    }

    bool allOk = std::all_of(tilingKeys.begin(), tilingKeys.end(),
                             [&](int64_t key) { return succeeded(initTilingKernel(key, builder)); });
    if (!allOk) return failure();

    if (tilingInfo_.isTilingFullyStatic()) {
      tilingInfo_.pruneUnusedTilingFuncs(*tilingInfo_.constTilingKey, module_);
    }

    if (failed(applyTilingImpl(builder))) return failure();
    if (failed(fixCallSitesAndCaller(builder))) return failure();
    return success();
  }

  LogicalResult computeTilingStructSizeFromTilingFuncs(
    OpBuilder &builder, const llvm::DenseMap<int64_t, mlir::func::FuncOp> &tilingFuncMap) {
    auto it0 = tilingFuncMap.find(0);
    if (it0 == tilingFuncMap.end()) {
      originalKernel_.emitError("tilingFuncMap does not contain key=0");
      return failure();
    }

    func::FuncOp tilingFunc0 = it0->second;

    auto *ctx = builder.getContext();
    auto katName = StringAttr::get(ctx, KernelArgTypeAttr::name);

    unsigned structSize = 0;

    for (auto [idx, arg] : llvm::enumerate(tilingFunc0.getArguments())) {
      DictionaryAttr dict = tilingFunc0.getArgAttrDict(idx);
      if (!dict) continue;

      Attribute attr = dict.get(katName);
      if (!attr) continue;

      auto katAttr = dyn_cast<KernelArgTypeAttr>(attr);
      if (!katAttr) continue;

      if (katAttr.getArgType() == KernelArgType::kTilingStruct) {
        auto memrefTy = dyn_cast<MemRefType>(arg.getType());
        if (!memrefTy || memrefTy.getRank() != 1 || !memrefTy.getElementType().isInteger(64)) {
          originalKernel_.emitError("tiling struct argument must be rank-1 memref<i64>");
          return failure();
        }

        int64_t dim0 = memrefTy.getDimSize(0);
        if (dim0 <= 0) {
          originalKernel_.emitError("tiling struct size must be a positive static dimension");
          return failure();
        }

        structSize = static_cast<unsigned>(dim0);
        break;
      }
    }

    if (structSize == 0) {
      structSize = 1;
    }

    tilingInfo_.tilingStructSize = structSize;
    return success();
  }

  LogicalResult runPostTilingProcedure(OpBuilder &) { return success(); }

  void copyHaccIOAttrsFrom(func::FuncOp orig, func::FuncOp dst) {
    if (std::optional<ArrayAttr> maybeArray = orig.getArgAttrs()) {
      ArrayAttr arr = *maybeArray;
      unsigned n = std::min<unsigned>(arr.size(), dst.getNumArguments());
      for (unsigned i = 0; i < n; ++i) {
        if (auto dict = dyn_cast_or_null<DictionaryAttr>(arr[i])) {
          SmallVector<NamedAttribute, 4> attrs;
          attrs.reserve(dict.size());
          std::copy(dict.begin(), dict.end(), std::back_inserter(attrs));
          for (const auto &na : attrs) dst.setArgAttr(i, na.getName(), na.getValue());
        }
      }
    }
  }

  void copyAttrsForDeviceFromHost(func::FuncOp host, func::FuncOp device) {
    for (auto &na : host->getAttrs()) {
      auto name = na.getName().strref();

      if (name == "hacc.function_kind" || name == "hacc.host_func_type" || name == "hacc.tiling_function" ||
          name == "mindspore_kernel") {
        continue;
      }

      if (name == "OperatorType" || name == "compute_capability" || name == "process") {
        device->setAttr(na.getName(), na.getValue());
      }
    }
  }

  func::FuncOp createHostTilingFuncOpHeader(OpBuilder &builder, MLIRContext *ctx, Location loc,
                                            ArrayRef<Type> argTypes) {
    auto llvmPtrTy = LLVM::LLVMPointerType::get(ctx);
    auto i64Ty = builder.getI64Type();

    unsigned sz = tilingInfo_.tilingStructSize;
    if (sz == 0) sz = 1;
    auto memrefTy = MemRefType::get({static_cast<int64_t>(sz)}, i64Ty);

    SmallVector<Type> fullArgs(argTypes.begin(), argTypes.end());
    fullArgs.push_back(llvmPtrTy);
    fullArgs.push_back(memrefTy);

    SmallVector<Type> resTypes;

    std::string name = kernelInfo_->baseKernelName + "_tiling_function";
    auto funcTy = FunctionType::get(ctx, fullArgs, resTypes);
    auto host = builder.create<func::FuncOp>(loc, name, funcTy);
    host.addEntryBlock();

    copyHaccIOAttrsFrom(originalKernel_, host);

    host->setAttr(HACCFuncTypeAttr::name, HACCFuncTypeAttr::get(ctx, HACCFuncType::HOST));
    host->setAttr(HostFuncTypeAttr::name, HostFuncTypeAttr::get(ctx, HostFuncType::kTilingFunction));

    unsigned numArgs = host.getNumArguments();
    unsigned keyIdx = numArgs - 2;
    unsigned tilingDataIdx = numArgs - 1;
    setTilingKeyAndDataArgAttrs(host, keyIdx, tilingDataIdx);

    return host;
  }

  Value selectKeyByDimension(OpBuilder &b, Location loc, ArrayRef<Value> args) {
    auto k0 = b.create<arith::ConstantIntOp>(loc, 0, 64);  // i64
    return k0;
  }

  LogicalResult createHostTilingFunction(OpBuilder &builder) {
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPoint(originalKernel_);

    auto *ctx = builder.getContext();
    auto loc = originalKernel_.getLoc();
    auto origTy = originalKernel_.getFunctionType();

    SmallVector<Type> argTypes(origTy.getInputs().begin(), origTy.getInputs().end());

    func::FuncOp host = createHostTilingFuncOpHeader(builder, ctx, loc, argTypes);
    if (!host) return failure();

    OpBuilder bodyBuilder(&host.getBody().front(), host.getBody().front().end());
    SmallVector<Value> args(host.getArguments().begin(), host.getArguments().end());

    unsigned numArgs = args.size();
    if (numArgs < 2) {
      host.emitError("host tiling function expects at least 2 tail args");
      return failure();
    }
    unsigned keyIdx = numArgs - 2;
    unsigned tilingDataIdx = numArgs - 1;

    SmallVector<Value> computeArgs(args.begin(), args.begin() + keyIdx);
    Value keyPtr = args[keyIdx];
    Value dataMem = args[tilingDataIdx];

    Value logicalKeyI64 = selectKeyByDimension(bodyBuilder, loc, computeArgs);
    Value keyIndex = bodyBuilder.create<arith::IndexCastUIOp>(loc, bodyBuilder.getIndexType(), logicalKeyI64);

    SmallVector<int64_t> caseKeys = tilingInfo_.getAllKeys();
    if (caseKeys.empty()) {
      host.emitError("no tiling cases recorded in TilingInfo for host tiling function");
      return failure();
    }

    auto switchOp = bodyBuilder.create<scf::IndexSwitchOp>(loc, TypeRange{}, keyIndex, ArrayRef<int64_t>(caseKeys),
                                                           /*numCases=*/caseKeys.size());

    for (unsigned i = 0; i < caseKeys.size(); ++i) {
      int64_t key = caseKeys[i];
      Region &reg = switchOp.getCaseRegions()[i];
      Block *blk = new Block();
      reg.push_back(blk);
      OpBuilder cb(blk, blk->begin());

      func::FuncOp perCaseF = tilingInfo_.getPerCaseTilingFunc(key);
      if (!perCaseF) {
        host.emitError() << "missing per-case tiling function for key=" << key;
        return failure();
      }

      SmallVector<Value> callArgs;
      callArgs.append(computeArgs.begin(), computeArgs.end());
      callArgs.push_back(keyPtr);
      callArgs.push_back(dataMem);

      cb.create<func::CallOp>(loc, perCaseF.getSymName(), perCaseF.getFunctionType().getResults(), callArgs);
      cb.create<scf::YieldOp>(loc);
    }

    {
      Region &defaultReg = switchOp.getDefaultRegion();
      Block *blk = new Block();
      defaultReg.push_back(blk);
      OpBuilder db(blk, blk->begin());

      Value falseVal = db.create<arith::ConstantIntOp>(loc, 0, db.getI1Type());
      auto msgAttr = db.getStringAttr("Invalid tiling key");

      db.create<cf::AssertOp>(loc, falseVal, msgAttr);

      db.create<scf::YieldOp>(loc);
    }

    bodyBuilder.create<func::ReturnOp>(loc);
    tilingInfo_.setHostTilingFunc(host);
    return success();
  }

  LogicalResult collectDeviceSignature(func::FuncOp orig, SmallVector<Type> &devInputs, SmallVector<Type> &devResults,
                                       int64_t /*key*/) {
    auto origTy = orig.getFunctionType();

    devInputs.clear();
    devInputs.reserve(origTy.getNumInputs());
    std::copy(origTy.getInputs().begin(), origTy.getInputs().end(), std::back_inserter(devInputs));

    devResults.clear();
    devResults.reserve(origTy.getNumResults());
    std::copy(origTy.getResults().begin(), origTy.getResults().end(), std::back_inserter(devResults));

    if (tilingInfo_.isTilingFullyStatic()) return success();

    auto *ctx = orig.getContext();
    auto i64Ty = IntegerType::get(ctx, 64);
    auto llvmPtrTy = LLVM::LLVMPointerType::get(ctx);

    unsigned sz = tilingInfo_.tilingStructSize;
    if (sz == 0) sz = 1;
    auto memrefTy = MemRefType::get({static_cast<int64_t>(sz)}, i64Ty);

    devInputs.push_back(llvmPtrTy);  // tiling_key: !llvm.ptr
    devInputs.push_back(memrefTy);   // tiling_struct: memref<Nxi64>

    return success();
  }

  func::FuncOp createAndAnnotateDeviceFunc(OpBuilder &builder, Location loc, StringRef name, FunctionType devTy,
                                           FunctionType /*origTy*/, unsigned blockDim, func::FuncOp hostTiling) {
    auto deviceFunc = builder.create<func::FuncOp>(loc, name, devTy);
    deviceFunc.addEntryBlock();

    copyHaccIOAttrsFrom(originalKernel_, deviceFunc);

    copyAttrsForDeviceFromHost(originalKernel_, deviceFunc);

    deviceFunc->setAttr(mockattr::kEnableAutoMarkBufferSize, builder.getUnitAttr());
    deviceFunc->setAttr(HACCFuncTypeAttr::name, HACCFuncTypeAttr::get(builder.getContext(), HACCFuncType::DEVICE));
    deviceFunc->setAttr(BlockDimAttr::name, builder.getI64IntegerAttr(blockDim));
    deviceFunc->setAttr(
      StringAttr::get(builder.getContext(), stringifyHACCToLLVMIRTranslateAttr(HACCToLLVMIRTranslateAttr::ENTRY)),
      builder.getUnitAttr());
    if (hostTiling) {
      deviceFunc->setAttr(
        TilingFunctionAttr::name,
        TilingFunctionAttr::get(builder.getContext(), FlatSymbolRefAttr::get(hostTiling.getSymNameAttr())));
    }

    if (!tilingInfo_.isTilingFullyStatic()) {
      unsigned numInputs = devTy.getNumInputs();
      unsigned keyIdx = numInputs - 2;
      unsigned tilingDataIdx = numInputs - 1;
      setTilingKeyAndDataArgAttrs(deviceFunc, keyIdx, tilingDataIdx);
    }

    return deviceFunc;
  }

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
    for (Operation *op : toClone) b.clone(*op, map);

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

  LogicalResult initTilingKernel(int64_t key, OpBuilder &builder) {
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPoint(originalKernel_);

    SmallVector<Type> devInputs, devResults;
    if (failed(collectDeviceSignature(originalKernel_, devInputs, devResults, key))) return failure();

    std::string keyStr = std::to_string(key);
    if (keyStr.size() == 1) keyStr = "0" + keyStr;

    std::string name = kernelInfo_->baseKernelName + "_" + keyStr;

    auto devTy = FunctionType::get(builder.getContext(), devInputs, devResults);
    auto origTy = originalKernel_.getFunctionType();

    auto deviceFunc = createAndAnnotateDeviceFunc(builder, originalKernel_.getLoc(), name, devTy, origTy,
                                                  kernelInfo_->blockDim, tilingInfo_.getPerCaseTilingFunc(key));

    (void)cloneKernelBodyToDeviceFunc(originalKernel_, deviceFunc);

    tilingInfo_.recordKernelFunc(key, deviceFunc);
    return success();
  }

  LogicalResult applyTilingImpl(OpBuilder &builder) {
    for (const auto &it : tilingInfo_.getTilingKey2KernelMap()) {
      int64_t key = it.getFirst();
      mlir::func::FuncOp f = it.getSecond();
      if (failed(mlir::autotiling::applyTilingFromTilingFunc(f, builder, tilingInfo_.isStaticShape))) return failure();
      tilingInfo_.recordKernelFunc(key, f);
    }
    return success();
  }

  LogicalResult fixCallSitesAndCaller(OpBuilder &builder) {
    auto *ctx = builder.getContext();
    auto oldTy = originalKernel_.getFunctionType();
    unsigned oldNumInputs = oldTy.getNumInputs();

    if (tilingInfo_.isTilingFullyStatic()) {
      return rewriteOriginalKernelBodyStatic(builder, ctx, oldTy, oldNumInputs);
    }

    auto i64Ty = builder.getI64Type();
    auto llvmPtrTy = LLVM::LLVMPointerType::get(ctx);

    unsigned sz = tilingInfo_.tilingStructSize;
    if (sz == 0) sz = 1;
    auto tilingdataTy = MemRefType::get({static_cast<int64_t>(sz)}, i64Ty);

    SmallVector<Type> newInputs(oldTy.getInputs().begin(), oldTy.getInputs().end());
    newInputs.push_back(llvmPtrTy);     // tiling_key as !llvm.ptr
    newInputs.push_back(tilingdataTy);  // tiling_struct: memref<Nxi64>

    auto newTy = FunctionType::get(ctx, newInputs, oldTy.getResults());
    originalKernel_.setType(newTy);

    if (failed(updateOriginalKernelArgAttrs(oldTy, oldNumInputs))) return failure();

    return rewriteOriginalKernelBodyDynamic(builder, ctx, oldTy, oldNumInputs);
  }

  LogicalResult updateOriginalKernelArgAttrs(FunctionType oldTy, unsigned oldNumInputs) {
    (void)oldTy;
    std::optional<ArrayAttr> maybeArr = originalKernel_.getArgAttrs();
    if (maybeArr) {
      ArrayAttr oldArr = *maybeArr;
      unsigned copyNum = std::min<unsigned>(oldArr.size(), oldNumInputs);
      for (unsigned i = 0; i < copyNum; ++i) {
        if (auto dict = dyn_cast_or_null<DictionaryAttr>(oldArr[i])) {
          originalKernel_.setArgAttrs(i, dict);
        }
      }
    }

    unsigned keyIdx = oldNumInputs;
    unsigned memrefIdx = oldNumInputs + 1;
    setTilingKeyAndDataArgAttrs(originalKernel_, keyIdx, memrefIdx);
    return success();
  }

  LogicalResult rewriteOriginalKernelBodyStatic(OpBuilder &builder, MLIRContext *ctx, FunctionType oldTy,
                                                unsigned oldNumInputs) {
    while (!originalKernel_.getBody().empty()) originalKernel_.getBody().front().erase();

    Block *entry = originalKernel_.addEntryBlock();
    OpBuilder b = OpBuilder::atBlockEnd(entry);
    Location loc = originalKernel_.getLoc();

    SmallVector<Value> args(entry->args_begin(), entry->args_end());
    if (args.size() != oldNumInputs) {
      originalKernel_.emitError("static tiling: unexpected number of arguments");
      return failure();
    }

    if (!tilingInfo_.isTilingFullyStatic() || !tilingInfo_.constTilingKey.has_value()) {
      originalKernel_.emitError("static tiling: constTilingKey not available");
      return failure();
    }

    int64_t constKey = *tilingInfo_.constTilingKey;
    std::string keyStr = std::to_string(constKey);
    if (keyStr.size() == 1) keyStr = "0" + keyStr;
    std::string devName = kernelInfo_->baseKernelName + "_" + keyStr;

    auto sym = SymbolTable::lookupSymbolIn(module_, StringAttr::get(ctx, devName));
    if (!sym) {
      originalKernel_.emitError() << "cannot find device kernel " << devName << " for static tiling";
      return failure();
    }
    auto devFunc = dyn_cast<func::FuncOp>(sym);
    if (!devFunc) {
      originalKernel_.emitError() << devName << " is not func.func (static tiling)";
      return failure();
    }

    auto call = b.create<func::CallOp>(loc, devFunc.getSymName(), TypeRange(oldTy.getResults()), args);
    b.create<func::ReturnOp>(loc, call.getResults());
    return success();
  }

  LogicalResult rewriteOriginalKernelBodyDynamic(OpBuilder &builder, MLIRContext *ctx, FunctionType oldTy,
                                                 unsigned oldNumInputs) {
    while (!originalKernel_.getBody().empty()) originalKernel_.getBody().front().erase();

    Block *entry = originalKernel_.addEntryBlock();
    OpBuilder b = OpBuilder::atBlockEnd(entry);
    Location loc = originalKernel_.getLoc();

    SmallVector<Value> args(entry->args_begin(), entry->args_end());
    Value keyPtr = args[oldNumInputs];       // !llvm.ptr
    Value dataMem = args[oldNumInputs + 1];  // memref<Nxi64>

    auto i64Ty = b.getI64Type();
    auto indexTy = b.getIndexType();
    Value keyI64 = b.create<LLVM::LoadOp>(loc, i64Ty, keyPtr);
    Value keyIndex = b.create<arith::IndexCastUIOp>(loc, indexTy, keyI64);

    SmallVector<int64_t> caseKeys = tilingInfo_.getAllKeys();
    if (caseKeys.empty()) {
      originalKernel_.emitError("no tiling cases recorded in TilingInfo for original kernel switch");
      return failure();
    }

    auto switchOp = b.create<scf::IndexSwitchOp>(loc, TypeRange(oldTy.getResults()), keyIndex,
                                                 ArrayRef<int64_t>(caseKeys), caseKeys.size());

    for (unsigned i = 0; i < caseKeys.size(); ++i) {
      int64_t key = caseKeys[i];

      Region &reg = switchOp.getCaseRegions()[i];
      Block *blk = new Block();
      reg.push_back(blk);
      OpBuilder cb(blk, blk->begin());

      SmallVector<Value> callArgs;
      callArgs.reserve(oldNumInputs + 2);
      for (unsigned a = 0; a < oldNumInputs; ++a) callArgs.push_back(args[a]);
      callArgs.push_back(keyPtr);
      callArgs.push_back(dataMem);

      std::string keyStr = std::to_string(key);
      if (keyStr.size() == 1) keyStr = "0" + keyStr;
      std::string devName = kernelInfo_->baseKernelName + "_" + keyStr;

      auto sym = SymbolTable::lookupSymbolIn(module_, StringAttr::get(ctx, devName));
      if (!sym) {
        originalKernel_.emitError() << "cannot find device kernel " << devName;
        return failure();
      }
      auto devFunc = dyn_cast<func::FuncOp>(sym);
      if (!devFunc) {
        originalKernel_.emitError() << devName << " is not func.func";
        return failure();
      }

      auto call = cb.create<func::CallOp>(loc, devFunc.getSymName(), TypeRange(oldTy.getResults()), callArgs);
      cb.create<scf::YieldOp>(loc, ValueRange(call.getResults()));
    }

    {
      Region &defaultReg = switchOp.getDefaultRegion();
      Block *blk = new Block();
      defaultReg.push_back(blk);
      OpBuilder db(blk, blk->begin());

      Value falseVal = db.create<arith::ConstantIntOp>(loc, 0, db.getI1Type());
      auto msgAttr = db.getStringAttr("Invalid tiling key");
      db.create<mlir::cf::AssertOp>(loc, falseVal, msgAttr);

      SmallVector<Value> outputArgs;

      auto maybeArgAttrArray = originalKernel_.getArgAttrs();
      ArrayAttr argAttrArray;
      if (maybeArgAttrArray) argAttrArray = *maybeArgAttrArray;

      auto katName = StringAttr::get(ctx, KernelArgTypeAttr::name);

      for (BlockArgument arg : entry->getArguments()) {
        unsigned argIdx = arg.getArgNumber();

        if (!argAttrArray || argIdx >= argAttrArray.size()) continue;

        auto dict = dyn_cast_or_null<DictionaryAttr>(argAttrArray[argIdx]);
        if (!dict) continue;

        Attribute attr = dict.get(katName);
        if (!attr) continue;

        if (auto katAttr = dyn_cast<KernelArgTypeAttr>(attr)) {
          if (katAttr.getArgType() == KernelArgType::kOutput) {
            outputArgs.push_back(arg);
          }
        }
      }

      SmallVector<Value> defaultResults;
      defaultResults.reserve(oldTy.getNumResults());

      for (unsigned i = 0, e = oldTy.getNumResults(); i < e; ++i) {
        defaultResults.push_back(outputArgs[i]);
      }

      db.create<scf::YieldOp>(loc, defaultResults);
    }

    b.create<func::ReturnOp>(loc, switchOp.getResults());
    return success();
  }

  LogicalResult createOrGetGetTilingStructSizeFunction(OpBuilder &builder) {
    ModuleOp module = originalKernel_->getParentOfType<ModuleOp>();
    if (!module) {
      originalKernel_.emitError("cannot find parent ModuleOp for original kernel");
      return failure();
    }

    std::string base = (kernelInfo_ ? kernelInfo_->baseKernelName : originalKernel_.getSymName().str());

    std::string hostName = base + "_get_tiling_struct_size_function";
    if (auto sym = SymbolTable::lookupSymbolIn(module, StringAttr::get(module.getContext(), hostName))) {
      if (isa<func::FuncOp>(sym)) return success();
    }

    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToStart(module.getBody());
    auto funcTy = FunctionType::get(module.getContext(), TypeRange{}, TypeRange{builder.getI64Type()});
    auto host = builder.create<func::FuncOp>(originalKernel_.getLoc(), hostName, funcTy);
    host.setVisibility(SymbolTable::Visibility::Public);

    auto *ctx = builder.getContext();
    host->setAttr(HACCFuncTypeAttr::name, HACCFuncTypeAttr::get(ctx, HACCFuncType::HOST));
    host->setAttr(HostFuncTypeAttr::name, HostFuncTypeAttr::get(ctx, HostFuncType::kGetTilingStructSizeFunction));

    Block *entry = host.addEntryBlock();
    OpBuilder b = OpBuilder::atBlockEnd(entry);

    int64_t sz = static_cast<int64_t>(tilingInfo_.tilingStructSize);
    if (sz <= 0) sz = 1;
    auto sizeConst = b.create<arith::ConstantIntOp>(host.getLoc(), sz, 64);
    b.create<func::ReturnOp>(host.getLoc(), ValueRange{sizeConst});
    return success();
  }

 protected:
  func::FuncOp originalKernel_;
  ModuleOp module_;
  std::unique_ptr<KernelInfo> kernelInfo_;
  TilingInfo tilingInfo_;
  static AutoTilingOptions options_;
};

AutoTilingOptions TilingBase::options_;

class PureElemwiseTiling : public TilingBase {
 public:
  using TilingBase::TilingBase;
  // LogicalResult applyTilingImpl(OpBuilder &) override { return success(); }
};

struct NPUAutoTiling : public mlir::impl::NPUAutoTilingBase<NPUAutoTiling> {
  NPUAutoTiling() = default;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (!module) {
      signalPassFailure();
      return;
    }

    SmallVector<func::FuncOp> kernels;
    module.walk([&](func::FuncOp f) {
      auto kind = f->getAttrOfType<StringAttr>(HACCFuncTypeAttr::name);
      if (kind && kind.getValue() == "DEVICE") {
        return;
      }
      kernels.push_back(f);
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
}  // namespace mlir

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> mlir::createNPUAutoTilingPass() {
  return std::make_unique<NPUAutoTiling>();
}
