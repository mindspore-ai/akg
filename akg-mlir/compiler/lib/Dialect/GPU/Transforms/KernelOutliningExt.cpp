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

#include "akg/Dialect/GPU/Transforms/GpuKernelOutliningExt.h"
#include "akg/Utils/AKGGlobalVars.hpp"
#include "akg/Utils/AnalysisForGpu.hpp"
#include "akg/Utils/IOHelper.hpp"

#include <limits>
#include <nlohmann/json.hpp>
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/GPU/Transforms/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/RegionUtils.h"

namespace mlir {
#define GEN_PASS_DECL_GPUKERNELOUTLININGEXT
#define GEN_PASS_DEF_GPUKERNELOUTLININGEXT
#include "akg/Dialect/GPU/Passes.h.inc"
}  // namespace mlir

using namespace mlir;
using namespace akgglobal;
using namespace mlir::akg::utils;

constexpr auto kVectorInitSize8 = 8;
constexpr auto kVectorInitSize4 = 4;

template <typename OpTy>
static void createForAllDimensions(OpBuilder &builder, const Location loc, SmallVectorImpl<Value> &values) {
  for (auto dim : {gpu::Dimension::x, gpu::Dimension::y, gpu::Dimension::z}) {
    values.push_back(builder.create<OpTy>(loc, builder.getIndexType(), dim));
  }
}

/// Adds operations generating block/thread ids and grid/block dimensions at the
/// beginning of the `launchFuncOpBody` region. Add mapping from argument in
/// entry block of `launchOpBody`, to the corresponding result value of the
/// added operations.
static void injectGpuIndexOperations(Location loc, Region &launchFuncOpBody, Region &launchOpBody, IRMapping &map) {
  OpBuilder builder(loc->getContext());
  Block &firstBlock = launchOpBody.front();
  builder.setInsertionPointToStart(&launchFuncOpBody.front());
  SmallVector<Value, 12> indexOps;
  createForAllDimensions<gpu::BlockIdOp>(builder, loc, indexOps);
  createForAllDimensions<gpu::ThreadIdOp>(builder, loc, indexOps);
  createForAllDimensions<gpu::GridDimOp>(builder, loc, indexOps);
  createForAllDimensions<gpu::BlockDimOp>(builder, loc, indexOps);
  // Replace the leading 12 function args with the respective thread/block index
  // operations. Iterate backwards since args are erased and indices change.
  for (const auto &indexOp : enumerate(indexOps)) {
    map.map(firstBlock.getArgument(indexOp.index()), indexOp.value());
  }
}

static bool idxIsInVector(size_t funcIdx, SmallVector<int, kVectorInitSize8> &mapResult) {
  return std::any_of(mapResult.begin(), mapResult.end(),
                     [funcIdx](int idx) { return idx == static_cast<int>(funcIdx); });
}

static void initOperandOrder(func::FuncOp funcOp, SetVector<Value> &operands,
                             SmallVector<int, kVectorInitSize8> &mapResult) {
  auto funcArguments = funcOp.getArguments();
  for (size_t idx = 0; idx < operands.size(); idx++) {
    for (size_t funcIdx = 0; funcIdx < funcArguments.size(); funcIdx++) {
      if (funcArguments[funcIdx] != operands[idx]) {
        continue;
      }
      mapResult[idx] = funcIdx;
      break;
    }
  }
}

static void getAdditionalOperandOrder(func::FuncOp funcOp, SetVector<Value> &operands,
                                      SmallVector<int, kVectorInitSize8> &mapResult,
                                      std::map<int, int> &additionalArgs) {
  auto funcArguments = funcOp.getArguments();
  for (size_t idx = 0; idx < operands.size(); idx++) {
    if (idxIsInVector(idx, mapResult)) {
      continue;
    }
    auto op = operands[idx];
    if (!op.getDefiningOp()) {
      continue;
    }
    if (!isa<mlir::arith::SubIOp>(op.getDefiningOp())) {
      continue;
    }
    auto sub = dyn_cast<mlir::arith::SubIOp>(op.getDefiningOp());
    auto rhs = sub.getRhs();
    for (size_t funcIdx = 0; funcIdx < funcArguments.size(); funcIdx++) {
      if (funcArguments[funcIdx] != rhs) {
        continue;
      }
      additionalArgs[funcIdx] = idx;
      break;
    }
  }
}

static void reviseProperOperandOrder(gpu::LaunchOp launchOp, SetVector<Value> &operands) {
  if (auto funcOp = launchOp->getParentOfType<func::FuncOp>()) {
    auto funcArguments = funcOp.getArguments();
    SmallVector<int, kVectorInitSize8> mapResult(operands.size(), -1);
    initOperandOrder(funcOp, operands, mapResult);

    for (size_t funcIdx = 0; funcIdx < funcArguments.size(); funcIdx++) {
      if (idxIsInVector(funcIdx, mapResult)) {
        continue;
      }
      mlir::Value v = Value();
      // try to match patterns for mapping from operand to argument
      GpuCommonUtils::findAllocOpForFuncArg(v, funcOp, funcArguments[funcIdx]);
      GpuCommonUtils::findExpandShapeOpForFuncArg(v, funcOp, funcArguments[funcIdx]);
      // cannot find any operand match to func arguments
      // scenarios 1: lack of pattern match, need to add more;
      // scenarios 2: this operand is from temp buffer, which may erase by promote-temp-buffer pass
      if (!v) {
        continue;
      }

      for (size_t idx = 0; idx < operands.size(); idx++) {
        if (v != operands[idx]) {
          continue;
        }
        mapResult[idx] = static_cast<int>(funcIdx);
        break;
      }
    }

    std::map<int, int> additionalArgs;
    getAdditionalOperandOrder(funcOp, operands, mapResult, additionalArgs);
    SmallVector<Value, kVectorInitSize8> tmpOperands(funcArguments);
    for (size_t i = 0; i < operands.size(); i++) {
      if (mapResult[i] >= 0) {
        tmpOperands[mapResult[i]] = operands[i];
      } else if (mapResult[i] == -1 && additionalArgs.find(mapResult[i]) == additionalArgs.end()) {
        tmpOperands.push_back(operands[i]);
      }
    }

    operands.clear();
    for (size_t idx = 0; idx < tmpOperands.size(); idx++) {
      (void)operands.insert(tmpOperands[idx]);
    }
  }
}

/// Outline the `gpu.launch` operation body into a kernel function. Replace
/// `gpu.terminator` operations by `gpu.return` in the generated function.
/// Set block and grid size bounds if known.
static gpu::GPUFuncOp outlineKernelFuncImpl(gpu::LaunchOp launchOp, StringRef kernelFnName,
                                            SetVector<Value> &operands) {
  Location loc = launchOp.getLoc();
  // Create a builder with no insertion point, insertion will happen separately
  // due to symbol table manipulation.
  OpBuilder builder(launchOp.getContext());
  Region &launchOpBody = launchOp.getBody();

  // Identify uses from values defined outside of the scope of the launch
  // operation.
  getUsedValuesDefinedAbove(launchOpBody, operands);

  reviseProperOperandOrder(launchOp, operands);

  // Create the gpu.func operation.
  SmallVector<Type, kVectorInitSize4> kernelOperandTypes;
  kernelOperandTypes.reserve(operands.size());
  for (Value operand : operands) {
    kernelOperandTypes.push_back(operand.getType());
  }
  FunctionType type = FunctionType::get(launchOp.getContext(), kernelOperandTypes, {});
  auto outlinedFunc = builder.create<gpu::GPUFuncOp>(loc, kernelFnName, type);
  outlinedFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), builder.getUnitAttr());

  IRMapping map;

  // Map the arguments corresponding to the launch parameters like blockIdx,
  // threadIdx, etc.
  Region &outlinedFuncBody = outlinedFunc.getBody();
  injectGpuIndexOperations(loc, outlinedFuncBody, launchOpBody, map);

  // Map arguments from gpu.launch region to the arguments of the gpu.func
  // operation.
  Block &entryBlock = outlinedFuncBody.front();
  for (const auto &operand : enumerate(operands)) {
    map.map(operand.value(), entryBlock.getArgument(operand.index()));
  }

  // Clone the region of the gpu.launch operation into the gpu.func operation.
  launchOpBody.cloneInto(&outlinedFuncBody, map);

  // Branch from entry of the gpu.func operation to the block that is cloned
  // from the entry block of the gpu.launch operation.
  Block &launchOpEntry = launchOpBody.front();
  Block *clonedLaunchOpEntry = map.lookup(&launchOpEntry);
  builder.setInsertionPointToEnd(&entryBlock);
  (void)builder.create<cf::BranchOp>(loc, clonedLaunchOpEntry);

  outlinedFunc.walk([](gpu::TerminatorOp op) {
    OpBuilder replacer(op);
    (void)replacer.create<gpu::ReturnOp>(op.getLoc());
    op.erase();
  });
  return outlinedFunc;
}

/// Replace `gpu.launch` operations with an `gpu.launch_func` operation
/// launching `kernelFunc`. The kernel func contains the body of the
/// `gpu.launch` with constant region arguments inlined.
static void convertToLaunchFuncOp(gpu::LaunchOp launchOp, gpu::GPUFuncOp kernelFunc, ValueRange operands) {
  OpBuilder builder(launchOp);
  // The launch op has an optional dynamic shared memory size. If it doesn't
  // exist, we use zero.
  Value asyncToken = launchOp.getAsyncToken();
  auto launchFunc = builder.create<gpu::LaunchFuncOp>(
    launchOp.getLoc(), kernelFunc, launchOp.getGridSizeOperandValues(), launchOp.getBlockSizeOperandValues(),
    launchOp.getDynamicSharedMemorySize(), operands, asyncToken ? asyncToken.getType() : nullptr,
    launchOp.getAsyncDependencies());
  launchOp.replaceAllUsesWith(launchFunc);
  launchOp.erase();
}

namespace {
/// Pass that moves ops which are likely an index computation into gpu.launch
/// body.

/// Pass that moves the kernel of each LaunchOp into its separate nested module.
///
/// This pass moves the kernel code of each LaunchOp into a function created
/// inside a nested module. It also creates an external function of the same
/// name in the parent module.
///
/// The gpu.modules are intended to be compiled to a cubin blob independently in
/// a separate pass. The external functions can then be annotated with the
/// symbol of the cubin accessor function.
class GpuKernelOutliningExt : public impl::GpuKernelOutliningExtBase<GpuKernelOutliningExt> {
 public:
  explicit GpuKernelOutliningExt(StringRef dlStr) {
    if (!dlStr.empty() && !dataLayoutStr.hasValue()) {
      dataLayoutStr = dlStr.str();
    }
  }

  GpuKernelOutliningExt(const GpuKernelOutliningExt &other)
      : GpuKernelOutliningExtBase(other), dataLayoutSpec(other.dataLayoutSpec) {
    dataLayoutStr = other.dataLayoutStr.getValue();
  }

  LogicalResult initialize(MLIRContext *context) override {
    // Initialize the data layout specification from the data layout string.
    if (!dataLayoutStr.empty()) {
      Attribute resultAttr = mlir::parseAttribute(dataLayoutStr, context);
      if (!resultAttr) {
        return failure();
      }

      dataLayoutSpec = dyn_cast<DataLayoutSpecInterface>(resultAttr);
      if (!dataLayoutSpec) {
        return failure();
      }
    }

    return success();
  }

  void doShapeAlign() {
    func::FuncOp mainFunc;
    getOperation()->walk([&](func::FuncOp op) { mainFunc = op; });
    ShapeAlignTool &tool = ShapeAlignTool::getInstance();
    auto mainFuncArgSizes = tool.getFuncArgSizes();
    if (!mainFunc || mainFuncArgSizes == 0) {
      return;
    }
    SmallVector<Value> gpuArgs;
    getOperation()->walk([&](gpu::LaunchFuncOp funcOp) {
      auto operands = funcOp.getKernelOperands();
      for (size_t i = 0; i < operands.size(); i++) {
        if (i >= mainFuncArgSizes) {
          continue;
        }
        auto operand = operands[i];
        gpuArgs.push_back(operand);
      }
    });

    for (size_t argIdx = 0; argIdx < tool.getFuncArgSizes(); ++argIdx) {
      Value mainArg = mainFunc.getBody().front().getArgument(argIdx);
      // We already match the order of args between main func and gpu func so we can directly use argIdx here.
      auto gpuArg = gpuArgs[argIdx];
      auto currShape = tool.getCurrShapeInfo(argIdx);
      if (tool.isOutput(argIdx)) {
        for (auto user : mainArg.getUsers()) {
          tool.alignOutputShape(user, gpuArg, currShape, mainFunc.getOperation());
        }
      } else {
        for (auto user : mainArg.getUsers()) {
          tool.alignInputShape(user, gpuArg, currShape);
        }
      }
      tool.updateCurrShapeInfo(argIdx, currShape);
    }
  }

  void RecordStaticShapeArgs() {
    std::vector<std::vector<int>> shapeArgs;
    size_t mainFuncSize = 0;
    getOperation()->walk([&](func::FuncOp func) { mainFuncSize = func.getBody().front().getArguments().size(); });
    getOperation()->walk([&](gpu::LaunchFuncOp funcOp) {
      auto operands = funcOp.getKernelOperands();
      for (size_t i = 0; i < mainFuncSize; i++) {
        mlir::MemRefType memrefType = cast<mlir::MemRefType>(operands[i].getType());
        int64_t offset;
        SmallVector<int64_t> strides;
        if (failed(getStridesAndOffset(memrefType, strides, offset)))
            return;
        std::vector<int> shapeArg;
        shapeArg.push_back(offset);
        for (auto s : memrefType.getShape()) {
          shapeArg.push_back(s);
        }
        for (auto s : strides) {
          shapeArg.push_back(s);
        }
        shapeArgs.push_back(shapeArg);
      }
    });

    nlohmann::json j = shapeArgs;
    std::string kernelName = "akg_kernel";
    (void)getOperation()->walk([&](func::FuncOp func) {
      if (func->hasAttr("mindspore_kernel")) {
        kernelName = func.getName().str();
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    (void)DirUtils::CheckOrCreateDirectory("./akg_kernel_meta/");
    std::string output_filename = "./akg_kernel_meta/" + kernelName + "_shape_arg.txt";
    if (llvm::writeToOutput(output_filename, [&](llvm::raw_ostream &OS) -> llvm::Error {
          OS << j.dump();
          return llvm::Error::success();
        })) {
      llvm::errs() << "Write json file to " << output_filename << " failed.\n";
    }
  }

  void runOnOperation() override {
    SymbolTable symbolTable(getOperation());
    bool modified = false;
    for (auto func : getOperation().getOps<func::FuncOp>()) {
      // Insert just after the function.
      Block::iterator insertPt(func->getNextNode());
      auto funcWalkResult = func.walk([&](gpu::LaunchOp op) {
        SetVector<Value> operands;
        std::string kernelFnName = Twine(op->getParentOfType<func::FuncOp>().getName(), "_kernel").str();

        gpu::GPUFuncOp outlinedFunc = outlineKernelFuncImpl(op, kernelFnName, operands);

        // Create nested module and insert outlinedFunc. The module will
        // originally get the same name as the function, but may be renamed on
        // insertion into the parent module.
        auto kernelModule = createKernelModule(outlinedFunc, symbolTable);
        (void)symbolTable.insert(kernelModule, insertPt);

        // Potentially changes signature, pulling in constants.
        convertToLaunchFuncOp(op, outlinedFunc, operands.getArrayRef());
        modified = true;
        return WalkResult::advance();
      });
      if (funcWalkResult.wasInterrupted()) {
        return signalPassFailure();
      }
    }

    // If any new module was inserted in this module, annotate this module as
    // a container module.
    if (modified) {
      getOperation()->setAttr(gpu::GPUDialect::getContainerModuleAttrName(), UnitAttr::get(&getContext()));
    }

    bool isDynamicShape = akgglobal::ShapeAlignTool::getInstance().getFuncArgSizes() > 0;
    if (isDynamicShape) {
      doShapeAlign();
    } else {
      RecordStaticShapeArgs();
    }
  }

 private:
  /// Returns a gpu.module containing kernelFunc and all callees (recursive).
  gpu::GPUModuleOp createKernelModule(gpu::GPUFuncOp kernelFunc, const SymbolTable &parentSymbolTable) {
    auto *context = getOperation().getContext();
    OpBuilder builder(context);
    auto kernelModule = builder.create<gpu::GPUModuleOp>(kernelFunc.getLoc(), kernelFunc.getName());

    // If a valid data layout spec was provided, attach it to the kernel module.
    // Otherwise, the default data layout will be used.
    if (dataLayoutSpec) {
      kernelModule->setAttr(DLTIDialect::kDataLayoutAttrName, dataLayoutSpec);
    }

    SymbolTable symbolTable(kernelModule);
    (void)symbolTable.insert(kernelFunc);

    SmallVector<Operation *, kVectorInitSize8> symbolDefWorklist = {kernelFunc};
    while (!symbolDefWorklist.empty()) {
      if (std::optional<SymbolTable::UseRange> symbolUses =
            SymbolTable::getSymbolUses(symbolDefWorklist.pop_back_val())) {
        for (SymbolTable::SymbolUse symbolUse : *symbolUses) {
          StringRef symbolName = cast<FlatSymbolRefAttr>(symbolUse.getSymbolRef()).getValue();
          if (symbolTable.lookup(symbolName)) {
            continue;
          }

          Operation *symbolDefClone = parentSymbolTable.lookup(symbolName)->clone();
          symbolDefWorklist.push_back(symbolDefClone);
          (void)symbolTable.insert(symbolDefClone);
        }
      }
    }

    return kernelModule;
  }

  Option<std::string> dataLayoutStr{*this, "data-layout-str",
                                    llvm::cl::desc("String containing the data layout specification to be "
                                                   "attached to the GPU kernel module")};

  DataLayoutSpecInterface dataLayoutSpec;
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::createGpuKernelOutliningExt(StringRef dataLayoutStr) {
  return std::make_unique<GpuKernelOutliningExt>(dataLayoutStr);
}
