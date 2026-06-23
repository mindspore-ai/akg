/**
 * Copyright 2023-2026 Huawei Technologies Co., Ltd
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

#include <limits>
#include <nlohmann/json.hpp>
#include "akg/Dialect/GPU/Transforms/GpuKernelOutliningExt.h"
#include "akg/Utils/GlobalVars.hpp"
#include "akg/Utils/AnalysisForGpu.hpp"
#include "akg/Utils/IOHelper.hpp"
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
#include "akg/Utils/SmallVectorSize.h"

namespace mlir {
#define GEN_PASS_DECL_GPUKERNELOUTLININGEXT
#define GEN_PASS_DEF_GPUKERNELOUTLININGEXT
#include "akg/Dialect/GPU/Passes.h.inc"

}  // namespace mlir

namespace {
using mlir::Attribute;
using mlir::Block;
using mlir::cast;
using mlir::DataLayoutSpecInterface;
using mlir::DLTIDialect;
using mlir::dyn_cast;
using mlir::failed;
using mlir::failure;
using mlir::FlatSymbolRefAttr;
using mlir::FunctionType;
using mlir::getStridesAndOffset;
using mlir::getUsedValuesDefinedAbove;
using mlir::IOHelper;
using mlir::IRMapping;
using mlir::isa;
using mlir::kSmallVectorSizeTwelve;
using mlir::Location;
using mlir::LogicalResult;
using mlir::MemRefType;
using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::OpBuilder;
using mlir::Operation;
using mlir::OperationPass;
using mlir::Region;
using mlir::SetVector;
using mlir::SmallVector;
using mlir::SmallVectorImpl;
using mlir::StringRef;
using mlir::success;
using mlir::SymbolTable;
using mlir::Twine;
using mlir::Type;
using mlir::UnitAttr;
using mlir::Value;
using mlir::ValueRange;
using mlir::WalkResult;
constexpr auto kVectorInitSize8 = 8;
constexpr auto kVectorInitSize4 = 4;

template <typename OpTy>
static void createForAllDimensions(OpBuilder &builder, const Location loc, SmallVectorImpl<Value> &values) {
  static const mlir::gpu::Dimension allDims[] = {mlir::gpu::Dimension::x, mlir::gpu::Dimension::y,
                                                 mlir::gpu::Dimension::z};
  std::transform(
    std::begin(allDims), std::end(allDims), std::back_inserter(values),
    [&builder, &loc](mlir::gpu::Dimension dim) { return builder.create<OpTy>(loc, builder.getIndexType(), dim); });
}

/// Adds operations generating block/thread ids and grid/block dimensions at the
/// beginning of the `launchFuncOpBody` region. Add mapping from argument in
/// entry block of `launchOpBody`, to the corresponding result value of the
/// added operations.
static void injectGpuIndexOperations(Location loc, Region &launchFuncOpBody, Region &launchOpBody, IRMapping &map) {
  OpBuilder builder(loc->getContext());
  Block &firstBlock = launchOpBody.front();
  builder.setInsertionPointToStart(&launchFuncOpBody.front());
  SmallVector<Value, kSmallVectorSizeTwelve> indexOps;
  createForAllDimensions<mlir::gpu::BlockIdOp>(builder, loc, indexOps);
  createForAllDimensions<mlir::gpu::ThreadIdOp>(builder, loc, indexOps);
  createForAllDimensions<mlir::gpu::GridDimOp>(builder, loc, indexOps);
  createForAllDimensions<mlir::gpu::BlockDimOp>(builder, loc, indexOps);
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

static void initOperandOrder(mlir::func::FuncOp funcOp, SetVector<Value> &operands,
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

static void getAdditionalOperandOrder(mlir::func::FuncOp funcOp, SetVector<Value> &operands,
                                      SmallVector<int, kVectorInitSize8> &mapResult,
                                      std::map<int, int> &additionalArgs) {
  auto funcArguments = funcOp.getArguments();
  for (size_t idx = 0; idx < operands.size(); idx++) {
    if (idxIsInVector(idx, mapResult)) {
      continue;
    }
    auto op = operands[idx];
    if (op.getDefiningOp() == nullptr) {
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

static void matchOperandIndex(Value v, size_t funcIdx, SetVector<Value> &operands,
                              SmallVector<int, kVectorInitSize8> &mapResult) {
  for (size_t idx = 0; idx < operands.size(); idx++) {
    if (v != operands[idx]) {
      continue;
    }
    mapResult[idx] = static_cast<int>(funcIdx);
    break;
  }
}

static void reviseProperOperandOrder(mlir::gpu::LaunchOp launchOp, SetVector<Value> &operands) {
  auto funcOp = launchOp->getParentOfType<mlir::func::FuncOp>();
  if (!funcOp) {
    return;
  }
  auto funcArguments = funcOp.getArguments();
  SmallVector<int, kVectorInitSize8> mapResult(operands.size(), -1);
  initOperandOrder(funcOp, operands, mapResult);

  for (size_t funcIdx = 0; funcIdx < funcArguments.size(); funcIdx++) {
    if (idxIsInVector(funcIdx, mapResult)) {
      continue;
    }
    mlir::Value v = Value();
    mlir::akg::utils::GpuCommonUtils::findAllocOpForFuncArg(v, funcOp, funcArguments[funcIdx]);
    mlir::akg::utils::GpuCommonUtils::findExpandShapeOpForFuncArg(v, funcOp, funcArguments[funcIdx]);
    if (!v) {
      continue;
    }

    matchOperandIndex(v, funcIdx, operands, mapResult);
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
  for (auto tmpOperand : tmpOperands) {
    (void)operands.insert(tmpOperand);
  }
}

/// Outline the `gpu.launch` operation body into a kernel function. Replace
/// `gpu.terminator` operations by `gpu.return` in the generated function.
/// Set block and grid size bounds if known.
static mlir::gpu::GPUFuncOp outlineKernelFuncImpl(mlir::gpu::LaunchOp launchOp, StringRef kernelFnName,
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
  std::transform(operands.begin(), operands.end(), std::back_inserter(kernelOperandTypes),
                 [](Value operand) { return operand.getType(); });
  FunctionType type = FunctionType::get(launchOp.getContext(), kernelOperandTypes, {});
  auto outlinedFunc = builder.create<mlir::gpu::GPUFuncOp>(loc, kernelFnName, type);
  outlinedFunc->setAttr(mlir::gpu::GPUDialect::getKernelFuncAttrName(), builder.getUnitAttr());

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
  (void)builder.create<mlir::cf::BranchOp>(loc, clonedLaunchOpEntry);

  outlinedFunc.walk([](mlir::gpu::TerminatorOp op) {
    OpBuilder replacer(op);
    (void)replacer.create<mlir::gpu::ReturnOp>(op.getLoc());
    op.erase();
  });
  return outlinedFunc;
}

/// Replace `gpu.launch` operations with an `gpu.launch_func` operation
/// launching `kernelFunc`. The kernel func contains the body of the
/// `gpu.launch` with constant region arguments inlined.
static void convertToLaunchFuncOp(mlir::gpu::LaunchOp launchOp, mlir::gpu::GPUFuncOp kernelFunc, ValueRange operands) {
  OpBuilder builder(launchOp);
  // The launch op has an optional dynamic shared memory size. If it doesn't
  // exist, we use zero.
  Value asyncToken = launchOp.getAsyncToken();
  auto launchFunc = builder.create<mlir::gpu::LaunchFuncOp>(
    launchOp.getLoc(), kernelFunc, launchOp.getGridSizeOperandValues(), launchOp.getBlockSizeOperandValues(),
    launchOp.getDynamicSharedMemorySize(), operands, asyncToken ? asyncToken.getType() : nullptr,
    launchOp.getAsyncDependencies());
  launchOp.replaceAllUsesWith(launchFunc);
  launchOp.erase();
}
}  // namespace

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
class KernelOutliningExt : public mlir::impl::GpuKernelOutliningExtBase<KernelOutliningExt> {
 public:
  explicit KernelOutliningExt(StringRef dlStr) {
    if (!dlStr.empty() && !dataLayoutStr.hasValue()) {
      dataLayoutStr = dlStr.str();
    }
  }

  KernelOutliningExt(const KernelOutliningExt &other)
      : GpuKernelOutliningExtBase(other), dataLayoutSpec(other.dataLayoutSpec) {
    // cppcheck-suppress useInitializationList
    dataLayoutStr = other.dataLayoutStr.getValue();
  }

  KernelOutliningExt &operator=(const KernelOutliningExt &other) {
    if (this != &other) {
      dataLayoutSpec = other.dataLayoutSpec;
      dataLayoutStr = other.dataLayoutStr.getValue();
    }
    return *this;
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
    mlir::func::FuncOp mainFunc = [this]() {
      mlir::func::FuncOp result;
      getOperation()->walk([&result](mlir::func::FuncOp op) { result = op; });
      return result;
    }();
    akgglobal::ShapeAlignTool &tool = akgglobal::ShapeAlignTool::getInstance();
    auto mainFuncArgSizes = tool.getFuncArgSizes();
    if (!mainFunc || mainFuncArgSizes == 0) {
      return;
    }
    SmallVector<Value> gpuArgs;
    getOperation()->walk([&gpuArgs, &mainFuncArgSizes](mlir::gpu::LaunchFuncOp funcOp) {
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
    getOperation()->walk(
      [&mainFuncSize](mlir::func::FuncOp func) { mainFuncSize = func.getBody().front().getArguments().size(); });
    getOperation()->walk([&mainFuncSize, &shapeArgs](mlir::gpu::LaunchFuncOp funcOp) {
      auto operands = funcOp.getKernelOperands();
      for (size_t i = 0; i < mainFuncSize; i++) {
        mlir::MemRefType memrefType = cast<mlir::MemRefType>(operands[i].getType());
        int64_t offset;
        SmallVector<int64_t> strides;
        if (failed(getStridesAndOffset(memrefType, strides, offset))) {
          return;
        }
        std::vector<int> shapeArg;
        shapeArg.push_back(offset);
        std::copy(memrefType.getShape().begin(), memrefType.getShape().end(), std::back_inserter(shapeArg));
        std::copy(strides.begin(), strides.end(), std::back_inserter(shapeArg));
        shapeArgs.push_back(shapeArg);
      }
    });

    nlohmann::json j = shapeArgs;
    std::string kernelName = "akg_kernel";
    (void)getOperation()->walk([&kernelName](mlir::func::FuncOp func) {
      if (func->hasAttr("mindspore_kernel")) {
        kernelName = func.getName().str();
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    (void)IOHelper::CheckOrCreateDirectory("./akg_kernel_meta/");
    std::string output_filename = "./akg_kernel_meta/" + kernelName + "_shape_arg.txt";
    if (llvm::writeToOutput(output_filename, [&j](llvm::raw_ostream &OS) -> llvm::Error {
          OS << j.dump();
          return llvm::Error::success();
        })) {
      llvm::errs() << "Write json file to " << output_filename << " failed.\n";
    }
  }

  void runOnOperation() override {
    SymbolTable symbolTable(getOperation());
    bool modified = false;
    for (auto func : getOperation().getOps<mlir::func::FuncOp>()) {
      // Insert just after the function.
      Block::iterator insertPt(func->getNextNode());
      auto funcWalkResult = func.walk([this, &symbolTable, &insertPt, &modified](mlir::gpu::LaunchOp op) {
        SetVector<Value> operands;
        std::string kernelFnName = Twine(op->getParentOfType<mlir::func::FuncOp>().getName(), "_kernel").str();

        mlir::gpu::GPUFuncOp outlinedFunc = outlineKernelFuncImpl(op, kernelFnName, operands);

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
      getOperation()->setAttr(mlir::gpu::GPUDialect::getContainerModuleAttrName(), UnitAttr::get(&getContext()));
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
  mlir::gpu::GPUModuleOp createKernelModule(mlir::gpu::GPUFuncOp kernelFunc, const SymbolTable &parentSymbolTable) {
    auto *context = getOperation().getContext();
    OpBuilder builder(context);
    auto kernelModule = builder.create<mlir::gpu::GPUModuleOp>(kernelFunc.getLoc(), kernelFunc.getName());

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
          if (symbolTable.lookup(symbolName) != nullptr) {
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

namespace mlir {
std::unique_ptr<OperationPass<ModuleOp>> createGpuKernelOutliningExt(StringRef dataLayoutStr) {
  return std::make_unique<KernelOutliningExt>(dataLayoutStr);
}
}  // namespace mlir
