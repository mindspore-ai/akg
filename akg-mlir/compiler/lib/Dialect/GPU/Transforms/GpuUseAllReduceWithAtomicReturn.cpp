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

#include "akg/Dialect/GPU/Transforms/GpuUseAllReduceWithAtomicReturn.h"
#include "akg/Utils/AKGGlobalVars.hpp"
#include "akg/Utils/AnalysisCommon.hpp"
#include "akg/Utils/AnalysisForGpu.hpp"

#include <optional>
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace akgglobal;

namespace mlir {
#define GEN_PASS_DECL_GPUUSEALLREDUCEWITHATOMICRETURN
#define GEN_PASS_DEF_GPUUSEALLREDUCEWITHATOMICRETURN
#include "akg/Dialect/GPU/Passes.h.inc"
}  // namespace mlir

using namespace mlir;

namespace mlir {
namespace gpu {
namespace {

// Convert XXXOp to Atomic kind
std::optional<arith::AtomicRMWKind> ConvertArithOpToAtomicKind(Operation *op) {
  return TypeSwitch<Operation *, std::optional<arith::AtomicRMWKind>>(op)
    .Case([](arith::AddFOp) { return arith::AtomicRMWKind::addf; })
    .Case([](arith::MulFOp) { return arith::AtomicRMWKind::mulf; })
    .Case([](arith::AddIOp) { return arith::AtomicRMWKind::addi; })
    .Case([](arith::AndIOp) { return arith::AtomicRMWKind::andi; })
    .Case([](arith::OrIOp) { return arith::AtomicRMWKind::ori; })
    .Case([](arith::MulIOp) { return arith::AtomicRMWKind::muli; })
    .Case([](arith::MinNumFOp) { return arith::AtomicRMWKind::minimumf; })
    .Case([](arith::MaxNumFOp) { return arith::AtomicRMWKind::maximumf; })
    .Case([](arith::MinSIOp) { return arith::AtomicRMWKind::mins; })
    .Case([](arith::MaxSIOp) { return arith::AtomicRMWKind::maxs; })
    .Case([](arith::MinUIOp) { return arith::AtomicRMWKind::minu; })
    .Case([](arith::MaxUIOp) { return arith::AtomicRMWKind::maxu; })
    .Default([](Operation *) -> std::optional<arith::AtomicRMWKind> { return std::nullopt; });
}

// Convert XXXOp to gpu::AllReduceOperation
std::optional<mlir::gpu::AllReduceOperation> ConvertArithOpToAllReduceOperation(Operation *op) {
  return TypeSwitch<Operation *, std::optional<mlir::gpu::AllReduceOperation>>(op)
    .Case([](arith::AddFOp) { return mlir::gpu::AllReduceOperation::ADD; })
    .Case([](arith::MulFOp) { return mlir::gpu::AllReduceOperation::MUL; })
    .Case([](arith::AddIOp) { return mlir::gpu::AllReduceOperation::ADD; })
    .Case([](arith::AndIOp) { return mlir::gpu::AllReduceOperation::AND; })
    .Case([](arith::OrIOp) { return mlir::gpu::AllReduceOperation::OR; })
    .Case([](arith::MulIOp) { return mlir::gpu::AllReduceOperation::MUL; })
    .Case([](arith::MinNumFOp) { return mlir::gpu::AllReduceOperation::MINIMUMF; })
    .Case([](arith::MaxNumFOp) { return mlir::gpu::AllReduceOperation::MAXIMUMF; })
    .Case([](arith::MinSIOp) { return mlir::gpu::AllReduceOperation::MINSI; })
    .Case([](arith::MaxSIOp) { return mlir::gpu::AllReduceOperation::MAXSI; })
    .Case([](arith::MinUIOp) { return mlir::gpu::AllReduceOperation::MINUI; })
    .Case([](arith::MaxUIOp) { return mlir::gpu::AllReduceOperation::MAXUI; });
}

// Collect information of reduction ops
struct ReduceOpInfo {
  mlir::Operation *op = nullptr;
  bool use_atomic_reduce;
};

// Check whether this kernel is post fusion reduction case. When reduction op is other compute ops'
// post op, it hints that this kernel is post fusion.
static bool IsPostFusion(Operation *redOp) {
  auto curOp = redOp->getNextNode();
  while (curOp) {
    if (!isa<memref::StoreOp>(curOp) && !isa<memref::LoadOp>(curOp) && !isa<memref::AllocOp>(curOp) &&
        !isa<memref::DeallocOp>(curOp) && !isa<scf::YieldOp>(curOp)) {
      SmallVector<Operation *, 8> prevOps;
      SmallVector<mlir::Value, 8> usedValues;
      CommonUtils::getAllPreviousRelatedOpsV2(curOp, prevOps, usedValues);
      for (auto op : prevOps) {
        if (op == redOp) {
          return true;
        }
      }
    }
    curOp = curOp->getNextNode();
  }
  return false;
}

// Match the fixed pattern for reduction stores.
static std::tuple<Operation *, Operation *, Operation *> matchReductionStorePatterns(Operation *redOp) {
  Operation *localStore = nullptr, *localLoad = nullptr, *globalStore = nullptr;
  auto curOp = redOp->getNextNode();
  while (curOp) {
    if (isa<memref::StoreOp>(curOp) && redOp->getResult(0) == curOp->getOperand(0)) {
      localStore = curOp;
    }
    if (localStore && isa<memref::LoadOp>(curOp) && localStore->getOperand(1) == curOp->getOperand(0)) {
      localLoad = curOp;
    }
    if (localLoad && isa<memref::StoreOp>(curOp) && localLoad->getResult(0) == curOp->getOperand(0)) {
      globalStore = curOp;
    }
    curOp = curOp->getNextNode();
  }
  return std::make_tuple(localStore, localLoad, globalStore);
}

// Main enter for reduction-Y rewrite. `ref`s are reference patterns. Emit atomic ops if
// block-level parallel reduction is supported.
static mlir::LogicalResult rewritePatternReduceY(ReduceOpInfo redInfo, OpBuilder &builder) {
  auto loc = redInfo.op->getLoc();
  builder.setInsertionPoint(redInfo.op);

  Operation *localStore = nullptr, *localLoad = nullptr, *globalStore = nullptr;
  std::tie(localStore, localLoad, globalStore) = matchReductionStorePatterns(redInfo.op);
  builder.setInsertionPointAfter(globalStore);

  // Generate `memref.atomic_rmw` op
  std::optional<arith::AtomicRMWKind> atomicKind = ConvertArithOpToAtomicKind(redInfo.op);
  if (atomicKind == std::nullopt) {
    llvm::errs() << "Error: invalid Operation switch to AtomicRMW, please check the type.\n";
    return mlir::failure();
  }
  auto storeOp = dyn_cast<memref::StoreOp>(globalStore);
  auto res_type = redInfo.op->getResultTypes()[0];
  (void)builder.create<memref::AtomicRMWOp>(loc, res_type, *atomicKind, localLoad->getResult(0), storeOp.getMemref(),
                                            storeOp.getIndices());

  if (globalStore) {
    globalStore->erase();
  }
  return mlir::success();
}

// Insert ops to build a statement like: if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
static void insertSingleThreadReduction(OpBuilder &builder, mlir::Location loc, Operation *launchOp,
                                        Operation *globalStore) {
  builder.setInsertionPointAfter(globalStore);
  auto threadArgs = dyn_cast<mlir::gpu::LaunchOp>(launchOp).getThreadIds();
  auto zeroIdx = builder.create<arith::ConstantIndexOp>(loc, 0);
  auto cmpX = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, threadArgs.x, zeroIdx);
  auto cmpY = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, threadArgs.y, zeroIdx);
  auto cmpZ = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, threadArgs.z, zeroIdx);
  auto xAndY = builder.create<arith::AndIOp>(loc, cmpX, cmpY);
  auto xAndYAndZ = builder.create<arith::AndIOp>(loc, xAndY, cmpZ);
  auto ifThreadsZero = builder.create<scf::IfOp>(loc, xAndYAndZ.getResult());
  builder.setInsertionPointToStart(&ifThreadsZero.getThenRegion().front());
}

// Main enter for reduction-X rewrite. Emit thread-level reduction
// with `AllReduceOperation` op. Emit atomic ops if block-level parallel reduction is supported.
static mlir::LogicalResult rewritePatternReduceX(ReduceOpInfo redInfo, OpBuilder &builder, Operation *launchOp) {
  Value operand_b = redInfo.op->getOperands()[0];
  auto res_type = redInfo.op->getResultTypes()[0];

  auto loc = redInfo.op->getLoc();
  mlir::MLIRContext *context = builder.getContext();
  builder.setInsertionPoint(redInfo.op);

  // Generate `gpu.all_reduce  add %X uniform {}`
  mlir::gpu::AllReduceOperation operationType = *ConvertArithOpToAllReduceOperation(redInfo.op);
  mlir::gpu::AllReduceOperationAttr op_attr = mlir::gpu::AllReduceOperationAttr::get(context, operationType);
  Value redValue = builder.create<gpu::AllReduceOp>(loc, res_type, operand_b, op_attr, true);
  auto allreduceOp = redValue.getDefiningOp();

  // In post fusion cases, we have broadcast reduction result to all threads, which
  // gpu.all_reduce has already done. so no need to add scf.if here
  bool isPostFusion = IsPostFusion(redInfo.op);

  Operation *localStore = nullptr, *localLoad = nullptr, *globalStore = nullptr;
  std::tie(localStore, localLoad, globalStore) = matchReductionStorePatterns(redInfo.op);

  if (!isPostFusion) {
    insertSingleThreadReduction(builder, loc, launchOp, globalStore);

    // Generate `memref.atomic_rmw` op
    if (redInfo.use_atomic_reduce) {
      std::optional<arith::AtomicRMWKind> atomicKind = ConvertArithOpToAtomicKind(redInfo.op);
      if (atomicKind == std::nullopt) {
        llvm::errs() << "Error: invalid Operation switch to AtomicRMW, please check the type.\n";
        return mlir::failure();
      }
      memref::StoreOp storeOp = dyn_cast<memref::StoreOp>(localStore);
      if (globalStore) {
        storeOp = dyn_cast<memref::StoreOp>(globalStore);
      }
      (void)builder.create<memref::AtomicRMWOp>(loc, res_type, *atomicKind, redValue, storeOp.getMemref(),
                                                storeOp.getIndices());
    } else {
      memref::StoreOp storeOp = dyn_cast<memref::StoreOp>(localStore);
      if (globalStore) {
        storeOp = dyn_cast<memref::StoreOp>(globalStore);
      }
      (void)builder.create<memref::StoreOp>(loc, redValue, storeOp.getMemref(), storeOp.getIndices());
    }
    if (globalStore) {
      globalStore->erase();
    }
    if (localLoad) {
      localLoad->erase();
    }
    if (localStore) {
      localStore->erase();
    }
  } else {
    redInfo.op->replaceAllUsesWith(allreduceOp);
  }
  redInfo.op->erase();
  return mlir::success();
}

// GpuUseAllReduceWithAtomicReturn rewrite the reduction-related ops to gpu dialect. For example, thread
// level reduction ops rewrite to gpu.allreduce; block-level reduction rewrite to atomic ops. This pass
// supports both reduce-X and reduce-Y algorithms
struct GpuUseAllReduceWithAtomicReturn
    : public impl::GpuUseAllReduceWithAtomicReturnBase<GpuUseAllReduceWithAtomicReturn> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    SmallVector<Operation *, 3> launchOps;
    funcOp.walk([&](Operation *launchOp) {
      if (isa<mlir::gpu::LaunchOp>(launchOp)) {
        launchOps.push_back(launchOp);
      }
    });

    bool isReduceY = GpuScheduleTool::getInstance().getReduceDirection() == (size_t)ReduceDirection::Y;
    for (auto launchOp : launchOps) {
      SmallVector<ReduceOpInfo, 3> redInfos;
      launchOp->walk([&](mlir::Operation *redOp) {
        // op name can be: addf, addi, maxf, maxi, ...
        bool parallelReduce = (redOp->hasAttr(akg::utils::kEnableParallelReduce) &&
                               redOp->getAttrOfType<BoolAttr>(akg::utils::kEnableParallelReduce).getValue());
        bool atomicAdd = (redOp->hasAttr(akg::utils::kEnableAtomicAdd) &&
                          redOp->getAttrOfType<BoolAttr>(akg::utils::kEnableAtomicAdd).getValue());
        if (parallelReduce || (atomicAdd && isReduceY)) {
          ReduceOpInfo redInfo;
          redInfo.use_atomic_reduce = atomicAdd;
          redInfo.op = redOp;
          redInfos.push_back(redInfo);
        }
      });

      for (ReduceOpInfo redInfo : redInfos) {
        OpBuilder builder(launchOp);
        // case 1. reduce-Y
        if (isReduceY && redInfo.use_atomic_reduce) {
          if (mlir::failed(rewritePatternReduceY(redInfo, builder))) {
            signalPassFailure();
          }
        } else {
          // case 2. reduce-X/All
          if (mlir::failed(rewritePatternReduceX(redInfo, builder, launchOp))) {
            signalPassFailure();
          }
        }
      }
    }
  }
};

}  // namespace
}  // namespace gpu
}  // namespace mlir

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> mlir::createGpuUseAllReduceWithAtomicReturnPass() {
  return std::make_unique<gpu::GpuUseAllReduceWithAtomicReturn>();
}
