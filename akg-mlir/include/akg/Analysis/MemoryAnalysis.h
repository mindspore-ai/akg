/**
 * Copyright 2026 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#ifndef COMPILER_INCLUDE_AKG_ANALYSIS_MEMORY_ANALYSIS_H_
#define COMPILER_INCLUDE_AKG_ANALYSIS_MEMORY_ANALYSIS_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir {

//  peak estimation (implemented in MemoryAnalysis.cpp).
struct PeakAnalysisInput {
  func::FuncOp func;
  llvm::DenseMap<scf::ForOp, int64_t> tileUpperBoundPerLoop;
  llvm::DenseMap<scf::ForOp, bool> isReduceXorAllVectorizeLoop;
  bool enableMultibuffer = true;
  bool alignBufferSizeTo256Bits = true;
};

struct PeakAnalysisResult {
  int64_t PeakBits{-1};
  bool valid{false};
};

void estimatePeakForTiling(const PeakAnalysisInput &input, PeakAnalysisResult &out);

void setTileUpperBoundForLoop(PeakAnalysisInput &input, scf::ForOp forOp, int64_t tileUpperBound);

void fillTileUpperBoundsByWalkOrder(PeakAnalysisInput &input, llvm::ArrayRef<int64_t> bounds);

// Debug: set each `scf.for` tile bound to its full static trip count (no splitting). Skips
// loops whose lb/ub/step are not constant integers.
void fillTileUpperBoundPerLoopWithFullAxisExtents(PeakAnalysisInput &input);

// Unit tile for all loops, run  peak analysis, print and return peak bits.
int64_t estimateAndPrintPeakWithUnitTileSize(func::FuncOp func);

}  // namespace mlir

#endif  // COMPILER_INCLUDE_AKG_ANALYSIS_MEMORY_ANALYSIS_H_
