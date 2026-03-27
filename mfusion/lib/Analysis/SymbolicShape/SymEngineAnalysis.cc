/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include "mfusion/Analysis/SymbolicShape/SymEngineAnalysis.h"

#include "symengine/integer.h"

namespace mfusion {
mlir::FailureOr<SymEngineAnalysis::SymExpr> SymEngineAnalysis::getOrAssignSymbol(mlir::Value value,
                                                                                 const SymbolNameResolver &resolver) {
  auto it = valueToSymMap_.find(value);
  if (it != valueToSymMap_.end()) {
    return it->second;
  }

  auto nameOrFailure = resolver(value);
  if (mlir::failed(nameOrFailure)) {
    return mlir::failure();
  }

  auto sym = builder_.makeSymbol(*nameOrFailure);
  valueToSymMap_.try_emplace(value, sym);
  return sym;
}

mlir::FailureOr<SymEngineAnalysis::SymExpr> SymEngineAnalysis::convertAffineExpr(
  mlir::AffineExpr expr, llvm::ArrayRef<SymExpr> dimSymbols, llvm::ArrayRef<SymExpr> symbolSymbols) const {
  if (auto constantExpr = mlir::dyn_cast<mlir::AffineConstantExpr>(expr)) {
    return builder_.makeInteger(constantExpr.getValue());
  }

  if (auto dimExpr = mlir::dyn_cast<mlir::AffineDimExpr>(expr)) {
    unsigned position = dimExpr.getPosition();
    if (position >= dimSymbols.size()) {
      return mlir::failure();
    }
    return dimSymbols[position];
  }

  if (auto symExpr = mlir::dyn_cast<mlir::AffineSymbolExpr>(expr)) {
    unsigned position = symExpr.getPosition();
    if (position >= symbolSymbols.size()) {
      return mlir::failure();
    }
    return symbolSymbols[position];
  }

  if (auto binExpr = mlir::dyn_cast<mlir::AffineBinaryOpExpr>(expr)) {
    auto lhs = convertAffineExpr(binExpr.getLHS(), dimSymbols, symbolSymbols);
    auto rhs = convertAffineExpr(binExpr.getRHS(), dimSymbols, symbolSymbols);
    if (mlir::failed(lhs) || mlir::failed(rhs)) {
      return mlir::failure();
    }

    switch (binExpr.getKind()) {
      case mlir::AffineExprKind::Add:
        return builder_.makeAdd(*lhs, *rhs);
      case mlir::AffineExprKind::Mul:
        return builder_.makeMul(*lhs, *rhs);
      case mlir::AffineExprKind::FloorDiv:
        return builder_.makeDiv(*lhs, *rhs);
      case mlir::AffineExprKind::CeilDiv: {
        auto divExpr = builder_.makeDiv(*lhs, *rhs);
        return builder_.makeCeil(divExpr);
      }
      default:
        return mlir::failure();
    }
  }

  return mlir::failure();
}

mlir::FailureOr<llvm::SmallVector<SymEngineAnalysis::SymExpr>> SymEngineAnalysis::applyAffineMap(
  mlir::AffineMap map, mlir::ValueRange symbols, const SymbolNameResolver &resolver) {
  if (map.getNumDims() != 0) {
    return mlir::failure();
  }

  if (symbols.size() != map.getNumSymbols()) {
    return mlir::failure();
  }

  llvm::SmallVector<SymExpr> symbolSyms;
  symbolSyms.reserve(symbols.size());
  for (mlir::Value symbolValue : symbols) {
    auto sym = getOrAssignSymbol(symbolValue, resolver);
    if (mlir::failed(sym)) {
      return mlir::failure();
    }
    symbolSyms.push_back(*sym);
  }

  llvm::SmallVector<SymExpr> results;
  results.reserve(map.getNumResults());
  for (auto expr : map.getResults()) {
    auto converted = convertAffineExpr(expr, /*dimSymbols=*/{}, symbolSyms);
    if (mlir::failed(converted)) {
      return mlir::failure();
    }
    results.push_back(*converted);
  }
  return results;
}

mlir::FailureOr<int64_t> SymEngineAnalysis::tryExtractInt64(const SymExpr &expr) const {
  if (!SymEngine::is_a<SymEngine::Integer>(*expr)) {
    return mlir::failure();
  }
  try {
    return static_cast<int64_t>(SymEngine::down_cast<const SymEngine::Integer &>(*expr).as_int());
  } catch (...) {
    return mlir::failure();
  }
}

bool SymEngineAnalysis::isStructurallyEqual(const SymExpr &lhs, const SymExpr &rhs) const {
  return SymEngine::eq(*lhs, *rhs);
}

bool SymEngineAnalysis::isStructurallyNotEqual(const SymExpr &lhs, const SymExpr &rhs) const {
  return SymEngine::neq(*lhs, *rhs);
}

void SymEngineAnalysis::reset() { valueToSymMap_.clear(); }

}  // namespace mfusion
