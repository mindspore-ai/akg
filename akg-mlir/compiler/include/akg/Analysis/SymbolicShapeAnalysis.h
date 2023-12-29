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

#ifndef COMPILER_INCLUDE_AKG_ANALYSIS_SYMBOLICSHAPEANALYSIS_H_
#define COMPILER_INCLUDE_AKG_ANALYSIS_SYMBOLICSHAPEANALYSIS_H_

#include <algorithm>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "mlir/IR/BuiltinTypes.h"
#include "symengine/basic.h"
#include "symengine/expression.h"
#include "symengine/integer.h"
#include "symengine/number.h"
#include "symengine/parser.h"
#include "symengine/parser/parser.h"
#include "symengine/symbol.h"

namespace mlir {
constexpr StringRef inline getSymbolShapeAttrName() { return "SymShapeAttr"; }
constexpr StringRef inline getFrontendSymbolAttrName() { return "frontend_symbol"; }

class SymbolicShapeAnalysis {
 public:
  ~SymbolicShapeAnalysis() {}
  SymbolicShapeAnalysis(const SymbolicShapeAnalysis &) = delete;
  SymbolicShapeAnalysis &operator=(const SymbolicShapeAnalysis &) = delete;
  static SymbolicShapeAnalysis &getInstance() {
    static SymbolicShapeAnalysis instance;
    return instance;
  }

  std::string newSymbolicDim() {
    const std::string s("s" + std::to_string(uniqueNum++));
    if (useCache) {
      const SymEngine::Expression expr(s);
      (void)symbolicStrExprMap.insert(std::make_pair(s, expr));
    }
    return s;
  }
  SymEngine::Expression getNewSymbolicDimExpr() {
    const std::string s("s" + std::to_string(uniqueNum++));
    const SymEngine::Expression expr(s);
    if (useCache) {
      (void)symbolicStrExprMap.insert(std::make_pair(s, expr));
    }
    return expr;
  }
  std::string newSymbolicDimFromNumber(int64_t dim) {
    const std::string s = std::to_string(dim);
    if (useCache) {
      const SymEngine::Expression expr(s);
      (void)symbolicStrExprMap.insert(std::make_pair(s, expr));
    }
    return s;
  }
  std::string getSymbolicDimFromExpression(SymEngine::Expression expr) {
    const std::string s = SymEngine::detail::poly_print(expr);
    if (useCache) {
      (void)symbolicStrExprMap.insert(std::make_pair(s, expr));
    }
    return s;
  }
  SymEngine::Expression getSymbolicExprFromStr(const std::string &symbol);
  template <class Container>
  llvm::SmallVector<SymEngine::Expression> getSymbolicExprsFromStrs(Container symbols);
  bool hasSymbolicShape(Type type) const;
  llvm::Optional<llvm::SmallVector<std::string>> getSymbolicShape(Type type) const;
  llvm::Optional<NamedAttribute> getSymbolShapeNamedAttr(Type type) const;
  llvm::Optional<llvm::SmallVector<SymEngine::Expression>> getSymbolicShapeExpr(Type type);
  llvm::Optional<std::string> getSymbolicDim(Type type, uint64_t idx) const;
  llvm::Optional<SymEngine::Expression> getSymbolicDimExpr(Type type, uint64_t idx);
  Type createNewSymbolicShape(Type type);
  Type updateSymbolicShape(Type type, NamedAttribute &nameAttr) const;
  Type updateSymbolicShape(Type type, const llvm::SmallVector<std::string> &symbolicShape) const;
  bool isRankedTensorStaticShape(Type type);
  bool isSameSymbolicDim(std::string lhs, std::string rhs);
  bool isSameSymbolicDim(Type lhs, uint64_t lhsIdx, Type rhs, uint64_t rhsIdx);
  bool isSameSymbolicShape(Type lhs, Type rhs);

 private:
  SymbolicShapeAnalysis() : uniqueNum(0), useCache(false) { symbolicStrExprMap.clear(); }
  // for compile time reasons, SymEngine::Expression can be cached. This option
  // is disabled by default.
  std::map<const std::string, const SymEngine::Expression> symbolicStrExprMap;
  uint64_t uniqueNum;
  bool useCache;
};

template <typename Container>
llvm::SmallVector<SymEngine::Expression> SymbolicShapeAnalysis::getSymbolicExprsFromStrs(Container symbols) {
  llvm::SmallVector<SymEngine::Expression> exprs;
  (void)std::transform(symbols.begin(), symbols.end(), std::back_inserter(exprs),
                       [this](const std::string &s) { return this->getSymbolicExprFromStr(s); });
  return exprs;
}
}  // namespace mlir
#endif  // COMPILER_INCLUDE_AKG_ANALYSIS_SYMBOLICSHAPEANALYSIS_H_
