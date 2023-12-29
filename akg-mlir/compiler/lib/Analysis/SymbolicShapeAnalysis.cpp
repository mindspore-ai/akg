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

#include "akg/Analysis/SymbolicShapeAnalysis.h"

#include <algorithm>
#include <string>
#include "akg/Analysis/TypeUtils.h"
#include "symengine/expression.h"

namespace mlir {
SymEngine::Expression SymbolicShapeAnalysis::getSymbolicExprFromStr(const std::string &symbol) {
  if (useCache && symbolicStrExprMap.find(symbol) != symbolicStrExprMap.end()) {
    return symbolicStrExprMap[symbol];
  }
  return SymEngine::Expression(symbol);
}

bool SymbolicShapeAnalysis::hasSymbolicShape(Type type) const {
  RankedTensorType ty = type.dyn_cast<RankedTensorType>();
  if (!ty) {
    return false;
  }
  mlir::DictionaryAttr dict = ty.getEncoding().dyn_cast_or_null<mlir::DictionaryAttr>();
  return dict && dict.contains(getSymbolShapeAttrName());
}

llvm::Optional<llvm::SmallVector<std::string>> SymbolicShapeAnalysis::getSymbolicShape(Type type) const {
  RankedTensorType ty = type.dyn_cast<RankedTensorType>();
  if (!ty || !hasSymbolicShape(ty)) {
    return std::nullopt;
  }
  mlir::DictionaryAttr dAttrs = ty.getEncoding().dyn_cast_or_null<mlir::DictionaryAttr>();
  ArrayAttr aAttrs = dAttrs.getAs<ArrayAttr>(getSymbolShapeAttrName());
  llvm::SmallVector<std::string> symbolicShape;
  (void)std::transform(aAttrs.getValue().begin(), aAttrs.getValue().end(), std::back_inserter(symbolicShape),
                       [](const Attribute &val) { return val.cast<StringAttr>().getValue().str(); });
  return symbolicShape;
}

llvm::Optional<NamedAttribute> SymbolicShapeAnalysis::getSymbolShapeNamedAttr(Type type) const {
  auto ty = type.dyn_cast<RankedTensorType>();
  if (!ty || !hasSymbolicShape(ty)) {
    return std::nullopt;
  }
  mlir::DictionaryAttr dAttrs = ty.getEncoding().dyn_cast_or_null<mlir::DictionaryAttr>();
  return dAttrs.getNamed(getSymbolShapeAttrName());
}

llvm::Optional<llvm::SmallVector<SymEngine::Expression>> SymbolicShapeAnalysis::getSymbolicShapeExpr(Type type) {
  llvm::Optional<llvm::SmallVector<std::string>> symbolicShape = getSymbolicShape(type);
  if (!symbolicShape) {
    return std::nullopt;
  }
  llvm::SmallVector<SymEngine::Expression> expr;
  for (auto const &symbol : *symbolicShape) {
    if (symbolicStrExprMap.find(symbol) != symbolicStrExprMap.end()) {
      (void)expr.emplace_back(symbolicStrExprMap[symbol]);
    } else {
      (void)expr.emplace_back(SymEngine::Expression(symbol));
    }
  }
  return expr;
}

llvm::Optional<std::string> SymbolicShapeAnalysis::getSymbolicDim(Type type, uint64_t idx) const {
  llvm::Optional<llvm::SmallVector<std::string>> symbolicShape = getSymbolicShape(type);
  if (!symbolicShape) {
    return std::nullopt;
  }
  return (*symbolicShape)[idx];
}

llvm::Optional<SymEngine::Expression> SymbolicShapeAnalysis::getSymbolicDimExpr(Type type, uint64_t idx) {
  llvm::Optional<llvm::SmallVector<std::string>> symbolicShape = getSymbolicShape(type);
  if (!symbolicShape) {
    return std::nullopt;
  }
  std::string symbol = (*symbolicShape)[idx];
  return getSymbolicExprFromStr(symbol);
}

Type SymbolicShapeAnalysis::createNewSymbolicShape(Type type) {
  auto ty = type.dyn_cast<RankedTensorType>();
  if (!ty) {
    return type;
  }
  mlir::DictionaryAttr dict = ty.getEncoding().dyn_cast_or_null<mlir::DictionaryAttr>();
  if (dict && dict.contains(getSymbolShapeAttrName())) {
    return type;
  }
  uint64_t rank = ty.cast<ShapedType>().getRank();
  ArrayRef<int64_t> shape = ty.cast<ShapedType>().getShape();
  llvm::SmallVector<Attribute> symShapeAttr;
  for (uint i = 0; i < rank; i++) {
    if (shape[i] == ShapedType::kDynamic) {
      (void)symShapeAttr.emplace_back(StringAttr::get(ty.getContext(), newSymbolicDim()));
    } else {
      (void)symShapeAttr.emplace_back(StringAttr::get(ty.getContext(), newSymbolicDimFromNumber(shape[i])));
    }
  }
  NamedAttribute namedAttr(StringAttr::get(ty.getContext(), getSymbolShapeAttrName()),
                           ArrayAttr::get(ty.getContext(), symShapeAttr));
  return updateTensorEncodingAttr(ty, namedAttr);
}

Type SymbolicShapeAnalysis::updateSymbolicShape(Type type, NamedAttribute &namedAttr) const {
  auto ty = type.dyn_cast<RankedTensorType>();
  if (!ty) {
    return type;
  }
  return updateTensorEncodingAttr(ty, namedAttr);
}

Type SymbolicShapeAnalysis::updateSymbolicShape(Type type, const llvm::SmallVector<std::string> &symbolicShape) const {
  auto ty = type.dyn_cast<RankedTensorType>();
  if (!ty) {
    return type;
  }
  llvm::SmallVector<Attribute> symShapeAttr;
  (void)std::transform(symbolicShape.begin(), symbolicShape.end(), std::back_inserter(symShapeAttr),
                       [&ty](const std::string &s) {
                         Attribute attr = StringAttr::get(ty.getContext(), s);
                         return attr;
                       });
  NamedAttribute namedAttr(StringAttr::get(ty.getContext(), getSymbolShapeAttrName()),
                           ArrayAttr::get(ty.getContext(), symShapeAttr));
  return updateTensorEncodingAttr(ty, namedAttr);
}

bool SymbolicShapeAnalysis::isRankedTensorStaticShape(Type type) {
  auto ty = type.dyn_cast<RankedTensorType>();
  if (!ty) {
    return false;
  }
  return ty.cast<ShapedType>().hasStaticShape();
}

bool SymbolicShapeAnalysis::isSameSymbolicDim(std::string lhs, std::string rhs) {
  if (lhs == rhs) {
    return true;
  }
  SymEngine::Expression lExpr = getSymbolicExprFromStr(lhs);
  SymEngine::Expression rExpr = getSymbolicExprFromStr(rhs);
  return lExpr == rExpr;
}

bool SymbolicShapeAnalysis::isSameSymbolicDim(Type lhs, uint64_t lhsIdx, Type rhs, uint64_t rhsIdx) {
  llvm::Optional<std::string> l = getSymbolicDim(lhs, lhsIdx);
  llvm::Optional<std::string> r = getSymbolicDim(rhs, rhsIdx);
  if (!l || !r) {
    return false;
  }
  return isSameSymbolicDim(*l, *r);
}

bool SymbolicShapeAnalysis::isSameSymbolicShape(Type lhs, Type rhs) {
  RankedTensorType l = lhs.dyn_cast<RankedTensorType>();
  RankedTensorType r = rhs.dyn_cast<RankedTensorType>();
  if (!l || !r) {
    return false;
  }
  uint64_t lRank = l.cast<ShapedType>().getRank();
  uint64_t rRank = r.cast<ShapedType>().getRank();
  if (lRank != rRank) {
    return false;
  }
  for (uint i = 0; i < lRank; i++) {
    if (!isSameSymbolicDim(lhs, i, rhs, i)) {
      return false;
    }
  }
  return true;
}

}  // namespace mlir
