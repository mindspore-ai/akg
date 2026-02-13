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

#include "mfusion/Dialect/Mfuse/Utils/SymbolicShapeUtils.h"
#include "mfusion/Dialect/Mfuse/Mfuse.h"

namespace mlir {
namespace mfuse {

// ---------------------------------------------------------------------------
// SymbolicShapeAttr: tensor encoding (mfuse.symshape), shape expressions
// ---------------------------------------------------------------------------

bool SymbolicShapeUtils::isSymbolicShapeEncoding(mlir::Attribute encoding) {
  return encoding && encoding.isa<mlir::mfuse::SymbolicShapeAttr>();
}

mlir::Attribute SymbolicShapeUtils::getSymbolicShapeAttrFromEncoding(mlir::Type type) {
  auto ranked = type.dyn_cast<mlir::RankedTensorType>();
  if (!ranked) {
    return {};
  }
  auto encoding = ranked.getEncoding();
  if (!encoding) {
    return {};
  }
  if (auto dict = encoding.dyn_cast<mlir::DictionaryAttr>()) {
    return dict.get(mlir::StringAttr::get(type.getContext(), kSymShapeKey));
  }
  if (isSymbolicShapeEncoding(encoding)) {
    return encoding;
  }
  return {};
}

mlir::Attribute SymbolicShapeUtils::mergeEncoding(mlir::RankedTensorType type, mlir::Attribute symshapeAttr) {
  auto encoding = type.getEncoding();
  mlir::MLIRContext *ctx = type.getContext();
  auto symKey = mlir::StringAttr::get(ctx, kSymShapeKey);
  auto baseKey = mlir::StringAttr::get(ctx, kBaseEncodingKey);

  // Common case: no existing encoding. Keep the tensor type compact by storing
  // the SymbolicShapeAttr directly as the encoding (instead of a DictionaryAttr
  // with a single entry).
  if (!encoding) {
    return symshapeAttr;
  }

  // Preserve the existing encoding when it is already the same symbolic-shape encoding.
  if (encoding == symshapeAttr) {
    return encoding;
  }

  // If the existing encoding is already a direct mfuse.symshape attribute,
  // don't wrap it in a dictionary; just replace/keep the direct symshape.
  if (isSymbolicShapeEncoding(encoding)) {
    return symshapeAttr;
  }

  if (auto dict = encoding.dyn_cast_or_null<mlir::DictionaryAttr>()) {
    auto existing = dict.get(symKey);
    if (existing == symshapeAttr) {
      return dict;
    }

    llvm::SmallVector<mlir::NamedAttribute> entries;
    entries.reserve(dict.getValue().size() + 1);
    bool replaced = false;
    for (const auto &entry : dict.getValue()) {
      if (entry.getName() == symKey) {
        entries.emplace_back(symKey, symshapeAttr);
        replaced = true;
        continue;
      }
      entries.push_back(entry);
    }
    if (!replaced) {
      entries.emplace_back(symKey, symshapeAttr);
    }
    auto updated = mlir::DictionaryAttr::get(ctx, entries);
    // If the dictionary only carries symshape, fold back to a direct encoding.
    if (updated.getValue().size() == 1 && updated.get(symKey) == symshapeAttr) {
      return symshapeAttr;
    }
    return updated;
  }

  llvm::SmallVector<mlir::NamedAttribute> entries;
  entries.reserve(encoding ? 2 : 1);
  entries.emplace_back(symKey, symshapeAttr);
  // Avoid duplicating when the base encoding is already the symshape encoding.
  if (encoding != symshapeAttr) {
    entries.emplace_back(baseKey, encoding);
  }
  return mlir::DictionaryAttr::get(ctx, entries);
}

mlir::RankedTensorType SymbolicShapeUtils::withSymbolicAttr(mlir::RankedTensorType type, mlir::Attribute symshapeAttr) {
  auto merged = mergeEncoding(type, symshapeAttr);
  if (merged == type.getEncoding()) {
    return type;
  }
  return mlir::RankedTensorType::get(type.getShape(), type.getElementType(), merged);
}

bool SymbolicShapeUtils::attachToValue(mlir::Value value, mlir::Attribute symshapeAttr) {
  auto ranked = value.getType().dyn_cast<mlir::RankedTensorType>();
  if (!ranked) {
    return false;
  }
  auto newType = withSymbolicAttr(ranked, symshapeAttr);
  if (newType == ranked) {
    return true;
  }

  if (auto result = value.dyn_cast<mlir::OpResult>()) {
    result.setType(newType);
    return true;
  }
  if (auto arg = value.dyn_cast<mlir::BlockArgument>()) {
    arg.setType(newType);
    return true;
  }
  return false;
}

// ---------------------------------------------------------------------------
// SymbolInfoAttr: func attribute mfuse.syminfo, per-symbol metadata (e.g. range)
// ---------------------------------------------------------------------------

mlir::Attribute SymbolicShapeUtils::getSymbolInfoAttr(mlir::func::FuncOp func, llvm::StringRef symbolName) {
  auto dict = getFuncSymInfo(func);
  if (!dict) {
    return {};
  }
  auto key = mlir::StringAttr::get(func.getContext(), symbolName);
  return dict.get(key);
}

}  // namespace mfuse
}  // namespace mlir
