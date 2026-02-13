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

#ifndef MFUSION_DIALECT_MFUSE_UTILS_SYMBOLIC_SHAPE_UTILS_H
#define MFUSION_DIALECT_MFUSE_UTILS_SYMBOLIC_SHAPE_UTILS_H

#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"

namespace mlir {
namespace mfuse {
/// Utilities for mfuse symbolic shape (SymbolicShapeAttr) and per-symbol metadata (SymbolInfoAttr).
struct SymbolicShapeUtils {
  /// Attribute key for symbolic-shape encoding on tensor types (mfuse.symshape).
  static inline constexpr const char *kSymShapeKey = "mfuse.symshape";
  /// Attribute key for preserving a base encoding when merging with symshape (mfuse.encoding).
  static inline constexpr const char *kBaseEncodingKey = "mfuse.encoding";
  /// FuncOp attribute key for per-symbol metadata dict (mfuse.syminfo).
  static inline constexpr const char *kFuncSymInfoKey = "mfuse.syminfo";

  // ---------------------------------------------------------------------------
  // SymbolicShapeAttr: tensor encoding (mfuse.symshape), shape expressions
  // ---------------------------------------------------------------------------

  /// Returns true iff encoding is a direct SymbolicShapeAttr (not wrapped in a dict).
  static bool isSymbolicShapeEncoding(mlir::Attribute encoding);

  /// Returns true iff type is a RankedTensorType with no dynamic dimensions.
  static inline bool isStaticShape(mlir::Type type) {
    auto ranked = type.dyn_cast<mlir::RankedTensorType>();
    return ranked && ranked.hasStaticShape();
  }

  /// Returns true when type carries symbolic-shape info: either its encoding is
  /// a direct SymbolicShapeAttr or a DictionaryAttr with key kSymShapeKey.
  static inline bool hasSymbolicShapeEncoding(mlir::Type type) {
    return isSymbolicShapeEncoding(getSymbolicShapeAttrFromEncoding(type));
  }

  /// Extracts the symbolic-shape attribute from type's encoding. Returns null
  /// if none. Result is generic Attribute; callers can dyn_cast to SymbolicShapeAttr.
  static mlir::Attribute getSymbolicShapeAttrFromEncoding(mlir::Type type);

  /// Merges symshapeAttr into type's encoding. Preserves any existing
  /// encoding under kBaseEncodingKey when the result would otherwise have multiple entries.
  static mlir::Attribute mergeEncoding(mlir::RankedTensorType type, mlir::Attribute symshapeAttr);

  /// Returns a RankedTensorType with the same shape and element type as type
  /// but encoding set to include symshapeAttr (via mergeEncoding).
  static mlir::RankedTensorType withSymbolicAttr(mlir::RankedTensorType type, mlir::Attribute symshapeAttr);

  /// Attaches symshapeAttr to value by updating its type. Succeeds only for
  /// OpResult or BlockArgument; returns false if type unchanged or value kind unsupported.
  static bool attachToValue(mlir::Value value, mlir::Attribute symshapeAttr);

  // ---------------------------------------------------------------------------
  // SymbolInfoAttr: func attribute mfuse.syminfo, per-symbol metadata (e.g. range)
  // ---------------------------------------------------------------------------

  /// Returns the operation that owns the builder's current insertion block, or null.
  static inline mlir::Operation *getInsertionParentOp(mlir::OpBuilder &builder) {
    mlir::Block *block = builder.getInsertionBlock();
    return block ? block->getParentOp() : nullptr;
  }

  /// Returns the enclosing func.func for the builder's insertion point, or null.
  static inline mlir::func::FuncOp getParentFunc(mlir::OpBuilder &builder) {
    if (mlir::Operation *parent = getInsertionParentOp(builder)) {
      return parent->getParentOfType<mlir::func::FuncOp>();
    }
    return {};
  }

  /// Returns the func's mfuse.syminfo dictionary attribute, or null if absent.
  static inline mlir::DictionaryAttr getFuncSymInfo(mlir::func::FuncOp func) {
    return func ? func->getAttrOfType<mlir::DictionaryAttr>(kFuncSymInfoKey) : mlir::DictionaryAttr();
  }

  /// Returns the mfuse.syminfo dict of the func containing the builder's insertion point.
  static inline mlir::DictionaryAttr getFuncSymInfo(mlir::OpBuilder &builder) {
    return getFuncSymInfo(getParentFunc(builder));
  }

  /// Looks up the per-symbol attribute for symbolName in func's mfuse.syminfo.
  /// Returns null if the dict or entry is missing. Callers can dyn_cast to SymbolInfoAttr.
  static mlir::Attribute getSymbolInfoAttr(mlir::func::FuncOp func, llvm::StringRef symbolName);

  /// Looks up the per-symbol attribute for symbolName in the syminfo of the func
  /// containing the builder's insertion point. Returns null if no func or no entry.
  static inline mlir::Attribute getSymbolInfoAttr(mlir::OpBuilder &builder, llvm::StringRef symbolName) {
    auto func = getParentFunc(builder);
    return func ? getSymbolInfoAttr(func, symbolName) : mlir::Attribute();
  }
};
}  // namespace mfuse
}  // namespace mlir

#endif  // MFUSION_DIALECT_MFUSE_UTILS_SYMBOLIC_SHAPE_UTILS_H
