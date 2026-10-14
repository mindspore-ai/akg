// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-FileCopyrightText: (c) 2026 Huawei Technologies Co., Ltd
//
// SPDX-License-Identifier: Apache-2.0
//
// This file was copied from project tenstorrent/tt-mlir
// (https://github.com/tenstorrent/tt-mlir) and modified for AKG-MLIR.

#include "akg/Dialect/EmitPy/IR/EmitPyTypes.h"

#include "akg/Dialect/EmitPy/IR/EmitPy.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/LogicalResult.h"

using namespace mlir;        // NOLINT(build/namespaces)
using namespace mlir::emitpy;  // NOLINT(build/namespaces)

#define GET_TYPEDEF_CLASSES
#include "akg/Dialect/EmitPy/IR/EmitPyOpsTypes.cpp.inc"

void EmitPyDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "akg/Dialect/EmitPy/IR/EmitPyOpsTypes.cpp.inc"  // NOLINT(build/include)
      >();
}

::mlir::LogicalResult mlir::emitpy::OpaqueType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::StringRef value) {
  if (value.empty()) {
    return emitError() << "expected non-empty string in !emitpy.opaque type";
  }
  return success();
}

::mlir::LogicalResult mlir::emitpy::ClassType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::StringRef name) {
  if (name.empty()) {
    return emitError() << "expected non-empty string in !emitpy.class type";
  }
  return success();
}

void DictType::print(::mlir::AsmPrinter &printer) const {
  auto keyType = getKeyType();
  auto valueType = getValueType();

  if (keyType || valueType) {
    printer << "<";
    auto opaqueType = keyType.dyn_cast<OpaqueType>();
    if (!opaqueType || opaqueType.getValue() != "Any") {
      printer.printType(keyType);
    } else {
      printer << "Any";
    }
    printer << ", ";
    opaqueType = valueType.dyn_cast<OpaqueType>();
    if (!opaqueType || opaqueType.getValue() != "Any") {
      printer.printType(valueType);
    } else {
      printer << "Any";
    }
    printer << ">";
  }
}

Type DictType::parse(::mlir::AsmParser &parser) {
  ::mlir::MLIRContext *context = parser.getContext();
  Type keyType;
  Type valueType;

  if (succeeded(parser.parseOptionalLess())) {
    if (succeeded(parser.parseOptionalKeyword("Any"))) {
      keyType = OpaqueType::get(context, "Any");
    } else {
      Type parsedKeyType;
      if (failed(parser.parseType(parsedKeyType))) {
        parser.emitError(parser.getNameLoc())
            << "expected specified key type or 'Any' keyword";
        return Type();
      }
      keyType = parsedKeyType;
    }

    if (failed(parser.parseComma())) {
      parser.emitError(parser.getNameLoc()) << "expected comma after key type";
      return Type();
    }

    if (succeeded(parser.parseOptionalKeyword("Any"))) {
      valueType = OpaqueType::get(context, "Any");
    } else {
      Type parsedValueType;
      if (failed(parser.parseType(parsedValueType))) {
        parser.emitError(parser.getNameLoc())
            << "expected specified value type or 'Any' keyword";
        return Type();
      }
      valueType = parsedValueType;
    }

    if (failed(parser.parseGreater())) {
      parser.emitError(parser.getNameLoc()) << "expected '>'";
      return Type();
    }
  }

  return DictType::get(context, keyType, valueType);
}
