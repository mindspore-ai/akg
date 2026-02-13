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

#include "mfusion/Dialect/Mfuse/Mfuse.h"

#include <limits>

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/Support/raw_ostream.h"
#include "symengine/parser.h"

namespace mlir::mfuse {

static mlir::ParseResult parseSymbolBound(mlir::AsmParser &parser, int64_t &value) {
  if (succeeded(parser.parseOptionalKeyword("inf"))) {
    value = std::numeric_limits<int64_t>::max();
    return mlir::success();
  }
  if (succeeded(parser.parseOptionalKeyword("ninf"))) {
    value = std::numeric_limits<int64_t>::min();
    return mlir::success();
  }
  return parser.parseInteger(value);
}

static void printSymbolBound(mlir::AsmPrinter &printer, int64_t value) {
  if (value == std::numeric_limits<int64_t>::max()) {
    printer << "inf";
    return;
  }
  if (value == std::numeric_limits<int64_t>::min()) {
    printer << "ninf";
    return;
  }
  printer << value;
}

mlir::Attribute SymbolInfoAttr::parse(mlir::AsmParser &parser, mlir::Type) {
  int64_t minVal = 0;
  int64_t maxVal = 0;
  if (parser.parseLess()) return {};
  if (parser.parseKeyword("range")) return {};
  if (parser.parseEqual()) return {};
  if (parser.parseLSquare()) return {};
  if (parseSymbolBound(parser, minVal)) return {};
  if (parser.parseComma()) return {};
  if (parseSymbolBound(parser, maxVal)) return {};
  if (parser.parseRSquare()) return {};
  if (parser.parseGreater()) return {};
  return SymbolInfoAttr::get(parser.getContext(), minVal, maxVal);
}

void SymbolInfoAttr::print(mlir::AsmPrinter &printer) const {
  printer << "<range=[";
  printSymbolBound(printer, getMinVal());
  printer << ", ";
  printSymbolBound(printer, getMaxVal());
  printer << "]>";
}

std::vector<SymEngine::RCP<const SymEngine::Basic>> SymbolicShapeAttr::getSymEngineExprs() const {
  std::vector<SymEngine::RCP<const SymEngine::Basic>> exprs;
  auto listAttr = getExprs();
  exprs.reserve(listAttr.size());

  for (auto attr : listAttr) {
    auto strAttr = attr.dyn_cast<mlir::StringAttr>();
    if (!strAttr) {
      llvm::errs() << "mfuse.symbolic_shape expects StringAttr entries\n";
      continue;
    }
    exprs.push_back(SymEngine::parse(strAttr.getValue().str()));
  }
  return exprs;
}

}  // namespace mlir::mfuse
