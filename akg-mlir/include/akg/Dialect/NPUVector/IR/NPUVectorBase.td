/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef NPUVECTOR_BASE
#define NPUVECTOR_BASE

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonTypeConstraints.td"

def NPUVector_Dialect : Dialect {
  let name = "npuvector";
  let cppNamespace = "::mlir::npuvector";
  let description = [{
    The `npu_vector` dialect provides vector operations for NPU (Neural Processing Unit)
    with support for dynamic shapes. This dialect is inspired by the LLVM Vector dialect
    but is specifically designed for NPU targets.
  }];

  let dependentDialects = [
    "memref::MemRefDialect",
    "vector::VectorDialect",
    "arith::ArithDialect"
  ];

  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 0;  // Disable to use custom printType/parseType
  let hasConstantMaterializer = 1;
  let extraClassDeclaration = [{
    /// Parse a type registered to this dialect.
    ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const override;

    /// Print a type registered to this dialect.
    void printType(::mlir::Type type,
                   ::mlir::DialectAsmPrinter &printer) const override;
  }];
}

// Base class for NPUVector dialect ops.
class NPUVector_Op<string mnemonic, list<Trait> traits = []>
    : Op<NPUVector_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Type Constraints for NPUVector
//===----------------------------------------------------------------------===//

// Predicate to check if a type is NPUVectorType
def IsNPUVectorTypePred : CPred<"::llvm::isa<::mlir::npuvector::NPUVectorType>($_self)">;

// Predicate to check if a type is VectorType or NPUVectorType
def IsVectorOrNPUVectorTypePred : Or<[
  CPred<"::llvm::isa<::mlir::VectorType>($_self)">,
  CPred<"::llvm::isa<::mlir::npuvector::NPUVectorType>($_self)">
]>;

// Type constraint for NPUVectorType with any rank
class NPUVectorOfAnyRankOf<list<Type> allowedTypes> :
  ShapedContainerType<allowedTypes, IsNPUVectorTypePred, "npuvector",
                      "::mlir::npuvector::NPUVectorType">;

// Type constraint for NPUVectorType with any element type and any rank
def NPUVectorOfAnyRank : NPUVectorOfAnyRankOf<[AnyType]>;

// Type constraint for either VectorType or NPUVectorType
def AnyVectorOrNPUVectorOfAnyRank : Type<
  IsVectorOrNPUVectorTypePred,
  "vector or npuvector",
  "::mlir::Type">;

#endif // NPUVECTOR_BASE

