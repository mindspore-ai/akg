/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
//===- ConvertFuncToLLVM.h - Convert Func to LLVM ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides a set of conversion patterns from the Func dialect to the LLVM IR
// dialect.
//
//===----------------------------------------------------------------------===//

#ifndef AKG_CONVERSION_FUNCTOLLVMEXT_CONVERTFUNCTOLLVMEXT_H
#define AKG_CONVERSION_FUNCTOLLVMEXT_CONVERTFUNCTOLLVMEXT_H

#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {

namespace LLVM {
class LLVMFuncOp;
}  // namespace LLVM

class ConversionPatternRewriter;
class DialectRegistry;
class LLVMTypeConverter;
class RewritePatternSet;
class SymbolTable;

/// Convert input FunctionOpInterface operation to LLVMFuncOp by using the
/// provided LLVMTypeConverter. Return failure if failed to so.
FailureOr<LLVM::LLVMFuncOp> convertFuncOpToLLVMExtFuncOp(FunctionOpInterface funcOp,
                                                         ConversionPatternRewriter &rewriter,
                                                         const LLVMTypeConverter &converter);

/// Collect the default pattern to convert a FuncOp to the LLVM dialect. If
/// `emitCWrappers` is set, the pattern will also produce functions
/// that pass memref descriptors by pointer-to-structure in addition to the
/// default unpacked form.
void populateFuncToLLVMExtFuncOpConversionPattern(LLVMTypeConverter &converter, RewritePatternSet &patterns);

/// Collect the patterns to convert from the Func dialect to LLVM. The
/// conversion patterns capture the LLVMTypeConverter and the LowerToLLVMOptions
/// by reference meaning the references have to remain alive during the entire
/// pattern lifetime.
///
/// The `symbolTable` parameter can be used to speed up function lookups in the
/// module. It's good to provide it, but only if we know that the patterns will
/// be applied to a single module and the symbols referenced by the symbol table
/// will not be removed and new symbols will not be added during the usage of
/// the patterns. If provided, the lookups will have O(calls) cumulative
/// runtime, otherwise O(calls * functions). The symbol table is currently not
/// needed if `converter.getOptions().useBarePtrCallConv` is `true`, but it's
/// not an error to provide it anyway.
void populateFuncToLLVMExtConversionPatterns(LLVMTypeConverter &converter, RewritePatternSet &patterns,
                                             const SymbolTable *symbolTable = nullptr);

}  // namespace mlir

#endif  // AKG_CONVERSION_FUNCTOLLVMEXT_CONVERTFUNCTOLLVMEXT_H
