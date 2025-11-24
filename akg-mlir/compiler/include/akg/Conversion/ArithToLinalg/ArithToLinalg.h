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
//===- ArithToLinalg.h - Arith to Linalg dialect conversion -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef COMPILER_INCLUDE_AKG_CONVERSION_ARITHTOLINALG_ARITHTOLINALG_H
#define COMPILER_INCLUDE_AKG_CONVERSION_ARITHTOLINALG_ARITHTOLINALG_H

#include <memory>

namespace mlir {

class RewritePatternSet;
class Pass;

#define GEN_PASS_DECL_CONVERTARITHTOLINALG
#include "akg/Conversion/Passes.h.inc"

namespace arith {
void populateArithToLinalgConversionPatterns(RewritePatternSet &patterns);
} // namespace arith

/// Creates a pass to convert the Arith dialect to the Linalg dialect.
std::unique_ptr<Pass> createArithToLinalgConversionPass();
} // namespace mlir

#endif // COMPILER_INCLUDE_AKG_CONVERSION_ARITHTOLINALG_ARITHTOLINALG_H
