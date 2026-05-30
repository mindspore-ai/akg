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
//===- ConvertFuncToLLVMExtPass.h - Pass entrypoint ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AKG_CONVERSION_FUNCTOLLVMEXT_CONVERTFUNCTOLLVMEXTPASS_H_
#define AKG_CONVERSION_FUNCTOLLVMEXT_CONVERTFUNCTOLLVMEXTPASS_H_

#include <memory>
#include <string>
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
class Pass;
#ifndef GEN_PASS_DECL_FUNCTOLLVMEXT
#define GEN_PASS_DECL_FUNCTOLLVMEXT
#include "mlir/Conversion/Passes.h.inc"
#endif
}  // namespace mlir

#endif  // AKG_CONVERSION_FUNCTOLLVMEXT_CONVERTFUNCTOLLVMEXTPASS_H_
