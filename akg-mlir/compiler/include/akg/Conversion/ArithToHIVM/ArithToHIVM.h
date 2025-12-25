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

#ifndef COMPILER_INCLUDE_AKG_CONVERSION_ARITHTOHIVM_ARITHTOHIVM_H
#define COMPILER_INCLUDE_AKG_CONVERSION_ARITHTOHIVM_ARITHTOHIVM_H

#include <memory>

namespace mlir {
class LLVMTypeConverter;
class RewritePatternSet;
class Pass;

#define GEN_PASS_DECL_CONVERTARITHTOHIVM
#include "akg/Conversion/Passes.h.inc"

namespace hivm {
void populateArithToHIVMConversionPatterns(RewritePatternSet &patterns);
}  // namespace hivm

/// Creates a pass to convert the HFusion dialect to the HIVM dialect.
// std::unique_ptr<Pass> createLinalgToHFusionConversionPass();
std::unique_ptr<Pass> createArithToHIVMConversionPass();

}  // namespace mlir

#endif  // COMPILER_INCLUDE_AKG_CONVERSION_ARITHTOHIVM_ARITHTOHIVM_H
