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

#ifndef AKG_CONVERSION_VECTORTOEMITPY_VECTORTOEMITPY_H
#define AKG_CONVERSION_VECTORTOEMITPY_VECTORTOEMITPY_H

#include <memory>
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

#define GEN_PASS_DECL_CONVERTVECTORTOEMITPY
#include "akg/Conversion/Passes.h.inc"

class TypeConverter;

namespace vector_to_emitpy {

/// Collects a set of patterns that rewrite tile-level NPUVector/Vector/Arith/
/// Math/MemRef operations into `emitpy.call_opaque` operations whose callees
/// are NumPy functions, ready for `akg-translate --mlir-to-pypto`.
void populateVectorToEmitPyConversionPatterns(RewritePatternSet &patterns,
                                              TypeConverter &typeConverter);

}  // namespace vector_to_emitpy

/// Creates a pass that converts tile-level vector computations to EmitPy ops.
std::unique_ptr<Pass> createVectorToEmitPyConversionPass();

}  // namespace mlir

#endif  // AKG_CONVERSION_VECTORTOEMITPY_VECTORTOEMITPY_H
