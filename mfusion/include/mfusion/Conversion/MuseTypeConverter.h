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

#ifndef MFUSION_CONVERSION_MUSE_TYPE_CONVERTER_H
#define MFUSION_CONVERSION_MUSE_TYPE_CONVERTER_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mfusion/Dialect/Muse/Muse.h"

namespace mlir {
namespace muse {

// Populate type conversions for converting scalar types to tensor types.
// This adds conversions for:
//   - IntegerType         -> tensor<int>
//   - Float16Type/BF16    -> tensor<f16/bf16>
//   - Float32Type         -> tensor<f32>
//   - Float64Type         -> tensor<f64>
//   - IndexType           -> tensor<i64>
inline void populateMuseScalarTypeConversions(mlir::TypeConverter &converter) {
  // Integer types
  converter.addConversion([](mlir::IntegerType type) -> mlir::Type { return mlir::RankedTensorType::get({}, type); });

  // Float types
  converter.addConversion([](mlir::Float16Type type) -> mlir::Type { return mlir::RankedTensorType::get({}, type); });
  converter.addConversion([](mlir::BFloat16Type type) -> mlir::Type { return mlir::RankedTensorType::get({}, type); });
  converter.addConversion([](mlir::Float32Type type) -> mlir::Type { return mlir::RankedTensorType::get({}, type); });
  converter.addConversion([](mlir::Float64Type type) -> mlir::Type { return mlir::RankedTensorType::get({}, type); });

  // Index type
  converter.addConversion([](mlir::IndexType type) -> mlir::Type {
    return mlir::RankedTensorType::get({}, mlir::IntegerType::get(type.getContext(), 64));
  });
}

// Populate all type conversions for converting standard MLIR types to Muse types.
// This includes scalar type conversions. Tensor types use built-in RankedTensorType.
inline void populateMuseTypeConversions(mlir::TypeConverter &converter) {
  populateMuseScalarTypeConversions(converter);
}

// Populate materialization functions for bridging between builtin types and Muse types.
// This uses UnrealizedConversionCastOp for bridging, which should be cleaned up
// later by reconcile-unrealized-casts pass.
inline void populateMuseTypeMaterializations(mlir::TypeConverter &converter) {
  // Target materialization: builtin types -> Muse types
  // Used when creating new ops that expect Muse types
  converter.addTargetMaterialization(
    [](mlir::OpBuilder &builder, mlir::Type toType, mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
      if (inputs.size() != 1) return {};
      return builder.create<mlir::UnrealizedConversionCastOp>(loc, toType, inputs).getResult(0);
    });

  // Source materialization: Muse types -> builtin types
  // Used to maintain compatibility with surrounding code that expects builtin types
  converter.addSourceMaterialization(
    [](mlir::OpBuilder &builder, mlir::Type toType, mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
      if (inputs.size() != 1) return {};
      return builder.create<mlir::UnrealizedConversionCastOp>(loc, toType, inputs).getResult(0);
    });
}
}  // namespace muse
}  // namespace mlir

#endif  // MFUSION_CONVERSION_MUSE_TYPE_CONVERTER_H
