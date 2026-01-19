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

#include "mfusion/Dialect/Muse/Muse.h"
#include <algorithm>
#include <iterator>

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace muse {

llvm::LogicalResult ReshapeOp::verify() {
  auto listType = mlir::dyn_cast<ListType>(getShape().getType());
  if (!listType || !mlir::isa<I64Type>(listType.getContainedType())) {
    return emitOpError("expects shape to be a muse.i64_array type");
  }
  return llvm::success();
}

namespace {

llvm::LogicalResult extractShapeFromSize(mlir::Value size, llvm::SmallVectorImpl<int64_t> &shape) {
  shape.clear();

  // Case 1: created from a constant i64 array.
  if (auto createOp = size.getDefiningOp<CreateI64ArrayOp>()) {
    auto arrAttr = mlir::cast<mlir::ArrayAttr>(createOp.getValueAttr());
    shape.reserve(arrAttr.size());
    std::transform(arrAttr.begin(), arrAttr.end(), std::back_inserter(shape),
                   [](mlir::Attribute attr) { return mlir::cast<mlir::IntegerAttr>(attr).getInt(); });
    return llvm::success();
  }

  // Case 2: list built from scalar i64 values (e.g. MakeListOp of Muse_I64Type).
  if (auto makeList = size.getDefiningOp<MakeListOp>()) {
    shape.reserve(makeList.getNumOperands());
    std::transform(makeList.getOperands().begin(), makeList.getOperands().end(), std::back_inserter(shape),
                   [](mlir::Value dimVal) {
                     // Only accept muse scalar i64 dimensions with constant value.
                     if (mlir::isa<mlir::muse::I64Type>(dimVal.getType())) {
                       if (auto cst = dimVal.getDefiningOp<CreateI64Op>()) {
                         return cst.getValueAttr().getInt();
                       }
                     }
                     return mlir::ShapedType::kDynamic;
                   });
    return llvm::success();
  }

  // Unknown producer or non-constant list, use dynamic dims.
  return llvm::failure();
}

mlir::muse::DeviceAttr buildDevice(mlir::MLIRContext *ctx, mlir::StringRef deviceStr) {
  auto parts = deviceStr.split(':');
  int64_t index = -1;
  if (!parts.second.empty()) {
    int64_t parsedIndex = -1;
    if (!parts.second.getAsInteger(10, parsedIndex)) {
      index = parsedIndex;
    }
  }
  auto typeAttr = mlir::StringAttr::get(ctx, parts.first);
  auto indexAttr = mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), index);
  return mlir::muse::DeviceAttr::get(ctx, typeAttr, indexAttr);
}

mlir::LogicalResult inferTensorWithShapeDtypeDevice(mlir::MLIRContext *ctx, std::optional<mlir::Location> loc,
                                                    mlir::ValueRange operands,
                                                    llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  if (operands.size() < 3) {
    return mlir::emitOptionalError(loc, "expected three operands: size, dtype, device");
  }

  mlir::Value size = operands[0];
  mlir::Value dtype = operands[1];
  mlir::Value device = operands[2];

  auto dtypeTy = mlir::dyn_cast<mlir::muse::DtypeType>(dtype.getType());
  if (!dtypeTy) {
    return mlir::emitOptionalError(loc, "dtype must be muse.dtype type");
  }
  mlir::Type elementType = dtypeTy.getElementType();

  llvm::SmallVector<int64_t> shape;
  if (mlir::failed(extractShapeFromSize(size, shape))) {
    return mlir::emitOptionalError(loc, "extract shape from size failed");
  }

  auto devConst = device.getDefiningOp<CreateStringOp>();
  if (!devConst) {
    return mlir::emitOptionalError(loc, "device must be constant string");
  }
  auto deviceAttr = mlir::dyn_cast<mlir::StringAttr>(devConst.getValueAttr());
  if (!deviceAttr || deviceAttr.getValue().empty()) {
    return mlir::emitOptionalError(loc, "device string is empty");
  }

  auto dev = buildDevice(ctx, deviceAttr.getValue());
  auto tensorTy = mlir::muse::TensorType::get(ctx, shape, elementType, dev);
  inferredReturnTypes.push_back(tensorTy);
  return llvm::success();
}

}  // namespace

llvm::LogicalResult ZerosOp::inferReturnTypes(mlir::MLIRContext *ctx, std::optional<mlir::Location> loc,
                                              mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                              mlir::OpaqueProperties properties, mlir::RegionRange regions,
                                              llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return inferTensorWithShapeDtypeDevice(ctx, loc, operands, inferredReturnTypes);
}

llvm::LogicalResult EmptyOp::inferReturnTypes(mlir::MLIRContext *ctx, std::optional<mlir::Location> loc,
                                              mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                              mlir::OpaqueProperties properties, mlir::RegionRange regions,
                                              llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return inferTensorWithShapeDtypeDevice(ctx, loc, operands, inferredReturnTypes);
}

}  // namespace muse
}  // namespace mlir
