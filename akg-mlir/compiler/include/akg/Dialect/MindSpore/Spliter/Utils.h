/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef COMPILER_INCLUDE_AKG_DIALECT_MINDSPORE_SPLITER_UTILS_H_
#define COMPILER_INCLUDE_AKG_DIALECT_MINDSPORE_SPLITER_UTILS_H_

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/ToolUtilities.h"

#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "akg/Dialect/MindSpore/IR/MindSporeOps.h"
#include "akg/Dialect/MindSpore/Spliter/MindSporeToJson.h"

namespace mlir::spliter {
template <typename K, typename V, typename Hash = std::hash<K>, typename KeyEqual = std::equal_to<K>>
using HashMap = std::unordered_map<K, V, Hash, KeyEqual>;

template <typename T, typename Hash = std::hash<T>, typename Equal = std::equal_to<T>>
using HashSet = std::unordered_set<T, Hash, Equal>;

using ShapeVector = std::vector<int64_t>;

constexpr auto kCpuProcess = "cpu";
constexpr auto kCudaProcess = "cuda";
constexpr auto kAicoreProcess = "aicore";
// format
constexpr auto kOpFormat_DEFAULT = "DefaultFormat";
constexpr auto kOpFormat_NC1KHKWHWC0 = "NC1KHKWHWC0";
constexpr auto kOpFormat_ND = "ND";
constexpr auto kOpFormat_NCHW = "NCHW";
constexpr auto kOpFormat_NHWC = "NHWC";
constexpr auto kOpFormat_HWCN = "HWCN";
constexpr auto kOpFormat_CHWN = "CHWN";
constexpr auto kOpFormat_NC1HWC0 = "NC1HWC0";
constexpr auto kOpFormat_FRAC_Z = "FRACTAL_Z";
constexpr auto kOpFormat_FRACTAL_Z = "FRACTAL_Z";
constexpr auto kOpFormat_FRAC_NZ = "FRACTAL_NZ";
constexpr auto kAttrSrcFormat = "src_format";
constexpr auto kAttrDstFormat = "dst_format";
// op name
constexpr auto kReshapeOpName = "Reshape";
constexpr auto kAbsOpName = "Abs";
constexpr auto kAddOpName = "Add";
constexpr auto kSubOpName = "Sub";
constexpr auto kRealDivOpName = "RealDiv";
constexpr auto kDivOpName = "Div";
constexpr auto kMulOpName = "Mul";
constexpr auto kLogOpName = "Log";
constexpr auto kExpOpName = "Exp";
constexpr auto kPowOpName = "Pow";
constexpr auto kSqrtOpName = "Sqrt";
constexpr auto kRsqrtOpName = "Rsqrt";
constexpr auto kNegOpName = "Neg";
constexpr auto kReciprocalOpName = "Reciprocal";
constexpr auto kCastOpName = "Cast";
constexpr auto kRoundOpName = "Round";
constexpr auto kMaximumOpName = "Maximum";
constexpr auto kMinimumOpName = "Minimum";
constexpr auto kSelectOpName = "Select";
constexpr auto kLessOpName = "Less";
constexpr auto kEqualOpName = "Equal";
constexpr auto kNotEqualOpName = "NotEqual";
constexpr auto kLessEqualOpName = "LessEqual";
constexpr auto kGreaterEqualOpName = "GreaterEqual";
constexpr auto kGreaterOpName = "Greater";
constexpr auto kCRealOpName = "CReal";
constexpr auto kCImagOpName = "CImag";
constexpr auto kComplexOpName = "Complex";
constexpr auto kStandardNormalOpName = "StandardNormal";
constexpr auto kIsNanOpName = "IsNan";
constexpr auto kIsInfOpName = "IsInf";
constexpr auto kIsFiniteOpName = "IsFinite";
constexpr auto kFloorDivOpName = "FloorDiv";
constexpr auto kModOpName = "Mod";
constexpr auto kFloorOpName = "Floor";
constexpr auto kFloorModOpName = "FloorMod";
constexpr auto kErfOpName = "Erf";
constexpr auto kErfcOpName = "Erfc";
constexpr auto kLogicalNotOpName = "LogicalNot";
constexpr auto kLogicalAndOpName = "LogicalAnd";
constexpr auto kLogicalOrOpName = "LogicalOr";
constexpr auto kSignOpName = "Sign";
constexpr auto kSinOpName = "Sin";
constexpr auto kCosOpName = "Cos";
constexpr auto kAsinOpName = "Asin";
constexpr auto kACosOpName = "ACos";
constexpr auto kTanhOpName = "Tanh";
constexpr auto kAsinhOpName = "Asinh";
constexpr auto kAcoshOpName = "Acosh";
constexpr auto kAtanOpName = "Atan";
constexpr auto kAtan2OpName = "Atan2";
constexpr auto kExpm1OpName = "Expm1";
constexpr auto kBroadcastToOpName = "BroadcastTo";
constexpr auto kTileOpName = "Tile";
constexpr auto kReduceSumOpName = "ReduceSum";
constexpr auto kReduceMaxOpName = "ReduceMax";
constexpr auto kReduceMinOpName = "ReduceMin";
constexpr auto kArgmaxOpName = "Argmax";
constexpr auto kArgminOpName = "Argmin";
constexpr auto kOpaqueOpName = "_opaque";
constexpr auto kTransposeOpName = "Transpose";
constexpr auto kLayoutTransformOpName = "LayoutTransform";
constexpr auto kMatMulOpName = "MatMul";
constexpr auto kPadAkgOpName = "PadAkg";
constexpr auto kUnPadAkgOpName = "UnPadAkg";
constexpr auto kBatchMatMulOpName = "BatchMatMul";
constexpr auto kCumSumOpName = "CumSum";
constexpr auto kOneHotOpName = "OneHot";
constexpr auto kStridedSliceOpName = "StridedSlice";
constexpr auto kStridedSliceOnnxOpName = "StridedSliceOnnx";
constexpr auto kConcatOpName = "Concat";
constexpr auto kGatherOpName = "Gather";
constexpr auto kShapeOpName = "Shape";
constexpr auto kConstantOfShapeOpName = "ConstantOfShape";
constexpr auto kTensorScatterAddOpName = "TensorScatterAdd";
constexpr auto kGatherNdOpName = "GatherNd";
constexpr auto kUnsortedSegmentSumOpName = "UnsortedSegmentSum";
constexpr auto kConv2DOpName = "Conv2D";
constexpr auto kTransDataOpName = "TransData";
constexpr auto kElemAnyOpName = "ElemAny";
constexpr auto kPool2DOpName = "Pool2D";
constexpr auto kAssignOpName = "Assign";
constexpr auto kTupleGetItemOpName = "TupleGetItem";
constexpr auto kAddNOpName = "AddN";
constexpr auto kTensorAddOpName = "TensorAdd";

template <typename T>
T shapeSize(const std::vector<T> &shape) {
  return std::accumulate(shape.begin(), shape.end(), static_cast<T>(1), std::multiplies<T>());
}

inline size_t longToSize(int64_t u) {
  if (u < 0) {
    llvm::errs() << "The int64_t value(" << u << ") is less than 0.";
    return 0;
  }
  return static_cast<size_t>(u);
}

inline int64_t sizeToLong(size_t u) {
  if (u > static_cast<size_t>((std::numeric_limits<int64_t>::max)())) {
    llvm::errs() << "The size_t value(" << u << ") exceeds the maximum value of int64_t.";
    return 0;
  }
  return static_cast<int64_t>(u);
}

inline size_t typeIdSize(const Type type) {
  if (type.isa<IntegerType>()) {
    return type.cast<IntegerType>().getWidth();
  } else if (type.isa<FloatType>()) {
    return type.cast<FloatType>().getWidth();
  }

  llvm_unreachable("unsupported data type");
}

inline std::string typeIdToString(const mlir::Type dataType) { return mlir::JsonOpBuilder::getDataType(dataType); }

inline bool isGraphKernelOp(Operation *op) {
  if (op->getDialect()->getNamespace() == mlir::tosa::TosaDialect::getDialectNamespace() ||
      op->getDialect()->getNamespace() == mlir::mindspore::MindSporeDialect::getDialectNamespace()) {
    return true;
  }
  return false;
}

}  // namespace mlir::spliter
#endif  // COMPILER_INCLUDE_AKG_DIALECT_MINDSPORE_SPLITER_UTILS_H_
