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

#ifndef AKG_UTILS_ANALYSISFORNPU_H
#define AKG_UTILS_ANALYSISFORNPU_H

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <string>
#include <utility>
#include <vector>
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "akg/Utils/Constants.h"

namespace mlir {
namespace akg {

constexpr int64_t kNpuBitsPerByte = 8;
constexpr int64_t kNpuUbAlignBytes = 32;
constexpr int64_t kNpuUbAlignBits = kNpuUbAlignBytes * kNpuBitsPerByte;

constexpr auto kSoc910B1 = "Ascend910B1";
constexpr auto kSoc910B2 = "Ascend910B2";
constexpr auto kSoc910B2C = "Ascend910B2C";
constexpr auto kSoc910B3 = "Ascend910B3";
constexpr auto kSoc910B4 = "Ascend910B4";
constexpr auto kSoc910B4_1 = "Ascend910B4-1";
constexpr auto kSoc950PR_9599 = "Ascend950PR_9599";

class HardwareConfig {
 public:
  HardwareConfig() = default;
  ~HardwareConfig() = default;

  uint32_t coreNumAic{0};
  uint32_t coreNumAiv{0};
  uint32_t l2{0};
  uint32_t l1{0};
  uint32_t l0a{0};
  uint32_t l0b{0};
  uint32_t l0c{0};
  uint32_t ub{0};
  bool isRegBasedArch{false};
  uint32_t RegVectorLength{0};
};

inline const std::vector<std::pair<std::string, HardwareConfig>> &getHardwareConfigs() {
  static const std::vector<std::pair<std::string, HardwareConfig>> kHardwareConfigs = {
    {kSoc910B1, HardwareConfig{/* coreNumAic = */ 24,
                               /* coreNumAiv = */ 48,
                               /* l2 = */ 201326592,
                               /* l1 = */ 524288,
                               /* l0a = */ 65536,
                               /* l0b = */ 65536,
                               /* l0c = */ 131072,
                               /* ub = */ 196608,
                               /* isRegBasedArch = */ false}},
    {kSoc910B2, HardwareConfig{/* coreNumAic = */ 24,
                               /* coreNumAiv = */ 48,
                               /* l2 = */ 201326592,
                               /* l1 = */ 524288,
                               /* l0a = */ 65536,
                               /* l0b = */ 65536,
                               /* l0c = */ 131072,
                               /* ub = */ 196608,
                               /* isRegBasedArch = */ false}},
    {kSoc910B2C, HardwareConfig{/* coreNumAic = */ 24,
                                /* coreNumAiv = */ 48,
                                /* l2 = */ 201326592,
                                /* l1 = */ 524288,
                                /* l0a = */ 65536,
                                /* l0b = */ 65536,
                                /* l0c = */ 131072,
                                /* ub = */ 196608,
                                /* isRegBasedArch = */ false}},
    {kSoc910B3, HardwareConfig{/* coreNumAic = */ 20,
                               /* coreNumAiv = */ 40,
                               /* l2 = */ 201326592,
                               /* l1 = */ 524288,
                               /* l0a = */ 65536,
                               /* l0b = */ 65536,
                               /* l0c = */ 131072,
                               /* ub = */ 196608,
                               /* isRegBasedArch = */ false}},
    {kSoc910B4, HardwareConfig{/* coreNumAic = */ 20,
                               /* coreNumAiv = */ 40,
                               /* l2 = */ 100663296,
                               /* l1 = */ 524288,
                               /* l0a = */ 65536,
                               /* l0b = */ 65536,
                               /* l0c = */ 131072,
                               /* ub = */ 196608,
                               /* isRegBasedArch = */ false}},
    {kSoc910B4_1, HardwareConfig{/* coreNumAic = */ 20,
                                 /* coreNumAiv = */ 40,
                                 /* l2 = */ 176160768,
                                 /* l1 = */ 524288,
                                 /* l0a = */ 65536,
                                 /* l0b = */ 65536,
                                 /* l0c = */ 131072,
                                 /* ub = */ 196608,
                                 /* isRegBasedArch = */ false}},
    {kSoc950PR_9599, HardwareConfig{/* coreNumAic = */ 36,
                                    /* coreNumAiv = */ 72,
                                    /* l2 = */ 134217728,
                                    /* l1 = */ 524288,
                                    /* l0a = */ 65536,
                                    /* l0b = */ 65536,
                                    /* l0c = */ 262144,
                                    /* ub = */ 253952,
                                    /* isRegBasedArch = */ true,
                                    /* RegVectorLength = */ 256}},
  };
  return kHardwareConfigs;
}

class NpuInfo {
 public:
  NpuInfo(const NpuInfo &) = delete;
  NpuInfo &operator=(const NpuInfo &) = delete;
  ~NpuInfo() {}

  static NpuInfo &getInstance(const std::string &socVersion) {
    static NpuInfo hardware_info(socVersion);
    return hardware_info;
  }

  inline uint32_t getCoreNumAic() const { return hwConfig.coreNumAic; }
  inline uint32_t getCoreNumAiv() const { return hwConfig.coreNumAiv; }
  inline uint32_t getL2Size() const { return hwConfig.l2; }
  inline uint32_t getL1Size() const { return hwConfig.l1; }
  inline uint32_t getL0aSize() const { return hwConfig.l0a; }
  inline uint32_t getL0bSize() const { return hwConfig.l0b; }
  inline uint32_t getL0cSize() const { return hwConfig.l0c; }
  inline uint32_t getUbSize() const { return hwConfig.ub; }
  inline bool isRegBasedArch() const { return hwConfig.isRegBasedArch; }
  inline uint32_t getRegVectorLength() const { return hwConfig.RegVectorLength; }

  const HardwareConfig &getConfigByVersion(const std::string &socVersion) const {
    static HardwareConfig invalidHardwareConfig;
    const auto &configs = getHardwareConfigs();
    const auto it = std::find_if(
      configs.begin(), configs.end(),
      [&socVersion](const std::pair<std::string, HardwareConfig> &entry) { return entry.first == socVersion; });
    if (it == configs.end()) {
      llvm::errs() << "The config for socVersion: " << socVersion << " does not exist.";
      return invalidHardwareConfig;
    }

    return it->second;
  }

 private:
  explicit NpuInfo(const std::string &socVersion) { hwConfig = getConfigByVersion(socVersion); }

  HardwareConfig hwConfig;
};

inline int64_t ceilDivInt64(int64_t lhs, int64_t rhs) { return (rhs <= 0) ? lhs : (lhs + rhs - 1) / rhs; }

inline int64_t alignUpInt64(int64_t value, int64_t alignment) {
  if (alignment <= 1) {
    return value;
  }
  return ceilDivInt64(value, alignment) * alignment;
}

inline int64_t multiplyAndCap(int64_t lhs, int64_t rhs) {
  if (lhs <= 0 || rhs <= 0) {
    return 0;
  }
  return (lhs > LLONG_MAX / rhs) ? LLONG_MAX : lhs * rhs;
}

inline int64_t getElementBitWidth(Type type) {
  Type elemType = type;
  if (auto shapedType = dyn_cast<ShapedType>(type)) {
    elemType = shapedType.getElementType();
  }
  if (elemType.isIndex()) {
    return 64;
  }
  if (!llvm::isa<IntegerType, FloatType>(elemType)) {
    return 0;
  }
  return static_cast<int64_t>(elemType.getIntOrFloatBitWidth());
}

inline int64_t getBishengStrideAlignTargetForBits(int64_t elementBits) {
  int64_t bitWidth = std::max<int64_t>(elementBits, 1);
  if (bitWidth >= kNpuUbAlignBits || kNpuUbAlignBits % bitWidth != 0) {
    return 1;
  }
  return kNpuUbAlignBits / bitWidth;
}

inline SmallVector<int32_t, kSmallVectorSizeSix> getDefaultBishengStrideAlignDims(int64_t rank) {
  if (rank <= 1) {
    return {};
  }
  return {static_cast<int32_t>(rank <= 2 ? 0 : rank - 3)};
}

inline SmallVector<int32_t, kSmallVectorSizeSix> getBishengLogicalStructuredStrideAlignDims(int64_t rank) {
  if (rank <= 1) {
    return {};
  }
  // BiShengIR MarkStrideAlign picks the last non-continuous dim before the
  // innermost unit-stride dimension. collectAlignUnits then expands dim+1, so
  // a rank-N structured vector buffer aligns its logical innermost dimension.
  return {static_cast<int32_t>(rank - 2)};
}

inline SmallVector<int32_t, kSmallVectorSizeSix> getBishengStorageStrideAlignDims(ArrayRef<char> staticDims) {
  if (staticDims.size() <= 1) {
    return {};
  }
  // This entry is for an already materialized storage shape. When ArithToHIVM
  // rank-extends a VBrc source, the trailing size-1 broadcast axis becomes the
  // innermost unit-stride dimension, so the raw stride-align mark starts one dim
  // further out than the logical vector-axis rule above. Dynamic dims between
  // that mark and the innermost axis block BiShengIR's static accumulation, so
  // the mark moves inward to the last such dynamic dim.
  int32_t alignDim = static_cast<int32_t>(staticDims.size() <= 2 ? 0 : staticDims.size() - 3);
  for (int64_t dim = alignDim + 1, e = static_cast<int64_t>(staticDims.size()) - 1; dim < e; ++dim) {
    if (!staticDims[dim]) {
      alignDim = static_cast<int32_t>(dim);
    }
  }
  return {alignDim};
}

inline SmallVector<int32_t, kSmallVectorSizeSix> getBishengNpuVectorStrideAlignDims(int64_t rank, int64_t elementBits) {
  // Sub-byte masks keep BiShengIR's default storage mark behavior; byte-addressable
  // vector buffers follow the structured path used by MarkStrideAlign.
  if (elementBits < kNpuBitsPerByte) {
    return getDefaultBishengStrideAlignDims(rank);
  }
  return getBishengLogicalStructuredStrideAlignDims(rank);
}

inline SmallVector<int32_t, kSmallVectorSizeSix> getBishengInlineBroadcastSourceStrideAlignDims(int64_t rank) {
  if (rank <= 0) {
    return {};
  }
  // Inline broadcast can propagate stride-align from the expanded operand back
  // to a lower-rank source. For rank-1 this is the only path that marks dim 0.
  return {static_cast<int32_t>(rank <= 2 ? 0 : rank - 2)};
}

inline SmallVector<int64_t, kSmallVectorSizeSix> collectBishengStrideAlignUnits(ArrayRef<int64_t> shape,
                                                                                ArrayRef<char> staticDims,
                                                                                ArrayRef<int32_t> alignDims,
                                                                                int64_t elementBits) {
  SmallVector<int64_t, kSmallVectorSizeSix> alignUnits(shape.size() + 1, 1);
  if (shape.empty() || staticDims.size() != shape.size()) {
    return alignUnits;
  }

  int64_t unit = getBishengStrideAlignTargetForBits(elementBits);
  SmallVector<int64_t, kSmallVectorSizeSix> alignTargets(shape.size(), 1);
  if (unit > 1) {
    for (int32_t dim : alignDims) {
      if (dim < 0 || static_cast<size_t>(dim) >= shape.size()) {
        continue;
      }
      size_t idx = static_cast<size_t>(dim);
      alignTargets[idx] = std::lcm(alignTargets[idx], unit);
    }
  }

  int64_t innerAlignedUnits = 1;
  int64_t shapeAccumulation = 1;
  for (int64_t dim = static_cast<int64_t>(shape.size()) - 1; dim >= 0; --dim) {
    size_t idx = static_cast<size_t>(dim);
    int64_t newAlignedUnits = std::lcm(innerAlignedUnits, alignTargets[idx]);
    alignUnits[idx + 1] = (shapeAccumulation % newAlignedUnits == 0) ? 1 : newAlignedUnits / innerAlignedUnits;
    innerAlignedUnits = newAlignedUnits;
    if (staticDims[idx]) {
      shapeAccumulation = multiplyAndCap(
        shapeAccumulation, std::lcm(std::max<int64_t>(shape[idx], 1), std::max<int64_t>(alignUnits[idx + 1], 1)));
    }
  }
  return alignUnits;
}

inline int64_t computeBishengStrideAlignedStorageBytes(ArrayRef<int64_t> shape, ArrayRef<char> staticDims,
                                                       ArrayRef<int32_t> alignDims, int64_t elementBits) {
  SmallVector<int64_t, kSmallVectorSizeSix> alignUnits =
    collectBishengStrideAlignUnits(shape, staticDims, alignDims, elementBits);

  int64_t elems = 1;
  for (size_t i = 0; i < alignUnits.size(); ++i) {
    int64_t dim = i < shape.size() ? shape[i] : 1;
    elems = multiplyAndCap(elems, alignUpInt64(std::max<int64_t>(dim, 1), alignUnits[i]));
  }
  return alignUpInt64(ceilDivInt64(multiplyAndCap(elems, std::max<int64_t>(elementBits, 1)), kNpuBitsPerByte),
                      kNpuUbAlignBytes);
}

inline int64_t computeBishengAlignedShapeBytes(ArrayRef<int64_t> shape, ArrayRef<int64_t> alignBytes,
                                               int64_t elemBytes) {
  if (shape.empty() || shape.size() != alignBytes.size() || elemBytes <= 0) {
    return 0;
  }
  int64_t elems = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    int64_t alignUnit = alignBytes[i] > 0 ? std::max<int64_t>(alignBytes[i] / elemBytes, 1) : 1;
    elems = multiplyAndCap(elems, alignUpInt64(std::max<int64_t>(shape[i], 1), alignUnit));
  }
  return alignUpInt64(multiplyAndCap(elems, elemBytes), kNpuUbAlignBytes);
}

inline int64_t computeBishengLastDimTransposeBufferBytes(ArrayRef<int64_t> sourceMaxShape,
                                                         ArrayRef<int64_t> sourceTypeShape,
                                                         ArrayRef<int64_t> permutation, Type elemType,
                                                         bool resultBuffer) {
  int64_t elementBits = getElementBitWidth(elemType);
  if (sourceTypeShape.empty() || sourceMaxShape.size() != sourceTypeShape.size() ||
      permutation.size() != sourceTypeShape.size() || elementBits <= 0) {
    return 0;
  }

  SmallVector<size_t, kSmallVectorSizeTwo> transposeDims;
  for (size_t i = 0; i < permutation.size(); ++i) {
    if (permutation[i] != static_cast<int64_t>(i)) {
      transposeDims.push_back(i);
    }
  }
  if (transposeDims.size() != 2 ||
      (transposeDims[0] != sourceTypeShape.size() - 1 && transposeDims[1] != sourceTypeShape.size() - 1)) {
    return 0;
  }

  SmallVector<int64_t, kSmallVectorSizeSix> sourceShape;
  sourceShape.reserve(sourceTypeShape.size());
  for (size_t i = 0; i < sourceTypeShape.size(); ++i) {
    sourceShape.push_back(ShapedType::isDynamic(sourceTypeShape[i]) ? sourceMaxShape[i] : sourceTypeShape[i]);
  }

  int64_t elemBytes = std::max<int64_t>(ceilDivInt64(elementBits, kNpuBitsPerByte), 1);
  SmallVector<int64_t, kSmallVectorSizeSix> resultShape(sourceShape.begin(), sourceShape.end());
  std::swap(resultShape[transposeDims[0]], resultShape[transposeDims[1]]);

  SmallVector<int64_t, kSmallVectorSizeSix> sourceAlign(sourceShape.size(), 0);
  SmallVector<int64_t, kSmallVectorSizeSix> resultAlign(sourceShape.size(), 0);
  for (size_t dim : transposeDims) {
    if (multiplyAndCap(sourceShape[dim], elemBytes) % kNpuUbAlignBytes != 0) {
      sourceAlign[dim] = kNpuUbAlignBytes;
    }
    if (multiplyAndCap(resultShape[dim], elemBytes) % kNpuUbAlignBytes != 0) {
      resultAlign[dim] = kNpuUbAlignBytes;
    }
  }

  int64_t alignedDim0Bytes = alignUpInt64(multiplyAndCap(sourceShape[transposeDims[0]], elemBytes), kNpuUbAlignBytes);
  int64_t alignedDim1Bytes = alignUpInt64(multiplyAndCap(sourceShape[transposeDims[1]], elemBytes), kNpuUbAlignBytes);
  bool hasDoubleAlignedDim =
    alignedDim0Bytes % (kNpuUbAlignBytes * 2) == 0 || alignedDim1Bytes % (kNpuUbAlignBytes * 2) == 0;
  // B32 (4-byte) types need double alignment on transpose, matching bishengir's
  // getUnAlignSizeInfo(VTransposeOp) which keys on elemTypeBytes == 4 (covers
  // both f32 and i32, not just f32).
  if (elemBytes == 4 && !hasDoubleAlignedDim) {
    sourceAlign[transposeDims[0]] = kNpuUbAlignBytes;
    sourceAlign[transposeDims[1]] = kNpuUbAlignBytes * 2;
    resultAlign[transposeDims[0]] = kNpuUbAlignBytes * 2;
    resultAlign[transposeDims[1]] = kNpuUbAlignBytes;
  }

  return resultBuffer ? computeBishengAlignedShapeBytes(resultShape, resultAlign, elemBytes)
                      : computeBishengAlignedShapeBytes(sourceShape, sourceAlign, elemBytes);
}

inline int64_t computeBishengStrideAlignedStorageBytesWithTrailingUnit(ArrayRef<int64_t> shape,
                                                                       ArrayRef<char> staticDims,
                                                                       ArrayRef<int32_t> alignDims,
                                                                       int64_t elementBits) {
  if (shape.empty() || staticDims.size() != shape.size() || elementBits <= 0) {
    return 0;
  }

  SmallVector<int64_t, kSmallVectorSizeSix> storageShape(shape.begin(), shape.end());
  SmallVector<char, kSmallVectorSizeSix> storageStaticDims(staticDims.begin(), staticDims.end());
  // BiShengIR EnableStrideAlign materializes aligned storage with a trailing unit dimension.
  storageShape.push_back(1);
  storageStaticDims.push_back(true);
  return computeBishengStrideAlignedStorageBytes(storageShape, storageStaticDims, alignDims, elementBits);
}

inline int64_t computeBishengStrideAlignedStorageBytes(ArrayRef<int64_t> shape, ArrayRef<int64_t> typeShape,
                                                       Type elemType) {
  int64_t elementBits = getElementBitWidth(elemType);
  if (shape.empty() || elementBits <= 0) {
    return 0;
  }

  SmallVector<char, kSmallVectorSizeSix> staticDims;
  staticDims.reserve(shape.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    staticDims.push_back(i < typeShape.size() && !ShapedType::isDynamic(typeShape[i]));
  }
  return computeBishengStrideAlignedStorageBytes(shape, staticDims, getBishengStorageStrideAlignDims(staticDims),
                                                 elementBits);
}

inline int64_t computeBishengNpuVectorStorageBytes(ArrayRef<int64_t> maxPerRankDim, ArrayRef<int64_t> typeShape,
                                                   Type elemType) {
  int64_t elementBits = getElementBitWidth(elemType);
  if (typeShape.empty() || maxPerRankDim.size() != typeShape.size() || elementBits <= 0) {
    return 0;
  }

  SmallVector<int64_t, kSmallVectorSizeSix> shape;
  SmallVector<char, kSmallVectorSizeSix> staticDims;
  shape.reserve(typeShape.size());
  staticDims.reserve(typeShape.size());
  for (size_t i = 0; i < typeShape.size(); ++i) {
    bool isDynamic = ShapedType::isDynamic(typeShape[i]);
    shape.push_back(isDynamic ? maxPerRankDim[i] : typeShape[i]);
    staticDims.push_back(!isDynamic);
  }
  return computeBishengStrideAlignedStorageBytesWithTrailingUnit(
    shape, staticDims, getBishengNpuVectorStrideAlignDims(static_cast<int64_t>(typeShape.size()), elementBits),
    elementBits);
}

inline int64_t computeBishengRank1StrideAlignedStorageBytes(ArrayRef<int64_t> maxPerRankDim,
                                                            ArrayRef<int64_t> typeShape, Type elemType) {
  int64_t elementBits = getElementBitWidth(elemType);
  if (typeShape.size() != 1 || maxPerRankDim.size() != 1 || elementBits <= 0) {
    return 0;
  }

  int64_t dim = ShapedType::isDynamic(typeShape.front()) ? maxPerRankDim.front() : typeShape.front();
  SmallVector<int64_t, kSmallVectorSizeOne> shape{dim};
  SmallVector<char, kSmallVectorSizeOne> staticDims{static_cast<char>(!ShapedType::isDynamic(typeShape.front()))};
  SmallVector<int32_t, kSmallVectorSizeOne> alignDims{0};
  return computeBishengStrideAlignedStorageBytesWithTrailingUnit(shape, staticDims, alignDims, elementBits);
}

inline int64_t computeBishengInlineBroadcastSourceStorageBytes(ArrayRef<int64_t> maxPerRankDim,
                                                               ArrayRef<int64_t> typeShape, Type elemType) {
  int64_t elementBits = getElementBitWidth(elemType);
  if (typeShape.empty() || maxPerRankDim.size() != typeShape.size() || elementBits <= 0) {
    return 0;
  }

  SmallVector<int64_t, kSmallVectorSizeSix> shape;
  SmallVector<char, kSmallVectorSizeSix> staticDims;
  shape.reserve(typeShape.size());
  staticDims.reserve(typeShape.size());
  for (size_t i = 0; i < typeShape.size(); ++i) {
    bool isDynamic = ShapedType::isDynamic(typeShape[i]);
    shape.push_back(isDynamic ? maxPerRankDim[i] : typeShape[i]);
    staticDims.push_back(!isDynamic);
  }
  return computeBishengStrideAlignedStorageBytesWithTrailingUnit(
    shape, staticDims, getBishengInlineBroadcastSourceStrideAlignDims(static_cast<int64_t>(typeShape.size())),
    elementBits);
}

inline int64_t computeBishengStructuredNpuVectorStorageBytes(ArrayRef<int64_t> maxPerRankDim,
                                                             ArrayRef<int64_t> typeShape, Type elemType) {
  return computeBishengNpuVectorStorageBytes(maxPerRankDim, typeShape, elemType);
}

}  // namespace akg
}  // namespace mlir
#endif  // AKG_UTILS_ANALYSISFORNPU_H
