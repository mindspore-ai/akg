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

#ifndef MFUSION_DIALECT_MFUSE_UTILS_OP_CONSTANTS_H
#define MFUSION_DIALECT_MFUSE_UTILS_OP_CONSTANTS_H

#include <cstddef>

namespace mlir {
namespace mfuse {

// index of input or output
constexpr size_t kIndex0 = 0;
constexpr size_t kIndex1 = 1;
constexpr size_t kIndex2 = 2;
constexpr size_t kIndex3 = 3;
constexpr size_t kIndex4 = 4;
constexpr size_t kIndex5 = 5;
constexpr size_t kIndex6 = 6;
constexpr size_t kIndex7 = 7;
constexpr size_t kIndex8 = 8;
constexpr size_t kIndex9 = 9;
constexpr size_t kIndex10 = 10;
constexpr size_t kIndex11 = 11;
constexpr size_t kIndex12 = 12;
constexpr size_t kIndex13 = 13;
constexpr size_t kIndex14 = 14;
constexpr size_t kIndex15 = 15;
constexpr size_t kIndex16 = 16;
constexpr size_t kIndex17 = 17;
constexpr size_t kIndex18 = 18;
constexpr size_t kIndex19 = 19;
constexpr size_t kIndex20 = 20;
constexpr size_t kIndex21 = 21;
constexpr size_t kIndex22 = 22;
constexpr size_t kIndex23 = 23;
constexpr size_t kIndex24 = 24;
constexpr size_t kIndex25 = 25;
constexpr size_t kIndex26 = 26;
constexpr size_t kIndex27 = 27;
constexpr size_t kIndex28 = 28;
constexpr size_t kIndex29 = 29;
constexpr size_t kIndex30 = 30;
constexpr size_t kIndex31 = 31;
constexpr size_t kIndex32 = 32;
constexpr size_t kIndex33 = 33;
constexpr size_t kIndex34 = 34;
constexpr size_t kIndex35 = 35;
constexpr size_t kIndex36 = 36;
constexpr size_t kIndex37 = 37;

// dim of shape
constexpr size_t kDim0 = 0;
constexpr size_t kDim1 = 1;
constexpr size_t kDim2 = 2;
constexpr size_t kDim3 = 3;
constexpr size_t kDim4 = 4;
constexpr size_t kDim5 = 5;
constexpr size_t kDim6 = 6;

// output size of op
constexpr size_t kOutputSize1 = 1;
constexpr size_t kOutputSize2 = 2;
constexpr size_t kOutputSize3 = 3;

// Number of input
constexpr size_t kInputNum0 = 0;
constexpr size_t kInputNum1 = 1;
constexpr size_t kInputNum2 = 2;
constexpr size_t kInputNum3 = 3;
constexpr size_t kInputNum4 = 4;
constexpr size_t kInputNum5 = 5;

// Split processor string
constexpr char kProcessorDVM[] = "DVM";
constexpr char kProcessorAKG[] = "AKGAscend";
constexpr char kProcessorASCENDNPUIR[] = "AscendNPUIR";

}  // namespace mfuse
}  // namespace mlir

#endif  // MFUSION_DIALECT_MFUSE_UTILS_OP_CONSTANTS_H
