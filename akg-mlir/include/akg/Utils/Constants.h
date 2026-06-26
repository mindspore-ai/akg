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

#ifndef AKG_UTILS_CONSTANTS_H_
#define AKG_UTILS_CONSTANTS_H_

namespace mlir {

inline constexpr unsigned kBoolBitWidth = 1;
inline constexpr unsigned kI8BitWidth = 8;
inline constexpr unsigned kI16BitWidth = 16;
inline constexpr unsigned kI32BitWidth = 32;
inline constexpr unsigned kI64BitWidth = 64;
inline constexpr unsigned kBitsPerByte = 8;
inline constexpr unsigned kBytesPerKb = 1024;

inline constexpr unsigned kBinaryOpOperandCount = 2;
inline constexpr unsigned kUnaryOpOperandCount = 1;
inline constexpr unsigned kTernaryOpOperandCount = 3;

inline constexpr unsigned kSmallVectorSizeZero = 0;
inline constexpr unsigned kSmallVectorSizeOne = 1;
inline constexpr unsigned kSmallVectorSizeTwo = 2;
inline constexpr unsigned kSmallVectorSizeThree = 3;
inline constexpr unsigned kSmallVectorSizeFour = 4;
inline constexpr unsigned kSmallVectorSizeFive = 5;
inline constexpr unsigned kSmallVectorSizeSix = 6;
inline constexpr unsigned kSmallVectorSizeEight = 8;
inline constexpr unsigned kSmallVectorSizeTwelve = 12;
inline constexpr unsigned kSmallVectorSizeSixteen = 16;
inline constexpr unsigned kSmallVectorSizeThirtyTwo = 32;
inline constexpr unsigned kSmallVectorSizeSixtyFour = 64;
inline constexpr unsigned kSmallVectorSizeOneHundredTwentyEight = 128;

}  // namespace mlir

#endif  // AKG_UTILS_CONSTANTS_H_
