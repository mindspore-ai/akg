/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef AKG_REDUCE_UTIL_H
#define AKG_REDUCE_UTIL_H
#include <iostream>
#include <cuda_fp16.h>
#include <string.h>

namespace akg_reduce {

const int ALL_REDUCE = 0;
const int REDUCE2D_X = 1;
const int REDUCE2D_Y = 2;
const int WARPSIZE = 32;

// Error detection functions
#ifndef GetGpuErr
#define GetGpuErr(e) \
  { GpuAssert((e), __FILE__, __LINE__); }
#endif

inline void GpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GET A GPU ERROR: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

// Select the Type by if condition
template <bool IF, typename TrueType, typename FalseType>
struct Select {
  typedef TrueType Type;
};

template <typename TrueType, typename FalseType>
struct Select<false, TrueType, FalseType> {
  typedef FalseType Type;
};

/**
 * @brief Type transform function for test cases
 *
 * @tparam T Target dtype
 * @tparam Y Original dtype
 * @param y  Original value
 * @return   value with target type
 */
template <typename T, typename Y>
__host__ __device__ T TypeTransform(Y y) {
  if (sizeof(T) == 2) {
    return __float2half((double)y);
  } else if (sizeof(Y) == 2) {
    return (T)(__half2float(y));
  } else
    return (T)y;
}

__host__ __device__ constexpr bool IsPowOfTwo(const unsigned int num) { 
  return !(num & (num - 1)); 
}

__host__ __device__ constexpr int GetUpperBound(const int length) {
  int upper_bound = 1;
  while (upper_bound * 2 <= length) {
    upper_bound *= 2;
  }
  return upper_bound;
}

}  // namespace akg_reduce

#endif  // AKG_REDUCE_UTIL_H