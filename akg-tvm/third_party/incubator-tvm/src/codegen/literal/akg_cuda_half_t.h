/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * 2021.7.9 - Add asin, acos, asinh, acosh.
 */

#ifndef AKG_CODEGEN_LITERAL_CUDA_HALF_T_H_
#define AKG_CODEGEN_LITERAL_CUDA_HALF_T_H_

static constexpr const char* _cuda_half_util = R"(
static int32_t const inff = 0x7F800000; // flt32 infinity
// Pack two half value.
static inline __device__ __host__ unsigned
__pack_half2(const half x, const half y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}
static inline __device__ __host__ half hpow(half x, half y) {
  float tmp_x = __half2float(x);
  float tmp_y = __half2float(y);
  float result = powf(tmp_x, tmp_y);
  return __float2half(result);
}
static inline __device__ __host__ half htanh(half x) {
  float tmp_x = __half2float(x);
  float result = tanhf(tmp_x);
  return __float2half(result);
}
static inline __device__ __host__ half hasin(half x) {
  float tmp_x = __half2float(x);
  float result = asinf(tmp_x);
  return __float2half(result);
}
static inline __device__ __host__ half hacos(half x) {
  float tmp_x = __half2float(x);
  float result = acosf(tmp_x);
  return __float2half(result);
}
static inline __device__ __host__ half hfabs(half x) {
  if (x < __float2half(0.0)) return -x;
  return x;
}
static inline __device__ __host__ half hasinh(half x) {
  float tmp_x = __half2float(x);
  float result = asinhf(tmp_x);
  return __float2half(result);
}
static inline __device__ __host__ half hacosh(half x) {
  float tmp_x = __half2float(x);
  float result = acoshf(tmp_x);
  return __float2half(result);
}
static inline __device__ __host__ half hatan(half x) {
  float tmp_x = __half2float(x);
  float result = atan(tmp_x);
  return __float2half(result);
}
static inline __device__ __host__ half hatan2(half lhs, half rhs) {
  float l = __half2float(lhs);
  float r = __half2float(rhs);
  float result = atan2(l, r);
  return __float2half(result);
}
static inline __device__ __host__ half hexpm1(half x) {
  float tmp_x = __half2float(x);
  float result = expm1(tmp_x);
  return __float2half(result);
}
static inline __device__ __host__ half fmod(half lhs, half rhs) {
  float l = __half2float(lhs);
  float r = __half2float(rhs);
  float data_div = l / r;
  float data_div_min = data_div < 0.0 ? data_div : 0.0;
  float data_div_max = data_div > 0.0 ? data_div : 0.0;
  float data_div_max_floor = floorf(data_div_max);
  float data_div_min_ceil = ceilf(data_div_min);
  float data_div_res = data_div_max_floor + data_div_min_ceil;
  return __float2half_rn(l - data_div_res * r);
}
static inline __device__ __host__ half herf(half x) {
  half result = erff(x);
  return result;
}
)";

#endif  // AKG_CODEGEN_LITERAL_CUDA_HALF_T_H_
