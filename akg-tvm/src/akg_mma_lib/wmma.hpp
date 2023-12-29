/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

/**
 * WMMA API Extension
 * CUDA provides an experimental PTX instruction mma.m8n8k4 which compute matrix FMA use Tensor Core
 * See detail: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions
 * This extension provides its C++ interface
 *
 * Sample:
 * #include "wmma.hpp"
 * __global__ void wmma_kernel(float *c_ptr, const half *a_ptr, const half *b_ptr) {
 *   akg::wmma::fragment<nvcuda::wmma::matrix_a, M16, N16, K4, half, nvcuda::wmma::col_major> frag_a;
 *   akg::wmma::fragment<nvcuda::wmma::matrix_b, M16, N16, K4, half, nvcuda::wmma::row_major> frag_b;
 *   akg::wmma::fragment<nvcuda::wmma::accumulator, M16, N16, K4, float> frag_c;
 *   akg::wmma::fragment<nvcuda::wmma::accumulator, M16, N16, K4, float> frag_d;
 *
 *   akg::wmma::fill_fragment(frag_c, 0.0f);
 *   akg::wmma::load_matrix_sync(frag_a, a_ptr, 16);
 *   akg::wmma::load_matrix_sync(frag_b, b_ptr, 16);
 *
 *   akg::wmma::mma_sync(frag_d, frag_a, frag_b, frag_c);
 *   akg::wmma::store_matrix_sync(c_ptr, frag_d, N, nvcuda::wmma::mem_row_major);
 * }
 */

#ifndef __WMMA_HPP__
#define __WMMA_HPP__
#include <mma.h>
#include <cuda_fp16.h>

namespace akg {
namespace wmma {

constexpr int M32 = 32;
constexpr int M16 = 16;
constexpr int N32 = 32;
constexpr int N16 = 16;
constexpr int K8 = 8;
constexpr int K4 = 4;
constexpr int L32 = 32;
constexpr int L8 = 8;

template <class T>
class Vector2;

template <>
class Vector2<half> {
 public:
  using Vector2Type = half2;
};

template <>
class Vector2<float> {
 public:
  using Vector2Type = float2;
};

template <bool, class T1, class T2>
class CastTypeFun;

template <class T1, class T2>
class CastTypeFun<true, T1, T2> {
 public:
  using CastType = T1;
};

template <class T1, class T2>
class CastTypeFun<false, T1, T2> {
 public:
  using CastType = T2;
};

template <int k, class T1, class T2>
class CastValueType {
 public:
  using CastType = typename CastTypeFun<(k % L8 == 0), T1, T2>::CastType;
};

template <class T>
inline __device__ T cast(const float src) {
  return src;
}

template <>
inline __device__ half cast(const float src) {
  return __float2half_rn(src);
}

template <class T>
inline __device__ T cast(const float2 src) {
  return src;
}

template <>
inline __device__ half2 cast(const float2 src) {
  return __float22half2_rn(src);
}

template <class T>
inline __device__ T cast(const half src) {
  return src;
}

template <>
inline __device__ float cast(const half src) {
  return __half2float(src);
}

template <class T>
inline __device__ T cast(const half2 src) {
  return src;
}

template <>
inline __device__ float2 cast(const half2 src) {
  return __half22float2(src);
}

inline __device__ unsigned get_lane_id() {
  unsigned lane_id;
  asm volatile (R"({mov.s32 %0, %laneid;})" : "=r"(lane_id));
  return lane_id;
}

template <class T, int size>
struct __align__(4) __frag_base {
  T x[size];
  enum { num_elements = size };
};

template <class Use, int m, int n, int k, class T, class Layout = void>
class fragment;
template <class Use, int k, class T, class Layout>
class fragment<Use, M16, N16, k, T, Layout> : public __frag_base<T, k> {};
template <int k, class T>
class fragment<nvcuda::wmma::accumulator, M16, N16, k, T> : public __frag_base<T, L8> {};
template <>
class fragment<nvcuda::wmma::matrix_a, M32, N32, K4, half, nvcuda::wmma::col_major> : public __frag_base<half, L8> {};
template <>
class fragment<nvcuda::wmma::matrix_b, M32, N32, K4, half, nvcuda::wmma::row_major> : public __frag_base<half, L8> {};
template <>
class fragment<nvcuda::wmma::accumulator, M32, N32, K4, float> : public __frag_base<float, L32> {};

template <class T, class S, int size>
__device__ inline void fill_fragment(__frag_base<T, size> &f, const S v) {
#pragma unroll
  for (unsigned i = 0; i < f.num_elements; i++) {
    f.x[i] = cast<T>(v);
  }
}

template <int k, class T>
__device__ inline void load_matrix_sync(
    fragment<nvcuda::wmma::matrix_a, M16, N16, k, half, nvcuda::wmma::col_major> &f,
    const T *const p, const unsigned ldm) {
  const unsigned lane_id = get_lane_id();
  const unsigned row = lane_id & 0x3;
  const unsigned col = (lane_id & 0x4) + ((lane_id >> 4) << 3);
  const unsigned offset = row * ldm + col;
  using Type = typename CastValueType<k, float2, float2>::CastType;
  Type *src = (Type *)(p + offset);
  Type *dst = (Type *)f.x;
#pragma unroll
  for (int i = 0; i < k / 4; i++) {
    dst[i] = src[i * ldm];
  }
}

template <class T>
__device__ inline void load_matrix_sync(
    fragment<nvcuda::wmma::matrix_a, M32, N32, K4, half, nvcuda::wmma::col_major> &f,
    const T *const p, const unsigned ldm) {
  const unsigned lane_id = get_lane_id();
  const unsigned row = lane_id & 0x3;
  const unsigned col = (lane_id & 0x8) + (lane_id & 0x10);
  const unsigned offset = row * ldm + col;

  using Type = float4;
  Type *src = (Type *)(p + offset);
  Type *dst = (Type *)f.x;
  dst[0] = src[0];
}

template <int k, class T>
__device__ inline void load_matrix_sync(
    fragment<nvcuda::wmma::matrix_a, M16, N16, k, half, nvcuda::wmma::row_major> &f,
    const T *const p, const unsigned ldm) {
  const unsigned lane_id = get_lane_id();
  const unsigned row = (lane_id & 0x7) + ((lane_id >> 4) << 3);
  const unsigned offset = row * ldm;

  using Type = typename CastValueType<k, float4, float2>::CastType;
  Type *src = (Type *)(p + offset);
  Type *dst = (Type *)f.x;
  dst[0] = src[0];
}

template <int k, class T>
__device__ inline void load_matrix_sync(
    fragment<nvcuda::wmma::matrix_b, M16, N16, k, half, nvcuda::wmma::col_major> &f,
    const T *const p, const unsigned ldm) {
  const unsigned lane_id = get_lane_id();
  const unsigned row = (lane_id & 0x3) + ((lane_id & 0x18) >> 1);
  const unsigned offset = row * ldm;

  using Type = typename CastValueType<k, float4, float2>::CastType;
  Type *src = (Type *)(p + offset);
  Type *dst = (Type *)f.x;
  dst[0] = src[0];
}

template <int k, class T>
__device__ inline void load_matrix_sync(
    fragment<nvcuda::wmma::matrix_b, M16, N16, k, half, nvcuda::wmma::row_major> &f,
    const T *const p, const unsigned ldm) {
  const unsigned lane_id = get_lane_id();
  const unsigned row = lane_id & 0x3;
  const unsigned col = (lane_id >> 3) << 2;
  const unsigned offset = row * ldm + col;
  using Type = typename CastValueType<k, float2, float2>::CastType;
  Type *src = (Type *)(p + offset);
  Type *dst = (Type *)f.x;
#pragma unroll
  for (int i = 0; i < k / 4; i++) {
    dst[i] = src[i * ldm];
  }
}

template <class T>
__device__ inline void load_matrix_sync(fragment<nvcuda::wmma::matrix_b, M32, N32, K4, half,
                                        nvcuda::wmma::row_major> &f, const T *const p, const unsigned ldm) {
  const unsigned lane_id = get_lane_id();
  const unsigned row = lane_id & 0x3;
  const unsigned col = ((lane_id & 0x4) << 1) + (lane_id & 0x10);
  const unsigned offset = row * ldm + col;

  using Type = float4;
  Type *src = (Type *)(p + offset);
  Type *dst = (Type *)f.x;
  dst[0] = src[0];
}

template <int k, class T>
__device__ inline void load_matrix_sync(fragment<nvcuda::wmma::accumulator, M16, N16, k, half, void> &f,
                                        const T *const p, const unsigned ldm, const nvcuda::wmma::layout_t layout) {
  const unsigned lane_id = get_lane_id();
  const unsigned row = (lane_id & 0x7) + ((lane_id >> 4) << 3);
  const unsigned col = ((lane_id & 0xf) >> 3) << 2;
  if (layout == nvcuda::wmma::mem_col_major) {
    const int offset = col * ldm + row;
    f.x[0] = cast<half>(p[offset]);
    f.x[1] = cast<half>(p[offset + ldm]);
    f.x[2] = cast<half>(p[offset + 2 * ldm]);
    f.x[3] = cast<half>(p[offset + 3 + ldm]);
    f.x[4] = cast<half>(p[offset + 8 * ldm]);
    f.x[5] = cast<half>(p[offset + 9 * ldm]);
    f.x[6] = cast<half>(p[offset + 10 * ldm]);
    f.x[7] = cast<half>(p[offset + 11 * ldm]);
  } else {
    const int offset = row * ldm + col;
    float2 *src = (float2 *)(p + offset);
    float2 *dst = (float2 *)f.x;
    dst[0] = src[0];
    dst[1] = src[2];
  }
}

template <int k, class T>
__device__ inline void load_matrix_sync(fragment<nvcuda::wmma::accumulator, M16, N16, k, float, void> &f,
                                        const T *const p, const unsigned ldm, const nvcuda::wmma::layout_t layout) {
  const unsigned lane_id = get_lane_id();
  const unsigned row = (lane_id & 0x5) + ((lane_id >> 4) << 3);
  const unsigned col = ((lane_id & 0x2)) + ((lane_id & 0x8) >> 1);
  if (layout == nvcuda::wmma::mem_col_major) {
    const int offset = col * ldm + row;
    f.x[0] = cast<float>(p[offset]);
    f.x[1] = cast<float>(p[offset + ldm]);
    f.x[2] = cast<float>(p[offset + 2]);
    f.x[3] = cast<float>(p[offset + 2 + ldm]);
    f.x[4] = cast<float>(p[offset + 8 * ldm]);
    f.x[5] = cast<float>(p[offset + 9 * ldm]);
    f.x[6] = cast<float>(p[offset + 2 + 8 * ldm]);
    f.x[7] = cast<float>(p[offset + 2 + 9 * ldm]);
  } else {
    using SrcType = typename Vector2<T>::Vector2Type;
    const int offset = row * ldm + col;
    SrcType *src = (SrcType *)(p + offset);
    float2 *dst = (float2 *)f.x;
    dst[0] = cast<float2>(src[0]);
    dst[1] = cast<float2>(src[ldm]);
    dst[2] = cast<float2>(src[4]);
    dst[3] = cast<float2>(src[ldm + 4]);
  }
}

template <class T>
__device__ inline void load_matrix_sync(fragment<nvcuda::wmma::accumulator, M32, N32, K4, float, void> &f,
                                        const T *const p, const unsigned ldm, const nvcuda::wmma::layout_t layout) {
  const unsigned lane_id = get_lane_id();
  const unsigned row = (lane_id & 0x1) + (lane_id & 0x18);
  const unsigned col = ((lane_id & 0x2)) + ((lane_id & 0x4) << 1);

  if (layout == nvcuda::wmma::mem_col_major) {
    const int offset = col * ldm + row;
    f.x[0] = cast<float>(p[offset + 0]);
    f.x[1] = cast<float>(p[offset + ldm]);
    f.x[2] = cast<float>(p[offset + 2]);
    f.x[3] = cast<float>(p[offset + 2 + ldm]);
    f.x[4] = cast<float>(p[offset + 16 * ldm]);
    f.x[5] = cast<float>(p[offset + 17 * ldm]);
    f.x[6] = cast<float>(p[offset + 16 * ldm + 2]);
    f.x[7] = cast<float>(p[offset + 17 * ldm + 2]);
    f.x[8] = cast<float>(p[offset + 4 * ldm]);
    f.x[9] = cast<float>(p[offset + 5 * ldm]);
    f.x[10] = cast<float>(p[offset + 4 * ldm + 2]);
    f.x[11] = cast<float>(p[offset + 5 * ldm + 2]);
    f.x[12] = cast<float>(p[offset + 20 * ldm]);
    f.x[13] = cast<float>(p[offset + 21 * ldm]);
    f.x[14] = cast<float>(p[offset + 20 * ldm + 2]);
    f.x[15] = cast<float>(p[offset + 21 * ldm + 2]);
    f.x[16] = cast<float>(p[offset + 4]);
    f.x[17] = cast<float>(p[offset + ldm + 4]);
    f.x[18] = cast<float>(p[offset + 6]);
    f.x[19] = cast<float>(p[offset + ldm + 6]);
    f.x[20] = cast<float>(p[offset + 16 * ldm + 4]);
    f.x[21] = cast<float>(p[offset + 17 * ldm + 4]);
    f.x[22] = cast<float>(p[offset + 16 * ldm + 6]);
    f.x[23] = cast<float>(p[offset + 17 * ldm + 6]);
    f.x[24] = cast<float>(p[offset + 4 * ldm + 4]);
    f.x[25] = cast<float>(p[offset + 5 * ldm + 4]);
    f.x[26] = cast<float>(p[offset + 4 * ldm + 6]);
    f.x[27] = cast<float>(p[offset + 5 * ldm + 6]);
    f.x[28] = cast<float>(p[offset + 20 * ldm + 4]);
    f.x[29] = cast<float>(p[offset + 21 * ldm + 4]);
    f.x[30] = cast<float>(p[offset + 20 * ldm + 6]);
    f.x[31] = cast<float>(p[offset + 21 * ldm + 6]);
  } else {
    using SrcType = typename Vector2<T>::Vector2Type;
    const int offset = row * ldm + col;
    SrcType *src = (SrcType *)(p + offset);
    float2 *dst = (float2 *)f.x;
    dst[0] = cast<float2>(src[0]);
    dst[ldm] = cast<float2>(src[1]);
    dst[8] = cast<float2>(src[2]);
    dst[ldm + 8] = cast<float2>(src[3]);
    dst[2] = cast<float2>(src[4]);
    dst[ldm + 2] = cast<float2>(src[5]);
    dst[10] = cast<float2>(src[6]);
    dst[ldm + 10] = cast<float2>(src[7]);
    dst[2 * ldm] = cast<float2>(src[8]);
    dst[3 * ldm] = cast<float2>(src[9]);
    dst[2 * ldm + 8] = cast<float2>(src[10]);
    dst[3 * ldm + 8] = cast<float2>(src[11]);
    dst[2 * ldm + 2] = cast<float2>(src[12]);
    dst[3 * ldm + 2] = cast<float2>(src[13]);
    dst[2 * ldm + 10] = cast<float2>(src[14]);
    dst[3 * ldm + 10] = cast<float2>(src[15]);
  }
}

template <int k, class T>
__device__ inline void store_matrix_sync(T *const p,
                                         const fragment<nvcuda::wmma::accumulator, M16, N16, k, half, void> &f,
                                         const unsigned ldm, const nvcuda::wmma::layout_t layout) {
  const unsigned lane_id = get_lane_id();
  const unsigned row = (lane_id & 0x7) + ((lane_id >> 4) << 3);
  const unsigned col = ((lane_id & 0xf) >> 3) << 2;
  if (layout == nvcuda::wmma::mem_col_major) {
    const int offset = col * ldm + row;
    p[offset + 0] = cast<T>(f.x[0]);
    p[offset + ldm] = cast<T>(f.x[1]);
    p[offset + 2 * ldm] = cast<T>(f.x[2]);
    p[offset + 3 * ldm] = cast<T>(f.x[3]);
    p[offset + 8 * ldm] = cast<T>(f.x[4]);
    p[offset + 9 * ldm] = cast<T>(f.x[5]);
    p[offset + 10 * ldm] = cast<T>(f.x[6]);
    p[offset + 11 * ldm] = cast<T>(f.x[7]);
  } else {
    const int offset = row * ldm + col;
    float2 *dst = (float2 *)(p + offset);
    float2 *src = (float2 *)f.x;
    dst[0] = src[0];
    dst[2] = src[1];
  }
}

template <int k, class T>
__device__ inline void store_matrix_sync(T *const p, fragment<nvcuda::wmma::accumulator, M16, N16, k, float, void> &f,
                                         const unsigned ldm, const nvcuda::wmma::layout_t layout) {
  const unsigned lane_id = get_lane_id();
  const unsigned row = (lane_id & 0x5) + ((lane_id >> 4) << 3);
  const unsigned col = ((lane_id & 0x2)) + ((lane_id & 0x8) >> 1);

  if (layout == nvcuda::wmma::mem_col_major) {
    const int offset = col * ldm + row;
    p[offset + 0] = cast<T>(f.x[0]);
    p[offset + ldm] = cast<T>(f.x[1]);
    p[offset + 2] = cast<T>(f.x[2]);
    p[offset + 2 + ldm] = cast<T>(f.x[3]);
    p[offset + 8 * ldm] = cast<T>(f.x[4]);
    p[offset + 9 * ldm] = cast<T>(f.x[5]);
    p[offset + 2 + 8 * ldm] = cast<T>(f.x[6]);
    p[offset + 2 + 9 * ldm] = cast<T>(f.x[7]);
  } else {
    using DstType = typename Vector2<T>::Vector2Type;
    const int offset = row * ldm + col;
    DstType *dst = (DstType *)(p + offset);
    float2 *src = (float2 *)f.x;
    dst[0] = cast<DstType>(src[0]);
    dst[ldm] = cast<DstType>(src[1]);
    dst[4] = cast<DstType>(src[2]);
    dst[ldm + 4] = cast<DstType>(src[3]);
  }
}

template <class T>
__device__ inline void store_matrix_sync(T *const p, fragment<nvcuda::wmma::accumulator, M32, N32, K4, float, void> &f,
                                         const unsigned ldm, const nvcuda::wmma::layout_t layout) {
  const unsigned lane_id = get_lane_id();
  const unsigned row = (lane_id & 0x1) + (lane_id & 0x18);
  const unsigned col = ((lane_id & 0x2)) + ((lane_id & 0x4) << 1);

  if (layout == nvcuda::wmma::mem_col_major) {
    const int offset = col * ldm + row;
    p[offset + 0] = cast<T>(f.x[0]);
    p[offset + ldm] = cast<T>(f.x[1]);
    p[offset + 2] = cast<T>(f.x[2]);
    p[offset + 2 + ldm] = cast<T>(f.x[3]);
    p[offset + 16 * ldm] = cast<T>(f.x[4]);
    p[offset + 17 * ldm] = cast<T>(f.x[5]);
    p[offset + 16 * ldm + 2] = cast<T>(f.x[6]);
    p[offset + 17 * ldm + 2] = cast<T>(f.x[7]);
    p[offset + 4 * ldm] = cast<T>(f.x[8]);
    p[offset + 5 * ldm] = cast<T>(f.x[9]);
    p[offset + 4 * ldm + 2] = cast<T>(f.x[10]);
    p[offset + 5 * ldm + 2] = cast<T>(f.x[11]);
    p[offset + 20 * ldm] = cast<T>(f.x[12]);
    p[offset + 21 * ldm] = cast<T>(f.x[13]);
    p[offset + 20 * ldm + 2] = cast<T>(f.x[14]);
    p[offset + 21 * ldm + 2] = cast<T>(f.x[15]);
    p[offset + 4] = cast<T>(f.x[16]);
    p[offset + ldm + 4] = cast<T>(f.x[17]);
    p[offset + 6] = cast<T>(f.x[18]);
    p[offset + ldm + 6] = cast<T>(f.x[19]);
    p[offset + 16 * ldm + 4] = cast<T>(f.x[20]);
    p[offset + 17 * ldm + 4] = cast<T>(f.x[21]);
    p[offset + 16 * ldm + 6] = cast<T>(f.x[22]);
    p[offset + 17 * ldm + 6] = cast<T>(f.x[23]);
    p[offset + 4 * ldm + 4] = cast<T>(f.x[24]);
    p[offset + 5 * ldm + 4] = cast<T>(f.x[25]);
    p[offset + 4 * ldm + 6] = cast<T>(f.x[26]);
    p[offset + 5 * ldm + 6] = cast<T>(f.x[27]);
    p[offset + 20 * ldm + 4] = cast<T>(f.x[28]);
    p[offset + 21 * ldm + 4] = cast<T>(f.x[29]);
    p[offset + 20 * ldm + 6] = cast<T>(f.x[30]);
    p[offset + 21 * ldm + 6] = cast<T>(f.x[31]);
  } else {
    using DstType = typename Vector2<T>::Vector2Type;
    const int offset = row * ldm + col;
    DstType *dst = (DstType *)(p + offset);
    float2 *src = (float2 *)f.x;
    dst[0] = cast<DstType>(src[0]);
    dst[ldm] = cast<DstType>(src[1]);
    dst[8] = cast<DstType>(src[2]);
    dst[ldm + 8] = cast<DstType>(src[3]);
    dst[2] = cast<DstType>(src[4]);
    dst[ldm + 2] = cast<DstType>(src[5]);
    dst[10] = cast<DstType>(src[6]);
    dst[ldm + 10] = cast<DstType>(src[7]);
    dst[2 * ldm] = cast<DstType>(src[8]);
    dst[3 * ldm] = cast<DstType>(src[9]);
    dst[2 * ldm + 8] = cast<DstType>(src[10]);
    dst[3 * ldm + 8] = cast<DstType>(src[11]);
    dst[2 * ldm + 2] = cast<DstType>(src[12]);
    dst[3 * ldm + 2] = cast<DstType>(src[13]);
    dst[2 * ldm + 10] = cast<DstType>(src[14]);
    dst[3 * ldm + 10] = cast<DstType>(src[15]);
  }
}

/*
 * FP32 MMA functions for shape 16x16xk
 */
#define MMA_M16N16_F32_F32(A_LAYOUT, B_LAYOUT, K)                                                 \
  __device__ inline void mma_sync(                                                                \
    fragment<nvcuda::wmma::accumulator, M16, N16, K, float> &d,                                   \
    const fragment<nvcuda::wmma::matrix_a, M16, N16, K, half, nvcuda::wmma::A_LAYOUT##_major> &a, \
    const fragment<nvcuda::wmma::matrix_b, M16, N16, K, half, nvcuda::wmma::B_LAYOUT##_major> &b, \
    const fragment<nvcuda::wmma::accumulator, M16, N16, K, float> &c) {                           \
    asm volatile ("{mma.sync.aligned.m8n8k4." #A_LAYOUT "." #B_LAYOUT ".f32.f16.f16.f32"          \
        "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9},"                                             \
        "{%10, %11}, {%12, %13, %14, %15, %16, %17, %18, %19};}"                                  \
        : "=f"(d.x[0]), "=f"(d.x[1]), "=f"(d.x[2]), "=f"(d.x[3]),                                 \
          "=f"(d.x[4]), "=f"(d.x[5]), "=f"(d.x[6]), "=f"(d.x[7])                                  \
        : "r"(*reinterpret_cast<const unsigned *>(a.x)),                                          \
          "r"(*reinterpret_cast<const unsigned *>(a.x + 2)),                                      \
          "r"(*reinterpret_cast<const unsigned *>(b.x)),                                          \
          "r"(*reinterpret_cast<const unsigned *>(b.x + 2)), "f"(c.x[0]),                         \
          "f"(c.x[1]), "f"(c.x[2]), "f"(c.x[3]),                                                  \
          "f"(c.x[4]), "f"(c.x[5]), "f"(c.x[6]), "f"(c.x[7]));                                    \
    for (int k = 4; k < K; k += 4) {                                                              \
      asm volatile ("{mma.sync.aligned.m8n8k4." #A_LAYOUT "." #B_LAYOUT ".f32.f16.f16.f32"        \
          "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9},"                                           \
          "{%10, %11}, {%12, %13, %14, %15, %16, %17, %18, %19};}"                                \
          : "=f"(d.x[0]), "=f"(d.x[1]), "=f"(d.x[2]), "=f"(d.x[3]),                               \
            "=f"(d.x[4]), "=f"(d.x[5]), "=f"(d.x[6]), "=f"(d.x[7])                                \
          : "r"(*reinterpret_cast<const unsigned *>(a.x + k)),                                    \
            "r"(*reinterpret_cast<const unsigned *>(a.x + k + 2)),                                \
            "r"(*reinterpret_cast<const unsigned *>(b.x + k)),                                    \
            "r"(*reinterpret_cast<const unsigned *>(b.x + k + 2)),                                \
            "f"(d.x[0]), "f"(d.x[1]), "f"(d.x[2]), "f"(d.x[3]),                                   \
            "f"(d.x[4]), "f"(d.x[5]), "f"(d.x[6]) "f"(d.x[7]));                                   \
    }                                                                                             \
  }

MMA_M16N16_F32_F32(col, col, 4);
MMA_M16N16_F32_F32(row, col, 4);
MMA_M16N16_F32_F32(col, row, 4);
MMA_M16N16_F32_F32(row, row, 4);
MMA_M16N16_F32_F32(col, col, 8);
MMA_M16N16_F32_F32(row, col, 8);
MMA_M16N16_F32_F32(col, row, 8);
MMA_M16N16_F32_F32(row, row, 8);

/*
 * FP16 MMA functions for shape 16x16xk
 */
#define MMA_M16N16_F16_F16(A_LAYOUT, B_LAYOUT, K)                                                 \
  __device__ inline void mma_sync(                                                                \
    fragment<nvcuda::wmma::accumulator, M16, N16, K, half> &d,                                    \
    const fragment<nvcuda::wmma::matrix_a, M16, N16, K, half, nvcuda::wmma::A_LAYOUT##_major> &a, \
    const fragment<nvcuda::wmma::matrix_b, M16, N16, K, half, nvcuda::wmma::B_LAYOUT##_major> &b, \
    const fragment<nvcuda::wmma::accumulator, M16, N16, K, half> &c) {                            \
    asm volatile ("{mma.sync.aligned.m8n8k4." #A_LAYOUT "." #B_LAYOUT ".f16.f16.f16.f16"          \
      "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, {%8, %9, %10, %11};}"                                \
      : "=r"(*reinterpret_cast<unsigned *>(d.x)),                                                 \
        "=r"(*reinterpret_cast<unsigned *>(d.x + 2)),                                             \
        "=r"(*reinterpret_cast<unsigned *>(d.x + 4)),                                             \
        "=r"(*reinterpret_cast<unsigned *>(d.x + 6))                                              \
      : "r"(*reinterpret_cast<const unsigned *>(a.x)),                                            \
        "r"(*reinterpret_cast<const unsigned *>(a.x + 2)),                                        \
        "r"(*reinterpret_cast<const unsigned *>(b.x)),                                            \
        "r"(*reinterpret_cast<const unsigned *>(b.x + 2)),                                        \
        "r"(*reinterpret_cast<const unsigned *>(c.x)),                                            \
        "r"(*reinterpret_cast<const unsigned *>(c.x + 2)),                                        \
        "r"(*reinterpret_cast<const unsigned *>(c.x + 4)),                                        \
        "r"(*reinterpret_cast<const unsigned *>(c.x + 6)));                                       \
    for (int k = 4; k < K; k += 4) {                                                              \
      asm volatile ("{mma.sync.aligned.m8n8k4." #A_LAYOUT "." #B_LAYOUT ".f16.f16.f16.f16"        \
        "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, {%8, %9, %10, %11};}"                              \
        : "=r"(*reinterpret_cast<unsigned *>(d.x)),                                               \
          "=r"(*reinterpret_cast<unsigned *>(d.x + 2)),                                           \
          "=r"(*reinterpret_cast<unsigned *>(d.x + 4)),                                           \
          "=r"(*reinterpret_cast<unsigned *>(d.x + 6))                                            \
        : "r"(*reinterpret_cast<const unsigned *>(a.x + k)),                                      \
          "r"(*reinterpret_cast<const unsigned *>(a.x + k + 2)),                                  \
          "r"(*reinterpret_cast<const unsigned *>(b.x + k)),                                      \
          "r"(*reinterpret_cast<const unsigned *>(b.x + k + 2)),                                  \
          "r"(*reinterpret_cast<const unsigned *>(d.x)),                                          \
          "r"(*reinterpret_cast<const unsigned *>(d.x + 2)),                                      \
          "r"(*reinterpret_cast<const unsigned *>(d.x + 4)),                                      \
          "r"(*reinterpret_cast<const unsigned *>(d.x + 6)));                                     \
    }                                                                                             \
  }

MMA_M16N16_F16_F16(col, col, 4);
MMA_M16N16_F16_F16(row, col, 4);
MMA_M16N16_F16_F16(col, row, 4);
MMA_M16N16_F16_F16(row, row, 4);
MMA_M16N16_F16_F16(col, col, 8);
MMA_M16N16_F16_F16(row, col, 8);
MMA_M16N16_F16_F16(col, row, 8);
MMA_M16N16_F16_F16(row, row, 8);

#define MMA_M32N32K4_F32_F32_(A_LAYOUT, B_LAYOUT, STEP) do {                                    \
  asm volatile ("{mma.sync.aligned.m8n8k4." #A_LAYOUT "." #B_LAYOUT ".f32.f16.f16.f32"          \
    "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9},"                                               \
    "{%10, %11}, {%12, %13, %14, %15, %16, %17, %18, %19};}"                                    \
    : "=f"(d.x[0 + (STEP << 3)]), "=f"(d.x[1 + (STEP << 3)]),                                   \
      "=f"(d.x[2 + (STEP << 3)]), "=f"(d.x[3 + (STEP << 3)]),                                   \
      "=f"(d.x[4 + (STEP << 3)]), "=f"(d.x[5 + (STEP << 3)]),                                   \
      "=f"(d.x[6 + (STEP << 3)]), "=f"(d.x[7 + (STEP << 3)])                                    \
    : "r"(*reinterpret_cast<const unsigned *>(a.x + ((STEP & 0x2) << 1))),                      \
      "r"(*reinterpret_cast<const unsigned *>(a.x + ((STEP & 0x2) << 1) + 2)),                  \
      "r"(*reinterpret_cast<const unsigned *>(b.x + ((STEP & 0x1) << 2))),                      \
      "r"(*reinterpret_cast<const unsigned *>(b.x + ((STEP & 0x1) << 2) + 2)),                  \
      "f"(c.x[0 + (STEP << 3)]), "f"(c.x[1 + (STEP << 3)]),                                     \
      "f"(c.x[2 + (STEP << 3)]), "f"(c.x[3 + (STEP << 3)]),                                     \
      "f"(c.x[4 + (STEP << 3)]), "f"(c.x[5 + (STEP << 3)]),                                     \
      "f"(c.x[6 + (STEP << 3)]), "f"(c.x[7 + (STEP << 3)]));                                    \
} while (0);

/*
 * FP32 MMA functions for shape 32x32x4
 */
#define MMA_M32N32K4_F32_F32(A_LAYOUT, B_LAYOUT)                                                   \
  __device__ inline void mma_sync(                                                                 \
    fragment<nvcuda::wmma::accumulator, M32, N32, K4, float> &d,                                   \
    const fragment<nvcuda::wmma::matrix_a, M32, N32, K4, half, nvcuda::wmma::A_LAYOUT##_major> &a, \
    const fragment<nvcuda::wmma::matrix_b, M32, N32, K4, half, nvcuda::wmma::B_LAYOUT##_major> &b, \
    const fragment<nvcuda::wmma::accumulator, M32, N32, K4, float> &c) {                           \
    MMA_M32N32K4_F32_F32_(A_LAYOUT, B_LAYOUT, 0)                                                   \
    MMA_M32N32K4_F32_F32_(A_LAYOUT, B_LAYOUT, 1)                                                   \
    MMA_M32N32K4_F32_F32_(A_LAYOUT, B_LAYOUT, 2)                                                   \
    MMA_M32N32K4_F32_F32_(A_LAYOUT, B_LAYOUT, 3)                                                   \
  }

MMA_M32N32K4_F32_F32(col, row);

template <class T, int size>
__device__ inline void fragment_add(__frag_base<T, size> &c, const __frag_base<T, size> &a,
                                    const __frag_base<T, size> &b) {
#pragma unroll
  for (unsigned i = 0; i < c.num_elements; i++) {
    c.x[i] = a.x[i] + b.x[i];
  }
}

template <class T, int size>
__device__ inline void fragment_sub(__frag_base<T, size> &c, const __frag_base<T, size> &a,
                                    const __frag_base<T, size> &b) {
#pragma unroll
  for (unsigned i = 0; i < c.num_elements; i++) {
    c.x[i] = a.x[i] - b.x[i];
  }
}

template <class T, int size>
__device__ inline void fragment_mul(__frag_base<T, size> &c, const __frag_base<T, size> &a,
                                    const __frag_base<T, size> &b) {
#pragma unroll
  for (unsigned i = 0; i < c.num_elements; i++) {
    c.x[i] = a.x[i] * b.x[i];
  }
}

template <class T, int size>
__device__ inline void fragment_div(__frag_base<T, size> &c, const __frag_base<T, size> &a,
                                    const __frag_base<T, size> &b) {
#pragma unroll
  for (unsigned i = 0; i < c.num_elements; i++) {
    c.x[i] = a.x[i] / b.x[i];
  }
}

template <class T, int size>
__device__ inline void fragment_add(__frag_base<T, size> &c, const __frag_base<T, size> &a, const T b) {
#pragma unroll
  for (unsigned i = 0; i < c.num_elements; i++) {
    c.x[i] = a.x[i] + b;
  }
}

template <class T, int size>
__device__ inline void fragment_sub(__frag_base<T, size> &c, const __frag_base<T, size> &a, const T b) {
#pragma unroll
  for (unsigned i = 0; i < c.num_elements; i++) {
    c.x[i] = a.x[i] - b;
  }
}

template <class T, int size>
__device__ inline void fragment_mul(__frag_base<T, size> &c, const __frag_base<T, size> &a, const T b) {
#pragma unroll
  for (unsigned i = 0; i < c.num_elements; i++) {
    c.x[i] = a.x[i] * b;
  }
}

template <class T, int size>
__device__ inline void fragment_div(__frag_base<T, size> &c, const __frag_base<T, size> &a, const T b) {
#pragma unroll
  for (unsigned i = 0; i < c.num_elements; i++) {
    c.x[i] = a.x[i] / b;
  }
}

}  // namespace wmma
}  // namespace akg

#endif  // __WMMA_HPP__
