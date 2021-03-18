/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef __WMMA_M16N16K4_HPP__
#define __WMMA_M16N16K4_HPP__
#include <mma.h>
#include <cuda_fp16.h>

namespace akg {
namespace wmma {

inline __device__ unsigned get_lane_id() {
  unsigned lane_id;
  asm(R"({mov.s32 %0, %laneid;})" : "=r"(lane_id));
  return lane_id;
}

template <typename T, int size>
struct __align__(4) __frag_base {
  T x[size];
  enum { num_elements = size };
};

template <class Use, int m, int n, int k, class T, class Layout = void>
class fragment;
template <>
class fragment<nvcuda::wmma::matrix_a, 16, 16, 4, half, nvcuda::wmma::col_major> : public __frag_base<half, 4> {};
template <>
class fragment<nvcuda::wmma::matrix_a, 16, 16, 4, half, nvcuda::wmma::row_major> : public __frag_base<half, 4> {};
template <>
class fragment<nvcuda::wmma::matrix_b, 16, 16, 4, half, nvcuda::wmma::col_major> : public __frag_base<half, 4> {};
template <>
class fragment<nvcuda::wmma::matrix_b, 16, 16, 4, half, nvcuda::wmma::row_major> : public __frag_base<half, 4> {};
template <>
class fragment<nvcuda::wmma::accumulator, 16, 16, 4, half> : public __frag_base<half, 8> {};
template <>
class fragment<nvcuda::wmma::accumulator, 16, 16, 4, float> : public __frag_base<float, 8> {};

template <class T, int size>
__device__ inline void fill_fragment(__frag_base<T, size> &f, const T v) {
#pragma unroll
  for (unsigned i = 0; i < f.num_elements; i++) {
    f.x[i] = v;
  }
}

template <class T>
__device__ inline void load_matrix_sync(
  akg::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 4, half, nvcuda::wmma::col_major> &f, const T *const p,
  const unsigned ldm) {
  const unsigned lane_id = get_lane_id();
  const unsigned row = lane_id & 0x3;
  const unsigned col = (lane_id & 0x4) + ((lane_id >> 4) << 3);
  const unsigned offset = row * ldm + col;

  float2 *src = (float2 *)(p + offset);
  float2 *dst = (float2 *)f.x;
  dst[0] = src[0];
}

template <class T>
__device__ inline void load_matrix_sync(
  akg::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 4, half, nvcuda::wmma::row_major> &f, const T *const p,
  const unsigned ldm) {
  const unsigned lane_id = get_lane_id();
  const unsigned row = (lane_id & 0x7) + ((lane_id >> 4) << 3);  // (l and (1111)b + l div 16 * 8)
  const unsigned offset = row * ldm;
  float2 *src = (float2 *)(p + offset);
  float2 *dst = (float2 *)f.x;
  dst[0] = src[0];
}

template <class T>
__device__ inline void load_matrix_sync(
  akg::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 4, half, nvcuda::wmma::col_major> &f, const T *const p,
  const unsigned ldm) {
  const unsigned lane_id = get_lane_id();
  const unsigned row = (lane_id & 0x3) + ((lane_id & 0x18) >> 1);
  const unsigned offset = row * ldm;

  float2 *src = (float2 *)(p + offset);
  float2 *dst = (float2 *)f.x;
  dst[0] = src[0];
}

template <class T>
__device__ inline void load_matrix_sync(
  akg::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 4, half, nvcuda::wmma::row_major> &f, const T *const p,
  const unsigned ldm) {
  const unsigned lane_id = get_lane_id();
  const unsigned row = lane_id & 0x3;
  const unsigned col = (lane_id >> 3) << 2;
  const unsigned offset = row * ldm + col;

  float2 *src = (float2 *)(p + offset);
  float2 *dst = (float2 *)f.x;
  dst[0] = src[0];
}

template <class T>
__device__ inline void load_matrix_sync(akg::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 4, half, void> &f,
                                        const T *const p, const unsigned ldm, const nvcuda::wmma::layout_t layout) {
  const unsigned lane_id = get_lane_id();
  const unsigned row = (lane_id & 0x7) + ((lane_id >> 4) << 3);
  const unsigned col = ((lane_id & 0xf) >> 3) << 2;
  if (layout == nvcuda::wmma::mem_col_major) {
    const int offset = row * ldm + col;
    f.x[0] = static_cast<T>(p[offset]);
    f.x[1] = static_cast<T>(p[offset + ldm]);
    f.x[2] = static_cast<T>(p[offset + 2 * ldm]);
    f.x[3] = static_cast<T>(p[offset + 3 + ldm]);
    f.x[4] = static_cast<T>(p[offset + 8 * ldm]);
    f.x[5] = static_cast<T>(p[offset + 9 * ldm]);
    f.x[6] = static_cast<T>(p[offset + 10 * ldm]);
    f.x[7] = static_cast<T>(p[offset + 11 * ldm]);
  } else {
    const int offset = row * ldm + col;
    float2 *src = (float2 *)(p + offset);
    float2 *dst = (float2 *)f.x;
    dst[0] = src[0];
    dst[1] = src[2];
  }
}

template <class T>
__device__ inline void store_matrix_sync(T *const p,
                                         const fragment<nvcuda::wmma::accumulator, 16, 16, 4, half, void> &f,
                                         const unsigned ldm, const nvcuda::wmma::layout_t layout) {
  const unsigned lane_id = get_lane_id();
  const unsigned row = (lane_id & 0x7) + ((lane_id >> 4) << 3);
  const unsigned col = ((lane_id & 0xf) >> 3) << 2;
  if (layout == nvcuda::wmma::mem_col_major) {
    const int offset = row * ldm + col;
    p[offset + 0] = static_cast<T>(f.x[0]);
    p[offset + ldm] = static_cast<T>(f.x[1]);
    p[offset + 2 * ldm] = static_cast<T>(f.x[2]);
    p[offset + 3 * ldm] = static_cast<T>(f.x[3]);
    p[offset + 8 * ldm] = static_cast<T>(f.x[4]);
    p[offset + 9 * ldm] = static_cast<T>(f.x[5]);
    p[offset + 10 * ldm] = static_cast<T>(f.x[6]);
    p[offset + 11 * ldm] = static_cast<T>(f.x[7]);
  } else {
    const int offset = row * ldm + col;
    float2 *dst = (float2 *)(p + offset);
    float2 *src = (float2 *)f.x;
    dst[0] = src[0];
    dst[2] = src[1];
  }
}

// ptx_isa_7.1.pdf page277
template <class T>
__device__ inline void load_matrix_sync(fragment<nvcuda::wmma::accumulator, 16, 16, 4, float, void> &f, T *const p,
                                        const unsigned ldm, const nvcuda::wmma::layout_t layout) {
  const unsigned lane_id = get_lane_id();
  const unsigned row = (lane_id & 0x5) + ((lane_id >> 4) << 3);
  const unsigned col = ((lane_id & 0x2)) + ((lane_id & 0x8) >> 1);
  if (layout == nvcuda::wmma::mem_col_major) {
    const int offset = row * ldm + col;
    f.x[0] = static_cast<T>(p[offset]);
    f.x[1] = static_cast<T>(p[offset + ldm]);
    f.x[2] = static_cast<T>(p[offset + 2]);
    f.x[3] = static_cast<T>(p[offset + 2 + ldm]);
    f.x[4] = static_cast<T>(p[offset + 8 * ldm]);
    f.x[5] = static_cast<T>(p[offset + 9 * ldm]);
    f.x[6] = static_cast<T>(p[offset + 2 + 8 * ldm]);
    f.x[7] = static_cast<T>(p[offset + 2 + 9 * ldm]);
  } else {
    const int offset = row * ldm + col;
    float2 *src = (float2 *)(p + offset);
    float2 *dst = (float2 *)f.x;
    dst[0] = src[0];
    dst[1] = src[ldm];
    dst[2] = src[4];
    dst[3] = src[ldm + 4];
  }
}

template <>
__device__ inline void load_matrix_sync(fragment<nvcuda::wmma::accumulator, 16, 16, 4, float, void> &f, half *const p,
                                        const unsigned ldm, const nvcuda::wmma::layout_t layout) {
  const unsigned lane_id = get_lane_id();
  const unsigned row = (lane_id & 0x5) + ((lane_id >> 4) << 3);
  const unsigned col = ((lane_id & 0x2)) + ((lane_id & 0x8) >> 1);
  if (layout == nvcuda::wmma::mem_col_major) {
    const int offset = row * ldm + col;
    f.x[0] = static_cast<half>(p[offset]);
    f.x[1] = static_cast<half>(p[offset + ldm]);
    f.x[2] = static_cast<half>(p[offset + 2]);
    f.x[3] = static_cast<half>(p[offset + 2 + ldm]);
    f.x[4] = static_cast<half>(p[offset + 8 * ldm]);
    f.x[5] = static_cast<half>(p[offset + 9 * ldm]);
    f.x[6] = static_cast<half>(p[offset + 2 + 8 * ldm]);
    f.x[7] = static_cast<half>(p[offset + 2 + 9 * ldm]);
  } else {
    const int offset = row * ldm + col;
    half2 *src = (half2 *)(p + offset);
    float2 *dst = (float2 *)f.x;
    dst[0] = __half22float2(src[0]);
    dst[1] = __half22float2(src[ldm]);
    dst[2] = __half22float2(src[4]);
    dst[3] = __half22float2(src[ldm + 4]);
  }
}

template <class T>
__device__ inline void store_matrix_sync(T *const p, fragment<nvcuda::wmma::accumulator, 16, 16, 4, float, void> &f,
                                         const unsigned ldm, const nvcuda::wmma::layout_t layout) {
  const unsigned lane_id = get_lane_id();
  const unsigned row = (lane_id & 0x5) + ((lane_id >> 4) << 3);
  const unsigned col = ((lane_id & 0x2)) + ((lane_id & 0x8) >> 1);

  if (layout == nvcuda::wmma::mem_col_major) {
    const int offset = row * ldm + col;
    p[offset + 0] = static_cast<T>(f.x[0]);
    p[offset + ldm] = static_cast<T>(f.x[1]);
    p[offset + 2] = static_cast<T>(f.x[2]);
    p[offset + 2 + ldm] = static_cast<T>(f.x[3]);
    p[offset + 8 * ldm] = static_cast<T>(f.x[4]);
    p[offset + 9 * ldm] = static_cast<T>(f.x[5]);
    p[offset + 2 + 8 * ldm] = static_cast<T>(f.x[6]);
    p[offset + 2 + 9 * ldm] = static_cast<T>(f.x[7]);
  } else {
    const int offset = row * ldm + col;
    float2 *dst = (float2 *)(p + offset);
    float2 *src = (float2 *)f.x;
    dst[0] = src[0];
    dst[ldm] = src[1];
    dst[4] = src[2];
    dst[ldm + 4] = src[3];
  }
}

template <>
__device__ inline void store_matrix_sync(half *const p, fragment<nvcuda::wmma::accumulator, 16, 16, 4, float, void> &f,
                                         const unsigned ldm, const nvcuda::wmma::layout_t layout) {
  const unsigned lane_id = get_lane_id();
  const unsigned row = (lane_id & 0x5) + ((lane_id >> 4) << 3);
  const unsigned col = ((lane_id & 0x2)) + ((lane_id & 0x8) >> 1);

  if (layout == nvcuda::wmma::mem_col_major) {
    const int offset = row * ldm + col;
    p[offset + 0] = __float2half(f.x[0]);
    p[offset + ldm] = __float2half(f.x[1]);
    p[offset + 2] = __float2half(f.x[2]);
    p[offset + 2 + ldm] = __float2half(f.x[3]);
    p[offset + 8 * ldm] = __float2half(f.x[4]);
    p[offset + 9 * ldm] = __float2half(f.x[5]);
    p[offset + 2 + 8 * ldm] = __float2half(f.x[6]);
    p[offset + 2 + 9 * ldm] = __float2half(f.x[7]);
  } else {
    const int offset = row * ldm + col;
    half2 *dst = (half2 *)(p + offset);
    float2 *src = (float2 *)f.x;
    dst[0] = __float22half2_rn(src[0]);
    dst[ldm] = __float22half2_rn(src[1]);
    dst[4] = __float22half2_rn(src[2]);
    dst[ldm + 4] = __float22half2_rn(src[3]);
  }
}

#define MMA_M16N16K4_F32_F32(A_LAYOUT, B_LAYOUT)                                             \
__device__ inline void mma_sync(                                                             \
  fragment<nvcuda::wmma::accumulator, 16,16,4,float> & d,                                    \
  const fragment<nvcuda::wmma::matrix_a, 16,16,4, half, nvcuda::wmma::A_LAYOUT##_major> & a, \
  const fragment<nvcuda::wmma::matrix_b, 16,16,4, half, nvcuda::wmma::B_LAYOUT##_major> & b, \
  const fragment<nvcuda::wmma::accumulator, 16,16,4, float> & c){                            \
    asm("{mma.sync.aligned.m8n8k4." #A_LAYOUT "." #B_LAYOUT"                                 \
        .f32.f16.f16.f32 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9},"                        \
        "{%10, %11}, {%12, %13, %14, %15, %16, %17, %18, %19};}"                             \
        :"=f"(d.x[0]),                                                                       \
         "=f"(d.x[1]),                                                                       \
         "=f"(d.x[2]),                                                                       \
         "=f"(d.x[3]),                                                                       \
         "=f"(d.x[4]),                                                                       \
         "=f"(d.x[5]),                                                                       \
         "=f"(d.x[6]),                                                                       \
         "=f"(d.x[7])                                                                        \
        :"r"(*reinterpret_cast<const unsigned *>(a.x)),                                      \
         "r"(*reinterpret_cast<const unsigned *>(a.x + 2)),                                  \
         "r"(*reinterpret_cast<const unsigned *>(b.x)),                                      \
         "r"(*reinterpret_cast<const unsigned *>(b.x + 2)),                                  \
         "f"(c.x[0]),                                                                        \
         "f"(c.x[1]),                                                                        \
         "f"(c.x[2]),                                                                        \
         "f"(c.x[3]),                                                                        \
         "f"(c.x[4]),                                                                        \
         "f"(c.x[5]),                                                                        \
         "f"(c.x[6]),                                                                        \
         "f"(c.x[7]));                                                                       \
  }                                                                                          \

MMA_M16N16K4_F32_F32(col, col);
MMA_M16N16K4_F32_F32(row, col);
MMA_M16N16K4_F32_F32(col, row);
MMA_M16N16K4_F32_F32(row, row);

#define MMA_M16N16K4_F16_F16(A_LAYOUT, B_LAYOUT)                                             \
__device__ inline void mma_sync(                                                             \
  fragment<nvcuda::wmma::accumulator, 16,16,4,half> & d,                                     \
  const fragment<nvcuda::wmma::matrix_a, 16,16,4, half, nvcuda::wmma::A_LAYOUT##_major> & a, \
  const fragment<nvcuda::wmma::matrix_b, 16,16,4, half, nvcuda::wmma::B_LAYOUT##_major> & b, \
  const fragment<nvcuda::wmma::accumulator, 16,16,4, half> & c){                             \
    asm("{mma.sync.aligned.m8n8k4." #A_LAYOUT "." #B_LAYOUT"                                 \
        .f16.f16.f16.f16 {%0, %1, %2, %3}, {%4, %5},"                                        \
        "{%6, %7}, {%8, %9, %10, %11};}"                                                     \
        :"=r"(*reinterpret_cast<unsigned *>(d.x)),                                           \
         "=r"(*reinterpret_cast<unsigned *>(d.x + 2)),                                       \
         "=r"(*reinterpret_cast<unsigned *>(d.x + 4)),                                       \
         "=r"(*reinterpret_cast<unsigned *>(d.x + 6))                                        \
        :"r"(*reinterpret_cast<const unsigned *>(a.x)),                                      \
         "r"(*reinterpret_cast<const unsigned *>(a.x + 2)),                                  \
         "r"(*reinterpret_cast<const unsigned *>(b.x)),                                      \
         "r"(*reinterpret_cast<const unsigned *>(b.x + 2)),                                  \
         "r"(*reinterpret_cast<const unsigned *>(c.x)),                                      \
         "r"(*reinterpret_cast<const unsigned *>(c.x + 2)),                                  \
         "r"(*reinterpret_cast<const unsigned *>(c.x + 4)),                                  \
         "r"(*reinterpret_cast<const unsigned *>(c.x + 6)));                                 \
  }                                                                                          \

MMA_M16N16K4_F16_F16(col, col);
MMA_M16N16K4_F16_F16(row, col);
MMA_M16N16K4_F16_F16(col, row);
MMA_M16N16K4_F16_F16(row, row);

template <class MatrixType, int M, int N, int K, class MemMajor, class T>
__device__ inline void print_fragment(const akg::wmma
                                      ::fragment<MatrixType, M, N, K, T, MemMajor> &frag, const char *name = "") {
  if ((threadIdx.x & 0x1f) == 0) {
    if (name[0] != '\0') {
      printf("%s = \n", name);
    }
  }

  for (unsigned i = 0; i < warpSize; i++) {
    if (i == (threadIdx.x & 0x1f)) {
      printf("threadIdx.x = %d", threadIdx.x);
      for (unsigned j = 0; j < frag.num_elements; j++) {
        float v;
        if (sizeof(T) == 2) {
          v = __half2float(frag.x[j]);
        } else {
          v = frag.x[j];
        }
        if (v == 0.0f) {
          printf(" %f ", 0.0f);
        } else if (v > 0) {
          printf(" %f ", v);
        } else {
          printf("%f ", v);
        }
      }
      printf("\n");
    }
    __syncthreads();
  }
}

}  // namespace akg
}  // namespace wmma

#endif  // __WMMA_M16N16_K4_HPP__
