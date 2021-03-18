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

#include <iostream>
#include <chrono>
#include <m16n16k4.hpp>

// Usage: nvcc -std=c++11 -lineinfo -lcublas -arch=sm_70 -DCUDA_ARCH_SM=70 -I./ mma_test.cu -o mma_test

// const int WARP_SIZE = 32;
const int M = 16;
const int N = 16;
const int K = 4;
const int MMA_M = 16;
const int MMA_N = 16;
const int MMA_K = 4;

template<typename CType=float, typename ABType=half>
__global__ void wmma_test_kernel(CType *const c_ptr, const ABType *const a_ptr, const ABType *const b_ptr) {
  akg::wmma::fragment<nvcuda::wmma::matrix_a, MMA_M, MMA_N, MMA_K, ABType, nvcuda::wmma::row_major> frag_a_row;
  akg::wmma::fragment<nvcuda::wmma::matrix_a, MMA_M, MMA_N, MMA_K, ABType, nvcuda::wmma::col_major> frag_a_col;
  akg::wmma::fragment<nvcuda::wmma::matrix_b, MMA_M, MMA_N, MMA_K, ABType, nvcuda::wmma::row_major> frag_b_row;
  akg::wmma::fragment<nvcuda::wmma::matrix_b, MMA_M, MMA_N, MMA_K, ABType, nvcuda::wmma::col_major> frag_b_col;
  akg::wmma::fragment<nvcuda::wmma::accumulator, MMA_M, MMA_N, MMA_K, CType> frag_c;

  akg::wmma::load_matrix_sync(frag_a_row, a_ptr, K);
  akg::wmma::load_matrix_sync(frag_a_col, a_ptr, M);
  akg::wmma::load_matrix_sync(frag_b_row, b_ptr, N);
  akg::wmma::load_matrix_sync(frag_b_col, b_ptr, K);

  akg::wmma::load_matrix_sync<CType>(frag_c, c_ptr, N, nvcuda::wmma::mem_row_major);
  print_fragment(frag_c, "frag_c");

  akg::wmma::fill_fragment<CType>(frag_c, 0.0f);

  akg::wmma::mma_sync(frag_c, frag_a_col, frag_b_col, frag_c); 

  print_fragment(frag_a_row, "frag_a_row");
  print_fragment(frag_a_col, "frag_a_col");
  print_fragment(frag_b_row, "frag_b_row");
  print_fragment(frag_b_col, "frag_b_col");

  akg::wmma::store_matrix_sync(frag_c, c_ptr, N, nvcuda::wmma::mem_row_major);
}

#define FP16_EXPONENT_BITS 0x1F
#define FP16_EXPONENT_SHIFT 10
#define FP16_EXPONENT_BIAS 15
#define FP16_MANTISSA_BITS 0x3ff
#define FP16_MANTISSA_SHIFT (23 - FP16_EXPONENT_SHIFT)
#define FP16_MAX_EXPONENT (FP16_EXPONENT_BITS << FP16_EXPONENT_SHIFT)

inline half FP32toFP16(float val) {
  unsigned int f32 = (*(unsigned int *)&val);
  unsigned short f16 = 0;

  /* Decode IEEE 754 little-endian 32-bit floating-point value */
  int sign = (f32 >> 16) & 0x8000;

  /* Map exponent to the range [-127,128] */
  int exponent = ((f32 >> 23) & 0xff) - 127;
  int mantissa = f32 & 0x007fffff;

  if (exponent == 128) { /* Infinity or NaN */
    f16 = sign | FP16_MAX_EXPONENT;
    if (mantissa) f16 |= (mantissa & FP16_MANTISSA_BITS);
  } else if (exponent > 15) { /* Overflow - flush to Infinity */
    f16 = sign | FP16_MAX_EXPONENT;
  } else if (exponent > -15) { /* Representable value */
    exponent += FP16_EXPONENT_BIAS;
    mantissa >>= FP16_MANTISSA_SHIFT;
    f16 = sign | exponent << FP16_EXPONENT_SHIFT | mantissa;

  } else {
    f16 = sign;
  }
  return *(half *)&f16;
}

template <class T>
void oneInit(T *data, int size) {
  for (int i = 0; i < size; ++i) {
    data[i] = (T)FP32toFP16(1.f);
  }
}

template <class T>
void randomInit(T *data, int size) {
  for (int i = 0; i < size; ++i) {
    data[i] = (T)FP32toFP16(i);
  }
}

using stype = half;
using dtype = float;

int main() {
  half *da;
  half *db;
  float *dc;
  half *dc_fp16;

  unsigned int size_A = M * K;
  unsigned int size_B = K * N;
  unsigned int size_C = M * N;
  unsigned int size_C_fp16 = M * N;
  unsigned int mem_size_A = sizeof(stype) * size_A;
  unsigned int mem_size_B = sizeof(stype) * size_B;
  unsigned int mem_size_C = sizeof(dtype) * size_C;
  unsigned int mem_size_C_fp16 = sizeof(stype) * size_C_fp16; 
  printf("M = %d, N = %d, K = %d\n", M, N, K);
  printf("size_A = %d, size_B = %d, size_C = %d, size_C_fp16 = %d\n", mem_size_A, mem_size_B, mem_size_C, mem_size_C_fp16);
  stype *h_A = (stype *)malloc(mem_size_A);
  stype *h_B = (stype *)malloc(mem_size_B);
  dtype *h_C = (dtype *)malloc(mem_size_C);
  stype *h_C_fp16 = (stype *)malloc(mem_size_C_fp16);
  // dtype *reference = (dtype *)malloc(mem_size_C);
  // stype *reference = (stype *)malloc(mem_size_C_fp16);

  randomInit<stype>(h_A, size_A);
  randomInit<stype>(h_B, size_B);
  randomInit<dtype>(h_C, size_C);
  randomInit<stype>(h_C_fp16, size_C_fp16);

  cudaMalloc(&da, mem_size_A);
  cudaMalloc(&db, mem_size_B);
  cudaMalloc(&dc, mem_size_C);
  cudaMalloc(&dc_fp16, mem_size_C_fp16);

  // copy host memory to device
  cudaMemcpy(da, h_A, mem_size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(db, h_B, mem_size_B, cudaMemcpyHostToDevice);
  cudaMemcpy(dc, h_C, mem_size_C, cudaMemcpyHostToDevice);
  cudaMemcpy(dc_fp16, h_C_fp16, mem_size_C_fp16, cudaMemcpyHostToDevice);

  dim3 threads, grid;
  threads = dim3(32);
  grid = dim3(1, 1);

  // CType == fp32 
  wmma_test_kernel<float, half><<<grid, threads>>>(dc, da, db);
  cudaDeviceSynchronize();
  auto error_code = cudaGetLastError();
  printf("CType == fp32, last error: %d\n", error_code);
  cudaMemcpy(h_C, dc, mem_size_C, cudaMemcpyDeviceToHost);

  // CType == fp16
  wmma_test_kernel<half, half><<<grid, threads>>>(dc_fp16, da, db);
  cudaDeviceSynchronize();
  error_code = cudaGetLastError();
  printf("CType == fp16, last error: %d\n", error_code);
  cudaMemcpy(h_C_fp16, dc_fp16, mem_size_C_fp16, cudaMemcpyDeviceToHost);

  free(h_A);
  free(h_B);
  free(h_C);
  free(h_C_fp16);
  // free(reference);

  cudaFree(da);
  cudaFree(db);
  cudaFree(dc);
  cudaFree(dc_fp16);
}
