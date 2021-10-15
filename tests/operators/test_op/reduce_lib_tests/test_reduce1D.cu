/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "../utils/util.cuh"
#include "../algorithm/shared_reduce.cuh"
#include "../store/store.cuh"
#include "../reduce.cuh"
#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <cmath>
using namespace akg_reduce;
using namespace std;

template <typename T>
void CompareResults(T *arr1, T *arr2, int len) {
  double total_err = 0.0;
  bool flag = true;
  for (auto i = 0; i < len; i++) {
    if (std::abs(TypeTransform<double, T>(arr1[i]) - TypeTransform<double, T>(arr2[i])) > 1e-03) {
      flag = false;
    }
    total_err += std::abs(TypeTransform<double, T>(arr1[i]) - TypeTransform<double, T>(arr2[i]));
  }
  if (flag) {
    printf("[CORRECT] Output is equal to Expected.\n");
  } else {
    printf("[INCORRECT] Output is not equal to Expected\n");
  }
  printf("Ouput (show few results):\n");
  for (auto i = 0; i < std::min(10, len); i++) {
    printf("%f ", TypeTransform<double, T>(arr1[i]));
  }
  printf("\n");
  printf("Expected:\n");
  for (auto i = 0; i < std::min(10, len); i++) {
    printf("%f ", TypeTransform<double, T>(arr2[i]));
  }
  printf("AVERAGE_ERROR = %f\n", total_err / (double)len);
  printf("\n");
}

template <typename T>
__global__ void ComputeResultSingleThread1D(int x_len, T *arr, T *output) {
  // 1D single thread computation: A Sum function using Kahan summation algorithm.
  // more info in test_kahan.cc
  T sum = 0.0;
  T low_bits = 0.0;
  T lower_val, cropped_sum;
  for (auto i = 0; i < x_len; i++) {
    lower_val = arr[i] - low_bits;
    cropped_sum = sum + lower_val;
    low_bits = (cropped_sum - sum) - lower_val;
    sum = cropped_sum;
  }
  output[0] = sum;
}

template <typename T, typename ReduceOp>
__global__ void ComputeResultGPUSingleBlock1D(int x_len, T *arr, T *output, int item_per_thread, ReduceOp op) {
  T temp_rf = 0.0;
  __shared__ T red_buf[64];
  __shared__ T temp_output[1];
  temp_output[0] = (T)0.0;
  for (int k = 0; k < item_per_thread; ++k) {
    if ((int)threadIdx.x + k * blockDim.x < x_len) {
      temp_rf += arr[(int)threadIdx.x + k * blockDim.x];
    }
  }
  __syncthreads();
  AkgReduce<T, ReduceOp, 64, ALL_REDUCE>(op, &temp_output[0], red_buf, temp_rf);
  __syncthreads();
  output[0] = temp_output[0];
}

template <typename T, typename ReduceOp>
__global__ void ComputeResultGPUMultiBlock1D(int x_len, T *arr, T *output, int item_per_thread, ReduceOp op) {
  T temp_rf = 0.0;
  __shared__ T red_buf[32];
  __shared__ T temp_output[1];  // temp storage for output
  temp_output[0] = (T)0.0;
  for (int k = 0; k < item_per_thread; ++k) {
    if (threadIdx.x + k * blockDim.x + blockIdx.x * blockDim.x * item_per_thread < x_len) {
      temp_rf += arr[threadIdx.x + k * blockDim.x + blockIdx.x * blockDim.x * item_per_thread];
    }
  }
  __syncthreads();
  AkgReduce<T, ReduceOp, 32, ALL_REDUCE>(op, &temp_output[0], red_buf, temp_rf);
  __syncthreads();
  if (threadIdx.x == 0) {
    AkgAtomicReturn<T, ReduceOp>(temp_output[0], &output[0], op);
  }
}

template <typename T>
void TestReduce1D(int x_len, string type_name, bool single_block = true, bool verbose = false) {
  printf("--- TEST CASE Reduce1D ---\n X = %d, TYPE = %s\n", x_len, type_name.c_str());
  int input_bytes = x_len * sizeof(T);
  int output_bytes = 1 * sizeof(T);
  T *h_I, *d_I, *h_O, *d_O, *expected_h_O, *expected_d_O;
  h_I = (T *)malloc(input_bytes);
  h_O = (T *)malloc(output_bytes);
  expected_h_O = (T *)malloc(output_bytes);

  // random initialize
  srand(time(0));
  for (auto i = 0; i < x_len; i++) {
    h_I[i] = TypeTransform<T, double>((rand() % 100) / 100.0);
  }

  if (verbose) {
    printf("[VERBOSE] random Input data:\n");
    for (auto i = 0; i < x_len; i++) {
      printf("%f ", TypeTransform<double, T>(h_I[i]));
    }
    printf("\n");
  }

  h_O[0] = TypeTransform<T, double>(0.0);
  expected_h_O[0] = TypeTransform<T, double>(0.0);

  // host to device
  GetGpuErr(cudaMalloc((void **)&d_I, input_bytes));
  GetGpuErr(cudaMemcpy((void *)d_I, (void *)h_I, input_bytes, cudaMemcpyHostToDevice));
  GetGpuErr(cudaMalloc((void **)&d_O, output_bytes));
  GetGpuErr(cudaMemcpy((void *)d_O, (void *)h_O, output_bytes, cudaMemcpyHostToDevice));
  GetGpuErr(cudaMalloc((void **)&expected_d_O, output_bytes));
  GetGpuErr(cudaMemcpy((void *)expected_d_O, (void *)expected_h_O, output_bytes, cudaMemcpyHostToDevice));

  // compute single thread results
  ComputeResultSingleThread1D<T><<<1, 1>>>(x_len, d_I, expected_d_O);
  GetGpuErr(cudaMemcpy((void *)expected_h_O, (void *)expected_d_O, output_bytes, cudaMemcpyDeviceToHost));

  if (single_block) {
    // compute GPU single-block results
    dim3 gridSize(1);
    dim3 blockSize(64);
    int item_per_thread = (x_len - 1) / blockSize.x + 1;
    ComputeResultGPUSingleBlock1D<T, akg_reduce::SumOp>
      <<<gridSize, blockSize>>>(x_len, d_I, d_O, item_per_thread, akg_reduce::SumOp());
    GetGpuErr(cudaMemcpy((void *)h_O, (void *)d_O, output_bytes, cudaMemcpyDeviceToHost));
  } else {
    // compute GPU multi-block results
    dim3 gridSize1(2);
    dim3 blockSize1(32);
    int item_per_block = (x_len - 1) / gridSize1.x + 1;
    int item_per_thread1 = (item_per_block - 1) / blockSize1.x + 1;
    ComputeResultGPUMultiBlock1D<T, akg_reduce::SumOp>
      <<<gridSize1, blockSize1>>>(x_len, d_I, d_O, item_per_thread1, akg_reduce::SumOp());
    GetGpuErr(cudaMemcpy((void *)h_O, (void *)d_O, output_bytes, cudaMemcpyDeviceToHost));
  }

  // compare GPU with CPU
  CompareResults<T>(h_O, expected_h_O, 1);

  GetGpuErr(cudaFree(expected_d_O));
  GetGpuErr(cudaFree(d_O));
  GetGpuErr(cudaFree(d_I));
  free(expected_h_O);
  free(h_O);
  free(h_I);
  printf("--- CASE END ---\n\n");
}

int main() {
  TestReduce1D<int>(128, "int", true);
  TestReduce1D<half>(128, "half", true);
  TestReduce1D<float>(128, "float", true);
  TestReduce1D<double>(128, "double", true);
  TestReduce1D<int>(128, "int", false);
  TestReduce1D<half>(128, "half", false);
  TestReduce1D<float>(128, "float", false);
  TestReduce1D<double>(128, "double", false);

  return 0;
}
