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
#include "../reduce.cuh"
#include "../store/store.cuh"

#include <cstdlib>
#include <ctime>
#include <cstdio>
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
    printf("Ouput (show few results):\n");
    for (auto i = 0; i < std::min(10, len); i++) {
      printf("%f ", TypeTransform<double, T>(arr1[i]));
    }
    printf("\n");
    printf("Expected:\n");
    for (auto i = 0; i < std::min(10, len); i++) {
      printf("%f ", TypeTransform<double, T>(arr2[i]));
    }
    printf("\n");
  }
  printf("AVERAGE_ERROR = %f\n", total_err / (double)len);
}

// Kahan summation for single thread Sum implement.
// More info in 'test_kahan.cc'
template <typename T>
__global__ void ComputeResultAlongXSingleThread(int x_len, int y_len, T *arr, T *output) {
  for (auto j = 0; j < y_len; j++) {
    T sum = 0.0;
    T low_bits = 0.0;
    T lower_val, cropped_sum;
    for (auto i = 0; i < x_len; i++) {
      lower_val = arr[i + j * x_len] - low_bits;
      cropped_sum = sum + lower_val;
      low_bits = (cropped_sum - sum) - lower_val;
      sum = cropped_sum;
    }
    output[j] = sum;
  }
}

template <typename T>
__global__ void ComputeResultAlongYSingleThread(int x_len, int y_len, T *arr, T *output) {
  for (auto i = 0; i < x_len; i++) {
    T sum = 0.0;
    T low_bits = 0.0;
    T lower_val, cropped_sum;
    for (auto j = 0; j < y_len; j++) {
      lower_val = arr[i + j * x_len] - low_bits;
      cropped_sum = sum + lower_val;
      low_bits = (cropped_sum - sum) - lower_val;
      sum = cropped_sum;
    }
    output[i] = sum;
  }
}

template <typename T, typename ReduceOp>
__global__ void ComputeResultAlongXGPUSingleBlock(int x_len, int y_len, T *arr, T *output, int item_per_thread,
                                                  ReduceOp op) {
  T T_red_rf = 0.0;
  __shared__ T red_buf[32];
  __shared__ T temp_output[1];  // temp storage for output
  temp_output[0] = (T) 0.0;
  for (int k = 0; k < item_per_thread; ++k) {
    if (threadIdx.x + k * blockDim.x < x_len && threadIdx.y + blockDim.y * blockIdx.x < y_len) {
      T_red_rf += arr[threadIdx.x + k * blockDim.x + threadIdx.y * x_len + blockIdx.y * blockDim.y * x_len];
    }
  }
  __syncthreads();
  AkgReduce<T, ReduceOp, 32, REDUCE2D_X>(op, &temp_output[0], red_buf, T_red_rf);
  __syncthreads();
  if (threadIdx.x == 0) {
    output[blockIdx.y * blockDim.y + threadIdx.y] = temp_output[0];
  }
}

template <typename T, typename ReduceOp>
__global__ void ComputeResultAlongXGPUMultiBlock(int x_len, int y_len, T *arr, T *output, int item_per_thread,
                                                 ReduceOp op) {
  T T_red_rf = 0.0;
  __shared__ T red_buf[32];
  __shared__ T temp_output[1];  // temp storage for output
  temp_output[0] = (T) 0.0;
  for (int k = 0; k < item_per_thread; ++k) {
    if (threadIdx.x + k * blockDim.x + blockIdx.x * blockDim.x * item_per_thread < x_len &&
        threadIdx.y + blockDim.y * blockIdx.y < y_len) {
      T_red_rf += arr[threadIdx.x + k * blockDim.x + blockIdx.x * blockDim.x * item_per_thread + threadIdx.y * x_len +
                      blockIdx.y * blockDim.y * x_len];
    }
  }
  __syncthreads();
  AkgReduce<T, ReduceOp, 32, REDUCE2D_X>(op, &temp_output[0], red_buf, T_red_rf);
  __syncthreads();
  if (threadIdx.x == 0) {
    AkgAtomicReturn<T, ReduceOp>(temp_output[0], &output[blockIdx.y * blockDim.y + threadIdx.y], op);
  }
}

template <typename T, typename ReduceOp>
__global__ void ComputeResultAlongYGPUSingleBlock(int x_len, int y_len, T *arr, T *output, int item_per_thread,
                                                  ReduceOp op, int sharedmem_x) {
  T T_red_rf = 0.0;
  __shared__ T red_buf[32];
  __shared__ T temp_output[1];  // temp storage for output, size is blockDimx.x
  temp_output[0] = (T) 0.0;
  for (int k = 0; k < item_per_thread; ++k) {
    if (threadIdx.x + blockIdx.x * blockDim.x < x_len && threadIdx.y + blockDim.y * k < y_len) {
      T_red_rf += arr[threadIdx.x + blockIdx.x * blockDim.x + threadIdx.y * x_len + k * blockDim.y * x_len];
    }
  }
  __syncthreads();
  AkgReduce<T, ReduceOp, 32, REDUCE2D_Y>(op, &temp_output[threadIdx.x], red_buf, T_red_rf, sharedmem_x);
  __syncthreads();
  if (threadIdx.y == 0) {
    AkgAtomicReturn<T, ReduceOp>(temp_output[threadIdx.x], &output[blockIdx.x * blockDim.x + threadIdx.x], op);
  }
}

template <typename T, typename ReduceOp>
__global__ void ComputeResultAlongYGPUMultiBlock(int x_len, int y_len, T *arr, T *output, int item_per_thread,
                                                 ReduceOp op, int sharedmem_x) {
  T T_red_rf = 0.0;
  __shared__ T red_buf[32];
  __shared__ T temp_output[1];  // temp storage for output, size is blockDimx.x
  temp_output[0] = (T) 0.0;
  for (int k = 0; k < item_per_thread; ++k) {
    if (threadIdx.x + blockIdx.x * blockDim.x < x_len &&
        threadIdx.y + blockDim.y * k + blockIdx.y * blockDim.y * item_per_thread < y_len) {
      T_red_rf += arr[threadIdx.x + blockIdx.x * blockDim.x + threadIdx.y * x_len + k * blockDim.y * x_len +
                      blockIdx.y * blockDim.y * item_per_thread * x_len];
    }
  }
  __syncthreads();
  AkgReduce<T, ReduceOp, 32, REDUCE2D_Y>(op, &temp_output[threadIdx.x], red_buf, T_red_rf, sharedmem_x);
  __syncthreads();
  if (threadIdx.y == 0) {
    AkgAtomicReturn<T, ReduceOp>(temp_output[threadIdx.x], &output[blockIdx.x * blockDim.x + threadIdx.x], op);
  }
}

template <typename T>
void TestReduce2DAlongX(int x_len, int y_len, string type_name, bool single_block = true, bool verbose = false) {
  printf("--- TEST CASE Reduce2DAlongX ---\n X = %d, Y = %d, TYPE = %s\n", x_len, y_len, type_name.c_str());
  int input_bytes = x_len * y_len * sizeof(T);
  int output_bytes = y_len * sizeof(T);
  T *h_I, *d_I, *h_O, *d_O, *expected_h_O, *expected_d_O;
  h_I = (T *)malloc(input_bytes);
  h_O = (T *)malloc(output_bytes);
  expected_h_O = (T *)malloc(output_bytes);

  // random initialize
  srand(time(0));
  for (auto i = 0; i < x_len * y_len; i++) {
    h_I[i] = TypeTransform<T, double>((rand() % 10000000) / 10000000.0);
  }

  if (verbose) {
    printf("[VERBOSE] random Input data:\n");
    for (auto j = 0; j < y_len; j++) {
      for (auto i = 0; i < x_len; i++) {
        printf("%f ", TypeTransform<double, T>(h_I[i + j * x_len]));
      }
      printf("\n");
    }
  }

  for (auto i = 0; i < y_len; i++) {
    h_O[i] = TypeTransform<T, double>(0.0);
    expected_h_O[i] = TypeTransform<T, double>(0.0);
  }

  // host to device
  GetGpuErr(cudaMalloc((void **)&d_I, input_bytes));
  GetGpuErr(cudaMemcpy((void *)d_I, (void *)h_I, input_bytes, cudaMemcpyHostToDevice));
  GetGpuErr(cudaMalloc((void **)&d_O, output_bytes));
  GetGpuErr(cudaMemcpy((void *)d_O, (void *)h_O, output_bytes, cudaMemcpyHostToDevice));
  GetGpuErr(cudaMalloc((void **)&expected_d_O, output_bytes));
  GetGpuErr(cudaMemcpy((void *)expected_d_O, (void *)expected_h_O, output_bytes, cudaMemcpyHostToDevice));

  // compute single thread resutls

  ComputeResultAlongXSingleThread<T><<<1, 1>>>(x_len, y_len, d_I, expected_d_O);
  GetGpuErr(cudaMemcpy((void *)expected_h_O, (void *)expected_d_O, output_bytes, cudaMemcpyDeviceToHost));

  if (single_block) {
    // compute GPU resutls
    dim3 gridSize(1, 8);
    dim3 blockSize(32, 1);
    int item_per_thread = (x_len - 1) / blockSize.x + 1;
    ComputeResultAlongXGPUSingleBlock<T, akg_reduce::SumOp>
      <<<gridSize, blockSize>>>(x_len, y_len, d_I, d_O, item_per_thread, akg_reduce::SumOp());
    GetGpuErr(cudaMemcpy((void *)h_O, (void *)d_O, output_bytes, cudaMemcpyDeviceToHost));

    // compare GPU with CPU
    CompareResults<T>(h_O, expected_h_O, y_len);
  } else {
    dim3 gridSize(2, 8);
    dim3 blockSize(32, 1);
    int item_per_block = (x_len - 1) / gridSize.x + 1;
    int item_per_thread = (item_per_block - 1) / blockSize.x + 1;
    ComputeResultAlongXGPUMultiBlock<T, akg_reduce::SumOp>
      <<<gridSize, blockSize>>>(x_len, y_len, d_I, d_O, item_per_thread, akg_reduce::SumOp());
    GetGpuErr(cudaMemcpy((void *)h_O, (void *)d_O, output_bytes, cudaMemcpyDeviceToHost));

    // compare GPU with CPU
    CompareResults<T>(h_O, expected_h_O, y_len);
  }

  GetGpuErr(cudaFree(expected_d_O));
  GetGpuErr(cudaFree(d_O));
  GetGpuErr(cudaFree(d_I));
  free(expected_h_O);
  free(h_O);
  free(h_I);
  printf("--- CASE END ---\n\n");
}

template <typename T>
void TestReduce2DAlongY(int x_len, int y_len, string type_name, bool single_block = true, bool verbose = false) {
  printf("--- TEST CASE Reduce2DAlongY ---\n X = %d, Y = %d, TYPE = %s\n", x_len, y_len, type_name.c_str());
  int input_bytes = x_len * y_len * sizeof(T);
  int output_bytes = x_len * sizeof(T);
  T *h_I, *d_I, *h_O, *d_O, *expected_h_O, *expected_d_O;
  h_I = (T *)malloc(input_bytes);
  h_O = (T *)malloc(output_bytes);
  expected_h_O = (T *)malloc(output_bytes);

  // random initialize
  srand(time(0));
  for (auto i = 0; i < x_len * y_len; i++) {
    h_I[i] = TypeTransform<T, double>((rand() % 10000000) / 10000000.0);
  }

  if (verbose) {
    printf("[VERBOSE] random Input data:\n");
    for (auto j = 0; j < y_len; j++) {
      for (auto i = 0; i < x_len; i++) {
        printf("%f ", TypeTransform<double, T>(h_I[i + j * x_len]));
      }
      printf("\n");
    }
  }

  for (auto i = 0; i < x_len; i++) {
    h_O[i] = TypeTransform<T, double>(0.0);
    expected_h_O[i] = TypeTransform<T, double>(0.0);
  }

  // host to device
  GetGpuErr(cudaMalloc((void **)&d_I, input_bytes));
  GetGpuErr(cudaMemcpy((void *)d_I, (void *)h_I, input_bytes, cudaMemcpyHostToDevice));
  GetGpuErr(cudaMalloc((void **)&d_O, output_bytes));
  GetGpuErr(cudaMemcpy((void *)d_O, (void *)h_O, output_bytes, cudaMemcpyHostToDevice));
  GetGpuErr(cudaMalloc((void **)&expected_d_O, output_bytes));
  GetGpuErr(cudaMemcpy((void *)expected_d_O, (void *)expected_h_O, output_bytes, cudaMemcpyHostToDevice));

  // compute single thread results

  ComputeResultAlongYSingleThread<T><<<1, 1>>>(x_len, y_len, d_I, expected_d_O);
  GetGpuErr(cudaMemcpy((void *)expected_h_O, (void *)expected_d_O, output_bytes, cudaMemcpyDeviceToHost));
  if (single_block) {
    // compute GPU results
    dim3 gridSize(8, 1);
    dim3 blockSize(1, 32);
    int item_per_thread = (y_len - 1) / blockSize.y + 1;
    int sharedmem_x = 1;
    ComputeResultAlongYGPUSingleBlock<T, akg_reduce::SumOp>
      <<<gridSize, blockSize>>>(x_len, y_len, d_I, d_O, item_per_thread, akg_reduce::SumOp(), sharedmem_x);
    GetGpuErr(cudaMemcpy((void *)h_O, (void *)d_O, output_bytes, cudaMemcpyDeviceToHost));

  } else {
    // compute GPU results
    dim3 gridSize(8, 2);
    dim3 blockSize(1, 32);
    int item_per_block = (y_len - 1) / gridSize.y + 1;
    int item_per_thread = (item_per_block - 1) / blockSize.y + 1;
    int sharedmem_x = 1;
    ComputeResultAlongYGPUMultiBlock<T, akg_reduce::SumOp>
      <<<gridSize, blockSize>>>(x_len, y_len, d_I, d_O, item_per_thread, akg_reduce::SumOp(), sharedmem_x);
    GetGpuErr(cudaMemcpy((void *)h_O, (void *)d_O, output_bytes, cudaMemcpyDeviceToHost));
  }
  // compare GPU with CPU
  CompareResults<T>(h_O, expected_h_O, x_len);

  GetGpuErr(cudaFree(expected_d_O));
  GetGpuErr(cudaFree(d_O));
  GetGpuErr(cudaFree(d_I));
  free(expected_h_O);
  free(h_O);
  free(h_I);
  printf("--- CASE END ---\n\n");
}

int main() {
  TestReduce2DAlongX<int>(128, 8, "int", true);
  TestReduce2DAlongX<half>(128, 8, "half", true);
  TestReduce2DAlongX<float>(128, 8, "float", true);
  TestReduce2DAlongX<double>(128, 8, "double", true);
  TestReduce2DAlongX<int>(128, 8, "int", false);
  TestReduce2DAlongX<float>(128, 8, "float", false);
  TestReduce2DAlongX<double>(128, 8, "double", false);

  TestReduce2DAlongY<int>(8, 128, "int", true);
  TestReduce2DAlongY<half>(8, 128, "half", true);
  TestReduce2DAlongY<float>(8, 128, "float", true);
  TestReduce2DAlongY<double>(8, 128, "double", true);
  TestReduce2DAlongY<int>(8, 128, "int", false);
  TestReduce2DAlongY<half>(8, 128, "half", false);
  TestReduce2DAlongY<float>(8, 128, "float", false);
  TestReduce2DAlongY<double>(8, 128, "double", false);

  return 0;
}
