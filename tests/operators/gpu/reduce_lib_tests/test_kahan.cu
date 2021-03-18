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

#include <iostream>
#include <cstdlib>
#include <ctime>
#include "../reduce.cuh"

using namespace std;
using namespace akg_reduce;

// This implement shows the difference between
// 'Kahan summation algorithm' and 'Direct summation'
// See more in https://en.wikipedia.org/wiki/Kahan_summation_algorithm

template <typename T>
T ComputeKahanCPU(T *arr, int len) {
  T sum = 0.0;
  T low_bits = 0.0;
  T lower_val, cropped_sum;
  for (auto i = 0; i < len; i++) {
    lower_val = arr[i] - low_bits;
    cropped_sum = sum + lower_val;
    low_bits = (cropped_sum - sum) - lower_val;
    sum = cropped_sum;
  }
  return sum;
}

template <typename T>
T ComputeDirectCPU(T *arr, int len) {
  T sum = 0.0;
  for (auto i = 0; i < len; i++) {
    sum += arr[i];
  }
  return sum;
}

template<typename T>
__global__ void ComputeKahanAdd(int len,T *arr, T* output){
  T sum = 0.0;
  T low_bits = 0.0;
  T lower_val, cropped_sum;
  for (auto i = 0; i < len; i ++){
    AkgKahanAdd(sum, arr[i], low_bits, lower_val, cropped_sum);
  }
  output[0] = sum;
}

template <typename T>
T TestKahanGPU(T *arr, int len) {
  int input_bytes = len * sizeof(T);
  int output_bytes = 1 * sizeof(T);
  T *d_I, *h_O, *d_O;
  h_O = (T *)malloc(output_bytes);

  GetGpuErr(cudaMalloc((void **)&d_I, input_bytes));
  GetGpuErr(cudaMemcpy((void *)d_I, (void *)arr, input_bytes, cudaMemcpyHostToDevice));
  GetGpuErr(cudaMalloc((void **)&d_O, output_bytes));

  ComputeKahanAdd<T><<<1, 1>>>(len, d_I, d_O);
  GetGpuErr(cudaMemcpy((void *)h_O, (void *)d_O, output_bytes, cudaMemcpyDeviceToHost));

  T ans = h_O[0];
  GetGpuErr(cudaFree(d_O));
  GetGpuErr(cudaFree(d_I));
  free(h_O);

  return ans;
}


int main() {
  srand(time(0));
  float arr[1000000];
  for (auto i = 0; i < 1000000; i++) {
    arr[i] = (float)(rand() % 1000000) / 1000000.0;
  }
  printf("Kahan result:  %f\n", ComputeKahanCPU<float>(arr, 1000000));
  printf("Direct result: %f\n", ComputeDirectCPU<float>(arr, 1000000));
  printf("Kahan result in GPU: %f\n", TestKahanGPU<float>(arr, 1000000));

  return 0;
}