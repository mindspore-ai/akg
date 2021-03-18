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
#include "../operators/reduce_operators.cuh"
#include <cstdio>

using namespace akg_reduce;
using namespace std;

// check whether the op-atomic transformation is correct.
// compile code: nvcc test_atomic.cu -arch=sm_70

template <typename T>
__global__ void AtomicTestSum(T *dest, T val) {
  SumOp<T> op;
  AtomicOp<T, op.identifier> atomic_op;
  atomic_op.Compute(&dest[threadIdx.x], val);
}

template <typename T>
__global__ void AtomicTestMax(T *dest, T val) {
  MaxOp<T> op;
  AtomicOp<T, op.identifier> atomic_op;
  atomic_op.Compute(&dest[threadIdx.x], val);
}

template <typename T>
__global__ void AtomicTestMin(T *dest, T val) {
  MinOp<T> op;
  AtomicOp<T, op.identifier> atomic_op;
  atomic_op.Compute(&dest[threadIdx.x], val);
}

template <typename T>
void TestAtomicSum() {
  cout << "TestAtomicSum" << endl;
  int items = 1000;
  int bytes = items * sizeof(T);
  T *h_a, *d_a;

  h_a = (T *)malloc(bytes);
  for (auto i = 0; i < items; i++) {
    if (sizeof(T) == 2) {
      h_a[i] = __float2half(0.0);
    } else {
      h_a[i] = 0.0;
    }
  }

  GetGpuErr(cudaMalloc((void **)&d_a, bytes));
  GetGpuErr(cudaMemcpy((void *)d_a, (void *)h_a, bytes, cudaMemcpyHostToDevice));

  dim3 grid(1000);
  dim3 block(1000);
  AtomicTestSum<T><<<grid, block>>>(d_a, 1.0);
  GetGpuErr(cudaPeekAtLastError());

  GetGpuErr(cudaMemcpy((void *)h_a, (void *)d_a, bytes, cudaMemcpyDeviceToHost));

  for (auto i = 0; i < 10; i++) {
    double tmp;
    if (sizeof(T) == 2) {
      tmp = __half2float(h_a[i]);
    } else {
      tmp = h_a[i];
    }
    printf("%f ", tmp);
  }
  printf("\n");

  GetGpuErr(cudaFree(d_a));
  free(h_a);
}

template <typename T>
void TestAtomicMax() {
  cout << "TestAtomicMax" << endl;
  int items = 10;
  int bytes = items * sizeof(T);
  T *h_a, *d_a;

  h_a = (T *)malloc(bytes);
  for (auto i = 0; i < items; i++) {
    if (sizeof(T) == 2) {
      h_a[i] = __float2half(i);
    } else {
      h_a[i] = i;
    }
  }

  GetGpuErr(cudaMalloc((void **)&d_a, bytes));
  GetGpuErr(cudaMemcpy((void *)d_a, (void *)h_a, bytes, cudaMemcpyHostToDevice));

  double val = 1.234567891012345;
  dim3 grid(10000);
  dim3 block(items);
  AtomicTestMax<T><<<grid, block>>>(d_a, val);
  GetGpuErr(cudaPeekAtLastError());

  GetGpuErr(cudaMemcpy((void *)h_a, (void *)d_a, bytes, cudaMemcpyDeviceToHost));

  for (auto i = 0; i < 5; i++) {
    double tmp;
    if (sizeof(T) == 2) {
      tmp = __half2float(h_a[i]);
    } else {
      tmp = h_a[i];
    }
    printf("%.12f ", tmp);
  }
  printf("\n");

  GetGpuErr(cudaFree(d_a));
  free(h_a);
}

template <typename T>
void TestAtomicMin() {
  cout << "TestAtomicMin" << endl;
  int items = 10;
  int bytes = items * sizeof(T);
  T *h_a, *d_a;

  h_a = (T *)malloc(bytes);
  for (auto i = 0; i < items; i++) {
    h_a[i] = __float2half(i);
  }

  GetGpuErr(cudaMalloc((void **)&d_a, bytes));
  GetGpuErr(cudaMemcpy((void *)d_a, (void *)h_a, bytes, cudaMemcpyHostToDevice));

  double val = 1.234567891012345;
  dim3 grid(10000);
  dim3 block(items);
  AtomicTestMin<T><<<grid, block>>>(d_a, val);
  GetGpuErr(cudaPeekAtLastError());

  GetGpuErr(cudaMemcpy((void *)h_a, (void *)d_a, bytes, cudaMemcpyDeviceToHost));

  for (auto i = 0; i < 5; i++) {
    double tmp;
    if (sizeof(T) == 2) {
      tmp = __half2float(h_a[i]);
    } else {
      tmp = h_a[i];
    }
    printf("%.12f ", tmp);
  }
  printf("\n");

  GetGpuErr(cudaFree(d_a));
  free(h_a);
}

int main() {
  TestAtomicSum<float>();
  TestAtomicSum<double>();
  TestAtomicSum<half>();

  TestAtomicMax<float>();
  TestAtomicMax<double>();
  TestAtomicMax<half>();

  TestAtomicMin<float>();
  TestAtomicMin<double>();
  TestAtomicMin<half>();

  return 0;
}
