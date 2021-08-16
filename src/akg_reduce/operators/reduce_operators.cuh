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

#ifndef AKG_REDUCE_REDUCE_OPERATORS_H
#define AKG_REDUCE_REDUCE_OPERATORS_H

#include <cuda_fp16.h>

namespace akg_reduce {

/*
  AkgReduce supports Sum, Max, Min, And(logical), Or(logical), Prod
*/
struct SumOp {
  // "sum" operator.
  template <typename T>
  __device__ __forceinline__ T operator()(const T &a, const T &b) const {
    return a + b;
  }

  // NOTE: operator "+" doesn't support const volatile half + const volatile half, we cast them.
  __device__ __forceinline__ volatile half operator()(const volatile half &a, const volatile half &b) const {
    return __hadd(((const half)a), ((const half)b));
  }
  const static int identifier = 0;
};

struct MaxOp {
  // "max" operator.
  template <typename T>
  __device__ __forceinline__ T operator()(const T &a, const T &b) const {
    return (b > a) ? (b) : (a);
  }
  __device__ __forceinline__ volatile half operator()(const volatile half &a, const volatile half &b) const {
    return (__hgt(((const half)a), ((const half)b))) ? (a) : (b);
  }
  const static int identifier = 1;
};

struct MinOp {
  // "min" operator
  template <typename T>
  __device__ __forceinline__ T operator()(const T &a, const T &b) const {
    return (b > a) ? (a) : (b);
  }
  __device__ __forceinline__ volatile half operator()(const volatile half &a, const volatile half &b) const {
    return (__hgt(((const half)a), ((const half)b))) ? (b) : (a);
  }
  const static int identifier = 2;
};

struct AndOp {
  // "and" operator(logical), only supports dtype bool or signed char
  template <typename T, typename Y>
  __device__ __forceinline__ T operator()(const T &a, const Y &b) const {
    // in logical "and" operator, the dtype must be bool, please check it.
    return a && b;
  }

  template <typename T, typename Y>
  __device__ __forceinline__ T operator()(const volatile T &a, const volatile Y &b) const {
    return a && b;
  }

  const static int identifier = 3;
};

struct OrOp {
  // "or" operator(logical), only supports dtype bool or signed char
  template <typename T, typename Y>
  __device__ __forceinline__ T operator()(const T &a, const Y &b) const {
    // in logical "or" operator, the dtype must be bool, please check it.
    return a || b;
  }

  template <typename T, typename Y>
  __device__ __forceinline__ T operator()(const volatile T &a, const volatile Y &b) const {
    return a || b;
  }

  const static int identifier = 4;
};

struct ProdOp {
  // "prod" operator.
  template <typename T>
  __device__ __forceinline__ T operator()(const T &a, const T &b) const {
    return a * b;
  }

  // NOTE: operator "*" doesn't support const volatile half * const volatile half, we cast them.
  __device__ __forceinline__ volatile half operator()(const volatile half &a, const volatile half &b) const {
    return __hmul(((const half)a), ((const half)b));
  }
  const static int identifier = 5;
};

// Implement of AtomicMax,Min for float by AtomicCAS
// since we don't need to final result in thread, we return void;

// NOTE:  any atomic operation can be implemented based on atomicCAS()
// REF: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html

template <typename T>
__device__ void AtomicMax(T *const addr, const T val) {
  atomicMax(addr, val);
}

// NOTE: atmoicMax/Min supports some dtypes(int ,unsigned int,
// unsigned long long int), no need to re-implement again.

/**
 * @brief atomic return function for MaxOp
 *
 * @param addr return address in global memory
 * @param val  the value in shared memory
 */
__device__ void atomicMax(float *const addr, const float val) {
  if (*addr >= val) return;

  unsigned int *const addr_as_ui = (unsigned int *)addr;
  unsigned int old = *addr_as_ui, assumed;
  do {
    assumed = old;
    if (__uint_as_float(assumed) >= val) break;
    old = atomicCAS(addr_as_ui, assumed, __float_as_uint(val));
  } while (assumed != old);
}

__device__ void atomicMax(double *const addr, const double val) {
  if (*addr >= val) return;

  // NOTE: since atomicCAS only support unsigned long long int, but no signed version,
  // we use unint64 to support our transformation. In fact, __longlong_as_double is a
  // function only for reinterpret, signed or unsigned don't change the bits in this address.
  unsigned long long int *const addr_as_ull = (unsigned long long int *)addr;
  unsigned long long int old = *addr_as_ull, assumed;
  do {
    assumed = old;
    if (__longlong_as_double(assumed) >= val) break;
    old = atomicCAS(addr_as_ull, assumed, __double_as_longlong(val));
  } while (assumed != old);
}

#if __CUDA_ARCH__ >= 700
__device__ void atomicMax(half *const addr, const half val) {
  if (__hge(*addr, val)) return;
  unsigned short int *const addr_as_usi = (unsigned short int *)addr;
  unsigned short int old = *addr_as_usi, assumed;
  do {
    assumed = old;
    if (__hge(__ushort_as_half(assumed), val)) break;
    old = atomicCAS(addr_as_usi, assumed, __half_as_ushort(val));
  } while (assumed != old);
}
#endif

template <typename T>
__device__ void AtomicMin(T *const addr, const T val) {
  atomicMin(addr, val);
}

// atomicMin: float, double, half
__device__ void atomicMin(float *const addr, const float val) {
  if (*addr <= val) return;

  unsigned int *const addr_as_ui = (unsigned int *)addr;
  unsigned int old = *addr_as_ui, assumed;
  do {
    assumed = old;
    if (__uint_as_float(assumed) <= val) break;
    old = atomicCAS(addr_as_ui, assumed, __float_as_uint(val));
  } while (assumed != old);
}

__device__ void atomicMin(double *const addr, const double val) {
  if (*addr <= val) return;

  unsigned long long int *const addr_as_ull = (unsigned long long int *)addr;
  unsigned long long int old = *addr_as_ull, assumed;
  do {
    assumed = old;
    if (__longlong_as_double(assumed) <= val) break;
    old = atomicCAS(addr_as_ull, assumed, __double_as_longlong(val));
  } while (assumed != old);
}

#if __CUDA_ARCH__ >= 700
__device__ void atomicMin(half *const addr, const half val) {
  if (__hle(*addr, val)) return;

  unsigned short int *const addr_as_usi = (unsigned short int *)addr;
  unsigned short int old = *addr_as_usi, assumed;
  do {
    assumed = old;
    if (__hle(__ushort_as_half(assumed), val)) break;
    old = atomicCAS(addr_as_usi, assumed, __half_as_ushort(val));
  } while (assumed != old);
}
#endif

// atomicAnd: int
__device__ void atomicAnd(int *const addr, const int val) {
  unsigned int *const addr_as_ui = (unsigned int *)addr;
  unsigned int old = *addr_as_ui, assumed;
  do {
    assumed = old;
    old = atomicCAS(addr_as_ui, assumed, (unsigned int)(val && int(assumed)));
  } while (assumed != old);
}

// atomicOr: int
__device__ void atomicOr(int *const addr, const int val) {
  unsigned int *const addr_as_ui = (unsigned int *)addr;
  unsigned int old = *addr_as_ui, assumed;
  do {
    assumed = old;
    old = atomicCAS(addr_as_ui, assumed, (unsigned int)(val || int(assumed)));
  } while (assumed != old);
}

/**
 * @brief atomic return function for ProdOp
 *
 * @param addr return address in global memory
 * @param val  the value in shared memory
 */
template <typename T>
__device__ void AtomicProd(T *const addr, const T val) {
  AtomicProd(addr, val);
}

__device__ void AtomicProd(int *const addr, const int val) {
  unsigned int *const addr_as_ui = (unsigned int *)addr;
  unsigned int old = *addr_as_ui, assumed;
  do {
    assumed = old;
    old = atomicCAS(addr_as_ui, assumed, (unsigned int)(val * int(assumed)));
  } while (assumed != old);
}

__device__ void AtomicProd(float *const addr, const float val) {
  unsigned int *const addr_as_ui = (unsigned int *)addr;
  unsigned int old = *addr_as_ui, assumed;
  do {
    assumed = old;
    old = atomicCAS(addr_as_ui, assumed, __float_as_uint(val * __uint_as_float(assumed)));
  } while (assumed != old);
}

__device__ void AtomicProd(double *const addr, const double val) {
  // NOTE: since atomicCAS only support unsigned long long int, but no signed version,
  // we use unint64 to support our transformation. In fact, __longlong_as_double is a
  // function only for reinterpret, signed or unsigned don't change the bits in this address.
  unsigned long long int *const addr_as_ull = (unsigned long long int *)addr;
  unsigned long long int old = *addr_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(addr_as_ull, assumed, __double_as_longlong(val * __longlong_as_double(assumed)));
  } while (assumed != old);
}

#if __CUDA_ARCH__ >= 700
__device__ void AtomicProd(half *const addr, const half val) {
  unsigned short int *const addr_as_usi = (unsigned short int *)addr;
  unsigned short int old = *addr_as_usi, assumed;
  do {
    assumed = old;
    old = atomicCAS(addr_as_usi, assumed, __half_as_ushort(val * __ushort_as_half(assumed)));
  } while (assumed != old);
}
#endif


// AtomicOp for diverse atomic ops by identifier
// Atomic sample in cuda: int atomicMax(int* address, int val);
template <typename T, int identifier>
struct AtomicOp;

template <typename T>
struct AtomicOp<T, 0> {
  __device__ __forceinline__ void Compute(T *global_addr, T value) { atomicAdd(global_addr, value); }
};

template <typename T>
struct AtomicOp<T, 1> {
  __device__ __forceinline__ void Compute(T *global_addr, T value) { atomicMax(global_addr, value); }
};

template <typename T>
struct AtomicOp<T, 2> {
  __device__ __forceinline__ void Compute(T *global_addr, T value) { atomicMin(global_addr, value); }
};

template <typename T>
struct AtomicOp<T, 3> {
  __device__ __forceinline__ void Compute(T *global_addr, T value) { atomicAnd(global_addr, value); }
};

template <typename T>
struct AtomicOp<T, 4> {
  __device__ __forceinline__ void Compute(T *global_addr, T value) { atomicOr(global_addr, value); }
};

template <typename T>
struct AtomicOp<T, 5> {
  __device__ __forceinline__ void Compute(T *global_addr, T value) { AtomicProd(global_addr, value); }
};

}  // namespace akg_reduce

#endif  // AKG_REDUCE_REDUCE_OPERATORS_H
