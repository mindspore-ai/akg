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

#ifndef AKG_REDUCE_H
#define AKG_REDUCE_H
#include "./utils/util.cuh"
#include "./algorithm/reduce_impl.cuh"
#include "./operators/reduce_operators.cuh"

namespace akg_reduce {
/**
 * Main functions of reduce module
 */

/**
 * @brief This function pick up the proper strategies for different kinds of cases automatically.

 * @tparam T                  Dtype: half, float, double, int, signed char, bool
 * @tparam ReduceOp           Operators for reduce: SumOp, MaxOp, MinOp, AndOp, OrOp
 * @tparam BlockDimX          Real blockDim.x
 * @tparam BlockDimY          Real blockDim.y
 * @tparam ReduceType         Types of reduce: ALL_REDUCE(0), REDUCE2D_X(1), REDUCE2D_Y(2)
 */
template <typename T, typename ReduceOp, size_t BlockDimX, size_t BlockDimY, int ReduceType>
__inline__ __device__ void AkgReduce(const ReduceOp op,         // The operator
                                     T *output_array,           // Addr of output in global/shared memory
                                     T *shared_array,           // Temp array in shared memory
                                     const T acc,               // Aggregated value in current thread
                                     const int sharedmem_x = 0  // Shared memory size of x axis, especially used for reduce2D along y.
) {
  // all-reduce
  if (ReduceType == ALL_REDUCE) {
    AllReduce<T, ReduceOp, BlockDimX>(op, output_array, shared_array, acc);
    return;
  }

  // reduce data from direction x
  if (ReduceType == REDUCE2D_X) {
    ReduceDirectionX<T, ReduceOp, BlockDimX>(op, output_array, shared_array, acc);
    return;
  }

  // reduce data from direction y
  if (ReduceType == REDUCE2D_Y) {
    ReduceDirectionY<T, ReduceOp, BlockDimX, BlockDimY>(op, output_array, shared_array, acc, sharedmem_x);
    return;
  }
}

/**
 * @brief Accumulation with kahan algorithm, only for sum operator
 * @tparam T                  Dtype: half, float, double, int;
 */
 template <typename T>
 __device__ __forceinline__ void AkgKahanAccumulation(T *y, 
                                                      T *t,
                                                      T *c,
                                                      T *acc,
                                                      const T input
 ) {
    y[0] = input - c[0];
    t[0] = acc[0] + y[0];
    c[0] = (t[0] - acc[0]) - y[0];
    acc[0] = t[0];
 }
 

/**
 * @brief Atomic return function, from shared memory to global memory
 * @tparam T                  Dtype: half, float, double, int, signed char, bool;
 * @tparam ReduceOp           Operators for reduce: SumOp, MaxOp, MinOp, AndOp, OrOp, ProdOp;
 */
template <typename T, typename ReduceOp>
__device__ __forceinline__ void AkgAtomicReturn(const T shared_result, // Reduction result on the shared memory
                                                T *output,             // Global output address
                                                const ReduceOp op      // The operator
) {
  AtomicOp<T, op.identifier> atomic_op;
  atomic_op.Compute(&output[0], shared_result);
}

}  // namespace akg_reduce

#endif  // AKG_REDUCE_H
