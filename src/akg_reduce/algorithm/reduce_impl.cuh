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
#ifndef AKG_REDUCE_REDUCE_IMPL_H
#define AKG_REDUCE_REDUCE_IMPL_H
#include "./shared_reduce.cuh"

namespace akg_reduce {

// The implements for specific selected algorithms,
// Supports 1D & 2D (x-axis or y-axis) reduce.

/**
 * @brief The enter of all-reduce reduction
 * 
 * @tparam T            Dtype: half, float, double, int, signed char, bool
 * @tparam ReduceOp     Operators for reduce: SumOp, MaxOp, MinOp, AndOp, OrOp
 * @tparam BlockDimX    Real blockDim.x
 */
template <typename T, typename ReduceOp, size_t BlockDimX>
__inline__ __device__ void AllReduce(const ReduceOp op,  // The operator
                                     T *output,          // Addr of output
                                     T *shared_array,    // Temp array in shared memory
                                     const T acc         // Aggregated value in current thread
) {
  const int tx = blockDim.x * threadIdx.y + threadIdx.x;

  HalvedReduce1D<ReduceOp, BlockDimX>(shared_array, acc, op, tx, 0);

  if (tx == 0) {
    output[0] = op(output[0], shared_array[0]);
  }
}

/**
 * @brief  The enter of reduce2D-X reduction.
 * 
 * @tparam T          Dtype: half, float, double, int, signed char, bool
 * @tparam ReduceOp   Operators for reduce: SumOp, MaxOp, MinOp, AndOp, OrOp
 * @tparam BlockDimX  Real blockDim.x
 */
template <typename T, typename ReduceOp, size_t BlockDimX>
__inline__ __device__ void ReduceDirectionX(const ReduceOp op,  // The operator
                                            T *output,          // Addr of output
                                            T *shared_array,    // Temp array in shared memory
                                            const T acc         // Aggregated value in current thread
) {
  const int tx = (blockDim.x * threadIdx.y + threadIdx.x) % BlockDimX;
  const int ty = (blockDim.x * threadIdx.y + threadIdx.x) / BlockDimX;

  HalvedReduce2DX<ReduceOp, BlockDimX>(shared_array, acc, op, tx, ty);

  if (tx == 0) {
    output[0] = op(output[0], shared_array[ty * BlockDimX]);  // shared_array maybe a part of an array
  }
}

/**
 * @brief  The enter of reduce2D-Y reduction.
 * 
 * @tparam T          Dtype: half, float, double, int, signed char, bool
 * @tparam ReduceOp   Operators for reduce: SumOp, MaxOp, MinOp, AndOp, OrOp
 * @tparam BlockDimX  Real blockDim.x
 * @tparam BlockDimY  Real blockDim.y
 */
template <typename T, typename ReduceOp, size_t BlockDimX, size_t BlockDimY>
__inline__ __device__ void ReduceDirectionY(const ReduceOp op,     // The operator
                                            T *output,             // Addr of output
                                            T *shared_array,       // Temp array in shared memory
                                            const T acc,           // Aggregated value in current thread
                                            const int sharedmem_x  // Shared memory size of x axis
) {
  const int tx = (blockDim.x * threadIdx.y + threadIdx.x) % BlockDimX;
  const int ty = (blockDim.x * threadIdx.y + threadIdx.x) / BlockDimX;

  HalvedReduce2DY<ReduceOp, BlockDimX, BlockDimY>(shared_array, acc, op, sharedmem_x, tx, ty);

  if (ty == 0) {
    output[0] = op(output[0], shared_array[tx]);
  }
}

}  // namespace akg_reduce

#endif  // AKG_REDUCE_REDUCE_IMPL_H