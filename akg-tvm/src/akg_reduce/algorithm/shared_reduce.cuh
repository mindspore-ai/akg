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

#ifndef AKG_REDUCE_SHARED_REDUCE_H
#define AKG_REDUCE_SHARED_REDUCE_H
#include "../utils/util.cuh"
#include "../operators/reduce_operators.cuh"

/*********************************************************
 * Algorithms of reduction computation in shared memory.
 * ********************************************************/
namespace akg_reduce {

/**
 * \brief Reduction in a warp using shfl functions. This func doesn't save when BlockDimX
 * isn't 2^X (X >= 0).
 *
 * \par
 * - Supports 1D or 2D reduction computation. The reduction direction is along x-axis.
 * - Exclude cases when T == bool/signed char, since shfl.sync funcs only support 16 bits,
 * - 32 bits and 64 bits.
 *
 * \tparam ReduceOp          Reduce operator type
 * \tparam BlockDimX         Real blockDim.x
 * \tparam T                 Dtype of reduction
 **/
template <typename ReduceOp, size_t BlockDimX, typename T>
__device__ __forceinline__ void WarpReduceShfl(T *shared_buf,     // Shared memory buffer
                                           const ReduceOp op, // Reduce operator
                                           const int tx = 0,  // Real threadIdx.x
                                           const int ty = 0   // Real threadIdx.y
) {
  const int tid = ty * BlockDimX + tx;
  T local_sum = shared_buf[tid];
  if (BlockDimX >= 32) {
    local_sum = op(local_sum, __shfl_down_sync(0xFFFFFFFF, local_sum, 16));
  }
  if (BlockDimX >= 16) {
    local_sum = op(local_sum, __shfl_down_sync(0xFFFFFFFF, local_sum, 8));
  }
  if (BlockDimX >= 8) {
    local_sum = op(local_sum, __shfl_down_sync(0xFFFFFFFF, local_sum, 4));
  }
  if (BlockDimX >= 4) {
    local_sum = op(local_sum, __shfl_down_sync(0xFFFFFFFF, local_sum, 2));
  }
  if (BlockDimX >= 2) {
    local_sum = op(local_sum, __shfl_down_sync(0xFFFFFFFF, local_sum, 1));
  }
  if (tx == 0) {
    shared_buf[tid] = local_sum;
  }
}

/**
 * \brief Reduction in a warp for all dtype using volatile shared algorithm.
 * This func is safe for all cases but little bit slower than WarpReduceShfl.
 *
 * \par
 * - Supports 1D or 2D reduction computation. The reduction direction is along x-axis.
 *
 * \tparam ReduceOp          Reduce operator type
 * \tparam BlockDimX         Real blockDim.x
 * \tparam T                 Dtype of reduction
 **/
template <typename ReduceOp, size_t BlockDimX, size_t UpperBound, typename T>
__device__ __forceinline__ void WarpReduceSafe(T *shared_buf,    // Shared memory buffer
                                                 const ReduceOp op, // Reduce operator
                                                 const int tx = 0,  // Real threadIdx.x
                                                 const int ty = 0   // Real threadIdx.y
) {
  const int tid = ty * BlockDimX + tx;
  if (UpperBound >= 32) {
    if (tx < 16)
      ((volatile T *)shared_buf)[tid] =
        op(((volatile T *)shared_buf)[tid], ((volatile T *)shared_buf)[tid + 16]);
  }
  __syncthreads();
  if (UpperBound >= 16) {
    if (tx < 8)
      ((volatile T *)shared_buf)[tid] =
        op(((volatile T *)shared_buf)[tid], ((volatile T *)shared_buf)[tid + 8]);
  }
  __syncthreads();
  if (UpperBound >= 8) {
    if (tx < 4)
      ((volatile T *)shared_buf)[tid] =
        op(((volatile T *)shared_buf)[tid], ((volatile T *)shared_buf)[tid + 4]);
  }
  __syncthreads();
  if (UpperBound >= 4) {
    if (tx < 2)
      ((volatile T *)shared_buf)[tid] =
        op(((volatile T *)shared_buf)[tid], ((volatile T *)shared_buf)[tid + 2]);
  }
  __syncthreads();
  if (UpperBound >= 2) {
    if (tx < 1)
      ((volatile T *)shared_buf)[tid] =
        op(((volatile T *)shared_buf)[tid], ((volatile T *)shared_buf)[tid + 1]);
  }
}

/**
 * \brief Reduction in a block along x axis.
 *
 * \par
 * - Supports 1D or 2D reduction computation. The reduction direction is along x-axis.
 *
 * \tparam ReduceOp          Reduce operator type
 * \tparam BlockDimX         Real blockDim.x
 * \tparam UpperBound        Lenght of x-axis after cur irregular shape
 * \tparam T                 Dtype of reduction
 *
 **/
template <typename ReduceOp, size_t BlockDimX, size_t UpperBound, typename T>
__device__ __forceinline__ void ReduceXInBlock(T *shared_buf,     // Shared memory buffer.
                                              const ReduceOp op, // Reduce operator.
                                              const int tx = 0,  // Real threadIdx.x
                                              const int ty = 0   // Real threadIdx.y
) {
  const int tid = ty * BlockDimX + tx;
  if (UpperBound >= 1024) {
    if (tx < 512) {
      shared_buf[tid] = op(shared_buf[tid], shared_buf[tid + 512]);
    }
    __syncthreads();
  }
  if (UpperBound >= 512) {
    if (tx < 256) {
      shared_buf[tid] = op(shared_buf[tid], shared_buf[tid + 256]);
    }
    __syncthreads();
  }
  if (UpperBound >= 256) {
    if (tx < 128) {
      shared_buf[tid] = op(shared_buf[tid], shared_buf[tid + 128]);
    }
    __syncthreads();
  }
  if (UpperBound >= 128) {
    if (tx < 64) {
      shared_buf[tid] = op(shared_buf[tid], shared_buf[tid + 64]);
    }
    __syncthreads();
  }
  if (UpperBound >= 64) {
    if (tx < 32) {
      shared_buf[tid] = op(shared_buf[tid], shared_buf[tid + 32]);
    }
  }
  __syncthreads();
  // choose proper algorithm for different scenarios.
  if (BlockDimX == UpperBound) {
    if (tx < 32)
      WarpReduceShfl<ReduceOp, BlockDimX>(shared_buf, op, tx, ty);
  } else {
    WarpReduceSafe<ReduceOp, BlockDimX, UpperBound>(shared_buf, op, tx, ty);
  }
  __syncthreads();
}

/**
 * \brief Reduction in a block along y axis.
 *
 * \par
 * - Support 2D reduction.
 * - Shared memory size along x axis could be set as proper odd to avoid bank conflicts.
 *
 * \tparam T                 Dtype of reduction
 * \tparam ReduceOp          Reduce operator type
 * \tparam BlockDimX         Real blockDim.x
 * \tparam BlockDimY         Real blockDim.y
 *
 **/
template <typename ReduceOp, size_t BlockDimX, size_t BlockDimY, typename T>
__device__ __forceinline__ void ReduceYInBlock(T *shared_buf,         // Shared memory buffer
                                              const ReduceOp op,      // Reduce operator
                                              const int sharedmem_x,  // Shared memory size of x axis
                                              const int tx = 0,       // Real threadIdx.x
                                              const int ty = 0        // Real threadIdx.y
) {
  const int tid = ty * sharedmem_x + tx;
  if (BlockDimY >= 1024) {
    if (ty < 512) {
      shared_buf[tid] = op(shared_buf[tid], shared_buf[tid + 512 * sharedmem_x]);
    }
    __syncthreads();
  }
  if (BlockDimY >= 512) {
    if (ty < 256) {
      shared_buf[tid] = op(shared_buf[tid], shared_buf[tid + 256 * sharedmem_x]);
    }
    __syncthreads();
  }
  if (BlockDimY >= 256) {
    if (ty < 128) {
      shared_buf[tid] = op(shared_buf[tid], shared_buf[tid + 128 * sharedmem_x]);
    }
    __syncthreads();
  }
  if (BlockDimY >= 128) {
    if (ty < 64) {
      shared_buf[tid] = op(shared_buf[tid], shared_buf[tid + 64 * sharedmem_x]);
    }
    __syncthreads();
  }
  if (BlockDimY >= 64) {
    if (ty < 32) {
      shared_buf[tid] = op(shared_buf[tid], shared_buf[tid + 32 * sharedmem_x]);
    }
    __syncthreads();
  }
  if (BlockDimY >= 32) {
    if (ty < 16) {
      shared_buf[tid] = op(shared_buf[tid], shared_buf[tid + 16 * sharedmem_x]);
    }
    __syncthreads();
  }
  if (BlockDimY >= 16) {
    if (ty < 8) {
      shared_buf[tid] = op(shared_buf[tid], shared_buf[tid + 8 * sharedmem_x]);
    }
    __syncthreads();
  }
  if (BlockDimY >= 8) {
    if (ty < 4) {
      shared_buf[tid] = op(shared_buf[tid], shared_buf[tid + 4 * sharedmem_x]);
    }
    __syncthreads();
  }
  if (BlockDimY >= 4) {
    if (ty < 2) {
      shared_buf[tid] = op(shared_buf[tid], shared_buf[tid + 2 * sharedmem_x]);
    }
    __syncthreads();
  }
  if (BlockDimY >= 2) {
    if (ty < 1) {
      shared_buf[tid] = op(shared_buf[tid], shared_buf[tid + 1 * sharedmem_x]);
    }
    __syncthreads();
  }
}

/**
 * \brief All-Reduce implementations with halved reduce algorithm.
 *
 * \par
 * - Support single-block reduction and multi-block reduction.
 * - Support 1D reduction.
 *
 * \tparam ReduceOp          Reduce operator type
 * \tparam BlockDimX         Real blockDim.x
 * \tparam T                 Dtype of reduction
 *
 **/
template <typename ReduceOp, size_t BlockDimX, typename T>
__device__ __forceinline__ void HalvedReduce1D(T *shared_buf,     // Shared memory buffer
                                               const T local_acc, // Aggregated value in current thread
                                               const ReduceOp op, // Reduce operator
                                               const int tx = 0,  // Real threadIdx.x
                                               const int ty = 0   // Real threadIdx.y
) {
                                                
  // load data to shared memory.
  shared_buf[tx] = local_acc;
  __syncthreads();

  const int tid = tx;
  if (IsPowOfTwo(BlockDimX)) {
    // Using unroll strategy.
    ReduceXInBlock<ReduceOp, BlockDimX, BlockDimX>(shared_buf, op, tx, ty);
  } else {

    constexpr int UpperBound = GetUpperBound(BlockDimX);

    if (tx + UpperBound < BlockDimX) {
      shared_buf[tid] = op(shared_buf[tid + UpperBound], shared_buf[tid]);
    }
    __syncthreads();
    
    ReduceXInBlock<ReduceOp, BlockDimX, UpperBound>(shared_buf, op, tx, ty);
  }
}

/**
 * \brief Reduce2D along x axis with halved reduce algorithm.
 *
 * \par
 * - Reduction direction is x axis.
 * - support single-block reduction and multi-blocks reduction.
 * - support 2D reduction.
 *
 * \tparam ReduceOp          Reduce operator type
 * \tparam BlockDimX         Real blockDim.x
 * \tparam T                 Dtype of reduction
 *
 **/

template <typename ReduceOp, size_t BlockDimX, typename T>
__device__ __forceinline__ void HalvedReduce2DX(T *shared_buf,     // Shared memory buffer
                                                const T local_acc, // Aggregated value in current thread
                                                const ReduceOp op, // Reduce operator
                                                const int tx = 0,  // Real threadIdx.x
                                                const int ty = 0   // Real threadIdx.y
) {
  const int tid = ty * BlockDimX + tx;

  // load data to shared memory.
  shared_buf[tid] = local_acc;
  __syncthreads();

  if (IsPowOfTwo(BlockDimX)) {
    ReduceXInBlock<ReduceOp, BlockDimX, BlockDimX>(shared_buf, op, tx, ty);
  } else {
    constexpr int UpperBound = GetUpperBound(BlockDimX);

    if (tx + UpperBound < BlockDimX) {
      shared_buf[tid] = op(shared_buf[tid + UpperBound], shared_buf[tid]);
    }
    __syncthreads();
    
    ReduceXInBlock<ReduceOp, BlockDimX, UpperBound>(shared_buf, op, tx, ty);
  }
}

/**
 * \brief Reduce2D along y axis with halved reduce algorithm.
 *
 * \par
 * - Reduction direction is y axis.
 * - support single-block reduction and multi-block reduction.
 * - support 2D reduction.
 * - support set sharedmem_x as a proper odd to avoid bank conflicts.
 *
 * \tparam T                 Dtype of reduction
 * \tparam ReduceOp          Reduce operator type
 * \tparam BlockDimX         Real blockDim.x
 * \tparam BlockDimY         Real blockDim.y
 *
 **/

template <typename ReduceOp, size_t BlockDimX, size_t BlockDimY, typename T>
__device__ __forceinline__ void HalvedReduce2DY(T *shared_buf,         // Shared memory buffer
                                                const T local_acc,     // Aggregated value in current thread
                                                const ReduceOp op,     // Reduce operator
                                                const int sharedmem_x, // Shared memory size of x axis
                                                const int tx = 0,      // Real threadIdx.x
                                                const int ty = 0       // Real threadIdx.y
) {
  const int tid = ty * sharedmem_x + tx;
  shared_buf[tid] = local_acc;
  __syncthreads();

  if (IsPowOfTwo(BlockDimY)) {
    ReduceYInBlock<ReduceOp, BlockDimX, BlockDimY>(shared_buf, op, sharedmem_x, tx, ty);
  } else {
    constexpr int UpperBound = GetUpperBound(BlockDimY);

    if (ty + UpperBound < BlockDimY) {
      shared_buf[tid] = op(shared_buf[tid + UpperBound * sharedmem_x], shared_buf[tid]);
    }
    __syncthreads();
    
    ReduceYInBlock<ReduceOp, BlockDimX, UpperBound>(shared_buf, op, sharedmem_x, tx, ty);
  }
}
}  // namespace akg_reduce

#endif  // AKG_REDUCE_SHARED_REDUCE_H