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

#ifndef PARIS_REDUCE_H
#define PARIS_REDUCE_H
#include "../akg_reduce/reduce.cuh"

namespace paris_reduce {

template <typename T, typename ReduceOp, size_t BlockSizeReduce>
__inline__ __device__ void ParisAllReduce(ReduceOp op, T *output, T *shared_array, T acc) {
  if (BlockSizeReduce > 32) {
    shared_array[threadIdx.x] = acc;
    __syncthreads();
    
    if (threadIdx.x < BlockSizeReduce / 2) {
      acc += shared_array[threadIdx.x + BlockSizeReduce / 2];
      shared_array[threadIdx.x] = acc;
    } 
    for (int delta = BlockSizeReduce / 4; delta > 16; delta /= 2) {
      __syncthreads();
      if (threadIdx.x <delta) {
        acc += shared_array[threadIdx.x + delta];
        shared_array[threadIdx.x] = acc;
      }
    }
  }

  if (BlockSizeReduce > 32) {
    if (threadIdx.x < 32) {
      acc += __shfl_down_sync(0xFFFFFFFF, acc , 16);
      acc += __shfl_down_sync(0xFFFFFFFF, acc , 8);
      acc += __shfl_down_sync(0xFFFFFFFF, acc , 4);
      acc += __shfl_down_sync(0xFFFFFFFF, acc , 2);
      acc += __shfl_down_sync(0xFFFFFFFF, acc , 1);
    }
  }

  if (BlockSizeReduce == 32) {
    acc += __shfl_down_sync(0xFFFFFFFF, acc , 16);
    acc += __shfl_down_sync(0xFFFFFFFF, acc , 8);
    acc += __shfl_down_sync(0xFFFFFFFF, acc , 4);
    acc += __shfl_down_sync(0xFFFFFFFF, acc , 2);
    acc += __shfl_down_sync(0xFFFFFFFF, acc , 1);
  }

  if (BlockSizeReduce == 16) {
    acc += __shfl_down_sync(0xFFFFFFFF, acc , 8);
    acc += __shfl_down_sync(0xFFFFFFFF, acc , 4);
    acc += __shfl_down_sync(0xFFFFFFFF, acc , 2);
    acc += __shfl_down_sync(0xFFFFFFFF, acc , 1);
  }

  if (BlockSizeReduce == 8) {
    acc += __shfl_down_sync(0xFFFFFFFF, acc , 4);
    acc += __shfl_down_sync(0xFFFFFFFF, acc , 2);
    acc += __shfl_down_sync(0xFFFFFFFF, acc , 1);
  }
  
  if (BlockSizeReduce == 4) {
    acc += __shfl_down_sync(0xFFFFFFFF, acc , 2);
    acc += __shfl_down_sync(0xFFFFFFFF, acc , 1);
  }

  if (BlockSizeReduce == 2) {
    acc += __shfl_down_sync(0xFFFFFFFF, acc , 1);
  }

  if (((int)threadIdx.x) == 0) {
    output[0] = acc;
  }

}

template <typename T, typename ReduceOp, size_t BlockSizeReduce>
__inline__ __device__ void ParisReduceX(ReduceOp op, T *output, T *shared_array, T acc) {

  if (BlockSizeReduce > 32) {
    shared_array[threadIdx.x + threadIdx.y * BlockSizeReduce] = acc;
    __syncthreads();
    if (threadIdx.x < BlockSizeReduce / 2) {
      acc += shared_array[(threadIdx.x +BlockSizeReduce / 2) + threadIdx.y * BlockSizeReduce];
      shared_array[threadIdx.x + threadIdx.y * BlockSizeReduce] = acc;
    }
    for (int delta = BlockSizeReduce / 4; delta > 16; delta /= 2) {
      __syncthreads();
      if (threadIdx.x <delta) {
        acc += shared_array[(threadIdx.x + delta) + threadIdx.y * BlockSizeReduce];
        shared_array[threadIdx.x + threadIdx.y * BlockSizeReduce] = acc;
      }
    }
  }

  if (BlockSizeReduce > 32) {
    if (threadIdx.x < 32) {
      acc += __shfl_down_sync(0xFFFFFFFF, acc , 16);
      acc += __shfl_down_sync(0xFFFFFFFF, acc , 8);
      acc += __shfl_down_sync(0xFFFFFFFF, acc , 4);
      acc += __shfl_down_sync(0xFFFFFFFF, acc , 2);
      acc += __shfl_down_sync(0xFFFFFFFF, acc , 1);
    }
  }

  if (BlockSizeReduce == 32) {
    acc += __shfl_down_sync(0xFFFFFFFF, acc , 16);
    acc += __shfl_down_sync(0xFFFFFFFF, acc , 8);
    acc += __shfl_down_sync(0xFFFFFFFF, acc , 4);
    acc += __shfl_down_sync(0xFFFFFFFF, acc , 2);
    acc += __shfl_down_sync(0xFFFFFFFF, acc , 1);
  }

  if (BlockSizeReduce == 16) {
    acc += __shfl_down_sync(0xFFFFFFFF, acc , 8);
    acc += __shfl_down_sync(0xFFFFFFFF, acc , 4);
    acc += __shfl_down_sync(0xFFFFFFFF, acc , 2);
    acc += __shfl_down_sync(0xFFFFFFFF, acc , 1);
  }

  if (BlockSizeReduce == 8) {
    acc += __shfl_down_sync(0xFFFFFFFF, acc , 4);
    acc += __shfl_down_sync(0xFFFFFFFF, acc , 2);
    acc += __shfl_down_sync(0xFFFFFFFF, acc , 1);
  }
  
  if (BlockSizeReduce == 4) {
    acc += __shfl_down_sync(0xFFFFFFFF, acc , 2);
    acc += __shfl_down_sync(0xFFFFFFFF, acc , 1);
  }

  if (BlockSizeReduce == 2) {
    acc += __shfl_down_sync(0xFFFFFFFF, acc , 1);
  }

  if (((int)threadIdx.x) == 0) {
    output[0] = acc;
  }

}

template <typename T, typename ReduceOp, size_t BlockSizeReduce>
__inline__ __device__ void ParisReduceY(ReduceOp op, T *output, T *shared_array, T acc) {
  shared_array[threadIdx.x * blockDim.y + threadIdx.y] = acc;
  for (int delta = blockDim.y / 2; delta > 0; delta /= 2) {
    __syncthreads();
    if (threadIdx.y < delta) {
      shared_array[threadIdx.x * blockDim.y + threadIdx.y] +=
        shared_array[threadIdx.x * blockDim.y + threadIdx.y + delta];
    }
  }

  if (threadIdx.y == 0) {
    output[0] = shared_array[threadIdx.x * BlockSizeReduce];
  }
}

template <typename T, typename ReduceOp, size_t BlockDimX, size_t BlockDimY, int ReduceType>
__inline__ __device__ void ParisReduce(
  ReduceOp op,         // the operator
  T *output_array,     // the addr of output in global/shared memory, single value
  T *shared_array,     // the temp array in shared memory
  T acc,               // aggregated value in current thread
  int unused = 0       // unused valiable only for aligning with akg_reduce lib
) {
  if (ReduceType == akg_reduce::ALL_REDUCE) {
    ParisAllReduce<T, ReduceOp, BlockDimX>(op, output_array, shared_array, acc);
    return;
  }

  if (ReduceType == akg_reduce::REDUCE2D_X) {
    ParisReduceX<T, ReduceOp, BlockDimX>(op, output_array, shared_array, acc);
  }

  if (ReduceType == akg_reduce::REDUCE2D_Y) {
    ParisReduceY<T, ReduceOp, BlockDimY>(op, output_array, shared_array, acc);
  }
}

template <typename T, typename ReduceOp>
__device__ __forceinline__ void ParisReturn(T shared_result, T *output, ReduceOp op) {
  akg_reduce::AkgAtomicReturn<T, ReduceOp>(shared_result, output, op);
}

}  // namespace paris_reduce

#endif // PARIS_REDUCE_H
