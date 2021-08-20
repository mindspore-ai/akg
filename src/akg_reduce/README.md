# AKG Reduce Lib

## 1. Introduction

AKG Reduce Lib is a reduction algorithms library for NVIDIA-GPU in AKG. AKG Reduce Lib aims to generate a better reduce kernels automatically by embedding template reduce code into the final CUDA code.

## 2. Features

- Focuses on 1D and 2D reduction in fixed shared memory size. (thanks to the reduce support in previous pass, like axis fusing)
- Uses "multi-blocks + atomic return" strategies which can utilize the benefit of GPU traits.  
- Provides an unified interface for cuda-code in codegen, which can satisfy diverse scenarios.

## 3. Usages

- The things you need to do are calling the function "AkgReduce" with kernel informations (dtype, reduce type, blockDim.x/y, reduction operator, etc) and "AkgAtomicReturn" for multi-block-reduce return. 

```Javascript
template <typename T, typename ReduceOp>
__global__ void Reduce1DMultiBlock(int x_len, T *arr, T *output, int item_per_thread, ReduceOp op) {
  T acc = 0.0;                  // aggregated value in current thread
  __shared__ T red_buf[32];     // buffer on shared memory for reduction computation 
  __shared__ T temp_output[1];  // temp storage for output
  temp_output[0] = (T)0.0;
  for (int k = 0; k < item_per_thread; ++k) {
    if (threadIdx.x + k * blockDim.x + blockIdx.x * blockDim.x * item_per_thread < x_len) {
      acc += arr[threadIdx.x + k * blockDim.x + blockIdx.x * blockDim.x * item_per_thread];
    }
  }
  __syncthreads();
  AkgReduce<T, ReduceOp, 32, 1, ALL_REDUCE>(op, &temp_output[0], red_buf, acc, 32);
  __syncthreads();
  if (threadIdx.x == 0) {
    AkgAtomicReturn<T, ReduceOp>(temp_output[0], &output[0], op);
  }
}
```

## 4. Updates

### 2021.8.20
- Fix bugs when using "shfl.down" function in irregular cases. Since "shfl.down" is a wrap-based functions but irregular reduction algorithms broke this rule. To safe reduction, we use volatile shared algorithm instead. 

### 2021.8.16
- Support ProdOp, AtomicProd. Now you can use akg-reduce-lib to implement a ReduceProd kernel.

### 2021.1.12
- Update the algorithms when those reduce-length are irregular (not the pow of 2). The new irregular reduction implementations have a better performance.


### 2021.1.8 

- Support one-dim-mapping feature. Now you can use one-dim kernel to describe reduce-2D cases by setting the real BlockDimX, BlockDimY on the AkgReduce interface.


### 2020.12.31 
- Use "shfl.down" function for warp-level reduce. (since "shfl" functions still don't support 1-btype dtype yet, we keep using "volatile shared" in bool or signed char cases).