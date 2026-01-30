import tilelang as tl
import tilelang.language as T
import torch

@tl.jit(out_idx=[-1])
def layer_norm_kernel_optimized(batch_size, features, dim1, dim2, block_size):
    
    @T.prim_func
    def main(x: T.Tensor((batch_size, features, dim1, dim2), "float16"), y: T.Tensor((batch_size, features, dim1, dim2), "float16")):
        
        total_size = features * dim1 * dim2

        with T.Kernel(batch_size, T.ceildiv(total_size, block_size), threads=block_size) as (sample_idx, bx):
            # 内存分配
            A_shared = T.alloc_shared((block_size,), "float32")
            A_pow_local = T.alloc_fragment((block_size,), "float32")
            A_powsum = T.alloc_fragment((1,), "float32")
            
            # 数据加载和计算
            for tid in T.Parallel(block_size):
                elem_idx = bx * block_size + tid
                
                if elem_idx < total_size:
                    c = elem_idx // (dim1 * dim2)
                    h = (elem_idx % (dim1 * dim2)) // dim2
                    w = elem_idx % dim2
                    input_val = x[sample_idx, c, h, w].astype("float32")
                    
                    A_shared[tid] = input_val
                    A_pow_local[tid] = input_val * input_val
                else:
                    A_shared[tid] = 0.0
                    A_pow_local[tid] = 0.0
            
            # ✅ 使用内置归约，避免同步和线程卡死
            T.reduce_sum(A_pow_local, A_powsum, dim=0)
            
            # 计算归一化因子
            for tid in T.Parallel(block_size):
                if tid == 0:
                    mean_val = A_powsum[0] / total_size
                    var_val = A_powsum[0] / total_size - mean_val * mean_val
                    A_powsum[0] = mean_val
                    A_powsum[1] = var_val
            
            # 应用归一化
            for tid in T.Parallel(block_size):
                elem_idx = bx * block_size + tid
                
                if elem_idx < total_size:
                    c = elem_idx // (dim1 * dim2)
                    h = (elem_idx % (dim1 * dim2)) // dim2
                    w = elem_idx % dim2
                    input_val = x[sample_idx, c, h, w].astype("float32")
                    
                    mean_val = A_powsum[0]
                    var_val = A_powsum[1]
                    normalized = (input_val - mean_val) / T.sqrt(var_val + 1e-5)
                    y[sample_idx, c, h, w] = normalized.astype("float16")

    return main

def layer_norm(input_tensor: torch.Tensor):
    
    batch_size = 16
    features = 64
    dim1 = 256
    dim2 = 256
    
    block_size = 256
    y = torch.zeros_like(input_tensor)
    kernel = layer_norm_kernel_optimized(batch_size, features, dim1, dim2, block_size)
    y = kernel(input_tensor)
    return y
