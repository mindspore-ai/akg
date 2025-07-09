# NPU硬件说明
## memory_system
├── gm (GlobalMemory/DeviceMemory):
|    └── size: 32GB
├── L2Cache: // 全部核共用
|    └── size: 96MB
├── Buffers(per ai_core) x 20: // 整个芯片有20个ai_core
|    ├── vector_core:
|    |    ├── 数量: 2块 // 每个ai_core有2个vector_core，所以一共有40个vector_core可同时数据搬运、计算等
|    |    └── unified_buffer:
|    |         ├── size: 192KB
|    |         └── data_align: 256B
|    └── cube_core:
|         ├── 数量: 1块 // 每个ai_core有1个cube_core，所以一共有20个cube_core，可同时数据搬运、计算等
|         ├── L1_buffer:
|         |    ├── size: 1MB
|         |    └── data_align: 256B
|         └── L0_buffers: // matmul计算专用缓冲区
|              ├── L0A_buffer: // 存储矩阵A的数据
|              |    ├── size: 64KB
|              |    ├── 用途: 存储输入矩阵A的m0xk0块数据
|              |    └── data_align: 256B
|              ├── L0B_buffer: // 存储矩阵B的数据
|              |    ├── size: 64KB
|              |    ├── 用途: 存储输入矩阵B的k0xn0块数据
|              |    └── data_align: 256B
|              └── L0C_buffer: // 存储矩阵C的结果
|                   ├── size: 128KB
|                   ├── 用途: 存储输出矩阵C的m0xn0块结果
|                   └── data_align: 256B
          

## compute_system 
├── vector_compute_unit:
|    ├── 约束: 需要unified_buffer能放下
|    ├── 约束: 只能做连续、带mask的vector计算
|    └── 功能: 常规的vector计算，带mask vector计算等
├── cube_compute_unit:
|    ├── 约束1: 需要m0xk0xsizeof(A.dtype) < L0A_buffer_size(64KB)
|    ├── 约束2: 需要n0xk0xsizeof(B.dtype) < L0B_buffer_size(64KB)
|    ├── 约束3: 需要m0xn0xsizeof(C.dtype==fp32) < L0C_buffer_size(128KB)
|    ├── 约束: 专门用于矩阵乘法等tensor计算
|    └── 功能: 高效的矩阵乘法，一次完成一个m0xk0xn0的矩阵乘法
