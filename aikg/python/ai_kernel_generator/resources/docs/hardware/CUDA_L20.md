# GPU硬件说明 - NVIDIA L20

## memory_system
├── gm (GlobalMemory/DeviceMemory):
|    └── size: 48GB GDDR6
├── shared_memory (per SM): // L20基于Ada Lovelace架构
|    └── shared_memory_block:
|         ├── 大小: 100KB (最大可用)
|         ├── from_data: ["gm"]
|         └── to_data: ["gm"]
└── registers (per SM):
     ├── 数量: 65536个32位寄存器
     ├── 最大线程数: 2048 (每SM)
     └── 最大blocks: 24 (每SM)

## compute_system
├── cuda_cores: // CUDA计算核心
|    ├── 数量: 待确认 (基于Ada Lovelace架构)
|    ├── 约束: 每个warp 32个线程同步执行
|    ├── 约束: 分支发散会降低效率
|    └── 功能: 标准浮点运算、整数运算、比较运算
└── tensor_cores: // 第四代Tensor Core (Ada架构)
     ├── 数量: 待确认
     ├── 支持精度: FP16, BF16, TF32, FP64, FP8, INT8, INT4
     ├── 约束: 需要特定的矩阵尺寸对齐
     └── 功能: 矩阵乘法累加(GEMM)操作

## 搬移Pipeline
- **Grid-Block-Thread层次**: Grid划分为Block，Block划分为Thread
- **内存合并访问**: 连续线程访问连续内存地址效率最高
- **同步约束**: Block内可同步，Block间无法同步

## 执行逻辑
全部warp并行执行，同一warp内线程SIMT同步执行。线程间同步通过__syncthreads()实现，用户需要显式管理线程块内的同步。

前置需求：
请你思考，并理解上面的GPU系统，包括存储结构、线程层次、内存访问模式等。

