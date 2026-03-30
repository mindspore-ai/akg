# Ascend910_9381 NPU 硬件规格

> SoC_version: Ascend910_9381 | AIC_version: AIC-C-220 | Cube/Vector 分离架构

## 芯片配置

- 24 个 AI Core，每个包含: 2 个 VEC + 1 个 CUBE（Cube/Vector 分离）
- 整个芯片: 48 VEC, 24 CUBE
- Cube 频率: 1800 MHz
- AI CPU: 6 个

## 数据通路

- MTE1: L1 → L0A/L0B
- MTE2: GM → UB/L1/L0A/L0B
- MTE3: UB → GM, L1 → L2 Cache
- FixP: L0C → L1/GM（可随路类型转换，支持 NZ→ND 变换）
- L2 Cache 自动缓存 GM 与 AI Core 间的数据交互（支持 prefetch，prefetch_num=8）

## 存储系统

| 存储层级 | 容量 | 共享范围 | 对齐 | 说明 |
|---------|------|---------|------|------|
| GM | 64GB | 全设备 | - | HBM 设备主存储 |
| L2 Cache | 192MB | 24 个 AI Core | - | 自动缓存，64 页，支持 prefetch |
| L1 Buffer | 512KB | 单 AI Core | 32B | Cube 通用缓存 |
| L0A | 64KB | 单 Cube | 32B | 左矩阵 A (m0×k0) |
| L0B | 64KB | 单 Cube | 32B | 右矩阵 B (k0×n0) |
| L0C | 128KB | 单 Cube | 32B | 结果矩阵 C (m0×n0)，支持累加 |
| UB | 192KB | 单 VEC | 32B | 向量运算缓存（64 bank × 4KB） |

## 计算单元

### VEC（向量计算单元）

- **性能**: 1 拍处理 128 Bytes 向量计算（每 AI Core 有 2 个 VEC）
- **约束**:
  - 数据需放入 UB (192KB)
  - 支持连续内存或带 mask 访问
  - 32 Bytes 对齐（UB block size）
  - 内部按 128 Bytes 分块处理

### CUBE（矩阵计算单元）

- **性能**: 1 拍完成 16×16×16 FP16 矩阵乘（8192 FLOPs/cycle）
- **支持精度**: FP16×FP16→FP16/FP32, FP32×FP32→FP32, HF32×HF32→FP32, INT8×INT8→INT32, INT4/INT2 量化
- **工作流程**:
  1. L0A ← 左矩阵 A (m0×k0)
  2. L0B ← 右矩阵 B (k0×n0)
  3. CUBE 执行矩阵乘: C = A × B
  4. 结果写入 L0C，支持累加
- **约束**:
  - `m0 × k0 × sizeof(A.dtype) ≤ 64KB`（L0A）
  - `k0 × n0 × sizeof(B.dtype) ≤ 64KB`（L0B）
  - `m0 × n0 × sizeof(C.dtype) ≤ 128KB`（L0C）
  - 32 Bytes 对齐
  - 自动按 16×16 分块，尾块自动补 0 计算
  - 支持稀疏计算（sparsity=1）

### 数据类型 MKN 配置

| 数据类型 | M | K | N |
|---------|---|---|---|
| FP16（默认） | 16 | 16 | 16 |
| INT8/UINT8 | 16 | 32 | 16 |
| INT4 | 16 | 64 | 16 |
| INT2/UINT2 | 16 | 128 | 16 |
| UINT1 | 16 | 256 | 16 |

## 内存带宽（GB/s per AI Core）

| 通路 | 读带宽 | 写带宽 |
|------|--------|--------|
| DDR (GM) | 32 | 32 |
| L2 Cache | 110 | 86 |
| L1 → L0A | 512 | - |
| L1 → L0B | 256 | - |
| L1 ↔ UB | 128 | 128 |
| L0C → UB | 256 | - |
| UB → L2/DDR | 64 | 64 |

## 优化策略

- **内存对齐**: 所有数据传输按照 32 Bytes 对齐（UB block size）
- **数据复用**:
  - 调整搬运顺序，让频繁访问数据缓存在 L2
  - L1/UB 双缓冲技术: 一块计算，一块加载
  - MTE 后台搬运与 CUBE/VEC 计算重叠
  - L2 prefetch 可提前加载数据，减少访存延迟
- **并行**: 24 个 AI Core 按需并行分配任务，CUBE 与 VEC 可同时执行不同任务
- **FixPipe**: 支持 L0C→L1 和 L0C→GM 的随路量化/反量化/类型转换/ReLU/NZ→ND 变换
