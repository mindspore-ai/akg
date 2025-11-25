# Ascend 910B1 NPU硬件规格说明

## 1. 硬件架构

**芯片配置:**
- 24个AI Core，每个包含: 2个VEC(向量计算单元) + 1个CUBE(矩阵计算单元) + 1个SU(标量计算单元)
- 整个芯片有48个VEC，24个CUBE，24个SU

**数据通路:**
- MTE1: L1 → L0A/L0B
- MTE2: GM → UB/L1/L0A/L0B  
- MTE3: UB → GM, L1 → L2 Cache
- FixP: L0C → L1/GM (可随路类型转换)
- L2 Cache自动缓存GM与AI Core间的数据交互

## 2. 存储系统

| 存储层级 | 容量 | 共享范围 | 对齐 | 说明 |
|---------|------|---------|------|------|
| GM | 64GB | 全设备 | - | 设备主存储 |
| L2 Cache | 192MB | 24个AI Core | - | 自动缓存 |
| L1 Buffer | 1MB | 单AI Core | 256Bytes | Cube通用缓存 |
| L0A | 64KB | 单Cube | 256Bytes | 左矩阵A (m0×k0) |
| L0B | 64KB | 单Cube | 256Bytes | 右矩阵B (k0×n0) |
| L0C | 128KB | 单Cube | 256Bytes | 结果矩阵C (m0×n0)，支持累加 |
| UB | 192KB | 单VEC | 256Bytes | 向量运算缓存 |

## 3. 计算单元

### 3.1 VEC (向量计算单元)

**性能:** 1拍处理 2 × 256 Bytes向量计算 (每AI Core有2个VEC)

**约束:**
- 数据需放入UB (192KB)
- 支持连续内存或带mask访问
- 256 Bytes对齐
- 内部按256 Bytes分块处理

### 3.2 CUBE (矩阵计算单元)

**性能:** 1拍完成 16×16×16 FP16矩阵乘 (8192 FLOPs/cycle)

**工作流程:**
1. L0A ← 左矩阵A (m0×k0)
2. L0B ← 右矩阵B (k0×n0)  
3. CUBE执行矩阵乘: C = A × B
4. 结果写入L0C，支持累加

**约束:**
- `m0 × k0 × sizeof(A.dtype) ≤ 64KB` (L0A)
- `k0 × n0 × sizeof(B.dtype) ≤ 64KB` (L0B)
- `m0 × n0 × sizeof(C.dtype) ≤ 128KB` (L0C)
- 256 Bytes对齐
- 自动按16×16分块，尾块自动补0计算

## 4. 优化策略

**内存对齐:** 所有L0/L1/UB数据传输按照256 Bytes对齐，以发挥最大性能

**数据复用:**
- 调整搬运顺序，让频繁访问数据缓存在L2
- L1/UB双缓冲技术: 一块计算，一块加载
- MTE后台搬运与CUBE/VEC计算重叠

**并行:** 24个AI Core按需并行分配任务，CUBE与VEC可同时执行不同任务
