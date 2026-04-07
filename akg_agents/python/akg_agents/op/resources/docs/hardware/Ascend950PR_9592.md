# Ascend950PR_9592 NPU 硬件规格

## 1. 硬件架构

**芯片配置**
- 36 个 AI Core，每个包含: 2 个 VEC（向量计算单元） + 1 个 CUBE（矩阵计算单元）
- 整个芯片: 72个 VEC, 36个 CUBE

**数据通路**

- MTE1: L1 → L0A/L0B
- MTE2: GM → UB/L1/L0A/L0B
- MTE3: UB → GM, L1 → L2 Cache
- FixPipe: L0C → L1/GM/UB（可随路类型转换，支持 NZ→ND 变换）
- L2 Cache 自动缓存 GM 与 AI Core 间的数据交互

## 2. 存储系统

| 存储层级 | 容量 | 共享范围 | 对齐 | 说明 |
|---------|------|---------|------|------|
| GM | 96GB | 全设备 | - | 设备主存储 |
| L2 Cache | 128MB | 36 个 AI Core | - | 自动缓存 |
| L1 Buffer | 512KB | 单 AI Core | 256B | Cube 通用缓存 |
| L0A | 64KB | 单 Cube | 256B | 左矩阵 A (m0×k0) |
| L0B | 64KB | 单 Cube | 256B | 右矩阵 B (k0×n0) |
| L0C | 256KB | 单 Cube | 256B | 结果矩阵 C (m0×n0)，支持累加 |
| UB | 248KB | 单 VEC | 256B | 向量运算缓存 |

### 2.1 UB存储结构 （Bank冲突优化）

| 参数 | 值 | 说明 |
|-----|-----|------|
| ub_size | 248KB | UB总容量 |
| ubbank_num | 16 | UB banks数量 |
| ubbank_size | 4096B | 每个bank大小 |
| ubsubbank_num | 64 | 子bank数量 |
| ubsubbank_size | 3968 | 单子访问块大小 |
| ubburst_in_one_block | 32 | 单 block突发长度 |
| ubbank_group_num | 8 | bank组数 |

**Bank冲突避免策略**
- 避免同一bank的连续访问：将数据跨bank交错存储
- ubbank_group_num=8，分组可减少冲突

## 3. 计算单元

### 3.1 VEC (向量计算单元)

**性能:** 1拍处理 2 × 128 Bytes向量计算 (每AI Core有2个VEC)，频率1800MHz

**约束:**
- 数据需放入UB (248KB)
- 支持连续内存或带mask访问
- 256 Bytes对齐
- 内部按256 Bytes分块处理

### 3.2 CUBE (矩阵计算单元)

**性能:** 1拍完成 16×16×16 FP16矩阵乘 (8192 FLOPs/cycle)，频率1800MHz

**工作流程:**
1. L0A ← 左矩阵A (m0×k0)
2. L0B ← 右矩阵B (k0×n0)
3. CUBE执行矩阵乘: C = A × B
4. 结果写入L0C，支持累加

**约束:**
- `m0 × k0 × sizeof(A.dtype) ≤ 64KB` (L0A)
- `k0 × n0 × sizeof(B.dtype) ≤ 64KB` (L0B)
- `m0 × n0 × sizeof(C.dtype) ≤ 256KB` (L0C)
- 256 Bytes对齐
- 自动按16×16分块，尾块自动补0计算

### 3.3 不同数据类型的M/K/N配置

| 数据类型 | M | K | N |
|---------|-----|-----|-----|
| Default (FP16) | 16 | 16 | 16 |
| DT_INT8 | 16 | 32 | 16 |
| DT_UINT8 | 16 | 32 | 16 |
| DT_INT4 | 16 | 64 | 16 |
| DT_INT2 | 16 | 128 | 16 |
| DT_UINT2 | 16 | 128 | 16 |
| DT_UINT1 | 16 | 256 | 16 |
| DT_FLOAT | 16 | 8 | 16 |
| DT_INT32 | 16 | 8 | 16 |

## 4. 内存带宽

| 路径 | 带宽比率 | 说明 |
|-----|---------|------|
| DDR (GM) | 80% | 设备主存，读写各80% |
| L2 Cache | 100% | 自动缓存，读写各100% |
| L1 → L0A | 256% | Cube矩阵加载A |
| L1 → L0B | 256% | Cube矩阵加载B |
| L1 → UB | 128% | 向量计算数据加载 |
| L0C → UB | 128% | 结果搬到UB |
| UB → L2 | 128% | UB写回L2 |
| UB → DDR | 128% | UB写回GM |

## 5. 优化策略

**内存对齐:** 所有L0/L1/UB数据传输按照256 Bytes对齐，以发挥最大性能

**数据复用:**
- 调整搬运顺序，让频繁访问数据缓存在L2
- L1/UB双缓冲技术: 一块计算，一块加载
- MTE后台搬运与CUBE/VEC计算重叠

**并行:** 36个AI Core按需并行分配任务，CUBE与VEC可同时执行不同任务
**FixPipe**: 支持 L0C→L1 、L0C→GM、L0C→UB 的随路量化/反量化/类型转换/ReLU/NZ→ND 变换
