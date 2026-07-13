# Transpose API 最佳实践

本文档聚焦 small-channel transpose 中常见的 API 组合、硬约束和反模式。

> **当前覆盖范围**：本文档当前仅覆盖**小通道 transpose**；大通道及通用 transpose 场景暂未包含，后续可按需要补充。

***

## 1. 核心计算链路

### 1.1 用 `TransDataTo5HD + Gather` 做小通道 transpose

原理说明：

- step1. TransDataTo5HD 每次一定要输入16行(不满16行时也需要填充有效数据,  比如第0行，否则会出现未知异常)，指令操作将16行\[16, N]转为16列\[N, 16]，一次repeat完成\[16,16]的转置，repeat (N + 15) / 16 次后，得到\[N, 16]
- step2. 当转置前的行数\[比如C]小于16时，需要通过Gather操作从前面TransDataTo5HD得到的\[N, 16]，gather出\[N, C]，offset需要在kernel中提前构造好；

### 1.2 `TransDataTo5HD` 转置输入

```cpp
constexpr uint32_t EPB16 = 16;
uint32_t repeats = tileNA / 16;

LocalTensor<half> srcList[16];
LocalTensor<half> dstList[16];
for (uint32_t i = 0; i < 16; ++i) {
    // 源地址按照输入行大小tileNA偏移，总共需要转置tileNA
    srcList[i] = halfLocal[(i < channelCount) ? (i * tileNA) : 0];
    // 目的地址按照 16个元素做偏移，一次repeat转置[16,16] 需要连续写 (tileNA + 15 ) / 16 个 [16,16]
    dstList[i] = vnLocal[EPB16 * i];
}

// 目的repeatstride 一次repeat转置输出[16,16]，多个repeat需要连续写 因此第一次写0byte，第二次写16*blocksize(32B)=128B，设置16个blocksize(32B)做偏移
uint16_t dstRS = (repeats == 1) ? 0 : 16;
// 源repeatstride 每个repeat按照输入的行方向消耗16个元素, 多个repeat按照行方向连续读, 设置1个blocksize(32B)做偏移
uint16_t srcRS = (repeats == 1) ? 0 : 1;
TransDataTo5HDParams params(false, false, static_cast<uint8_t>(repeats), dstRS, srcRS);
TransDataTo5HD<half>(dstList, srcList, params);
```

**具体样例（C=3, tileNA=32）：理解输入输出数据排布**

> 记输入矩阵元素为 `a[r][c]`，其中 r 为通道索引（0..C-1），c 为列索引（0..tileNA-1）。

**输入 `halfLocal`**（`[C, tileNA]` = `[3, 32]` 按行主序 flat 存储）：

```
halfLocal flat 地址:   0 ...  31 |  32 ...  63 |  64 ...  95
含义(矩阵行列):     Row0[0..31] | Row1[0..31] | Row2[0..31]
元素值:           a[0][0..31] | a[1][0..31] | a[2][0..31]
```

```
Row 0:  [a00, a01, a02, ..., a0_15 | a0_16, a0_17, ..., a0_31]
Row 1:  [a10, a11, a12, ..., a1_15 | a1_16, a1_17, ..., a1_31]
Row 2:  [a20, a21, a22, ..., a2_15 | a2_16, a2_17, ..., a2_31]
```

**srcList / dstList 构建**：

```
srcList[0] = &halfLocal[ 0]  → Row 0
srcList[1] = &halfLocal[32]  → Row 1
srcList[2] = &halfLocal[64]  → Row 2
srcList[3..15] = &halfLocal[0]   ← 填充行（不足16行用 Row0 填充，否则异常）

dstList[i] = &vnLocal[16 * i]    (i = 0..15)
```

**执行过程**（repeats = 32/16 = 2）：

| Repeat | 读取源列范围 | srcList[i] 偏移 | 转置子块 | 写入 vnLocal 范围 |
|--------|------------|----------------|---------|-----------------|
| 0 | 源列 [0..15] | +0 元素 | `[16,16]` | 元素 [0..255] （16×16） |
| 1 | 源列 [16..31] | +16 元素 | `[16,16]` | 元素 [256..511]（16×16） |

**Repeat 0 输出**（dstRS=16, srcRS=1）：

```
vnLocal[  0..15 ]:  [a[0][0],  a[1][0],  a[2][0],  *, *, ..., *]
vnLocal[ 16..31 ]:  [a[0][1],  a[1][1],  a[2][1],  *, *, ..., *]
vnLocal[ 32..47 ]:  [a[0][2],  a[1][2],  a[2][2],  *, *, ..., *]
...
vnLocal[240..255]:  [a[0][15], a[1][15], a[2][15], *, *, ..., *]
```

**Repeat 1 输出**：

```
vnLocal[256..271]:  [a[0][16], a[1][16], a[2][16], *, *, ..., *]
vnLocal[272..287]:  [a[0][17], a[1][17], a[2][17], *, *, ..., *]
...
vnLocal[496..511]:  [a[0][31], a[1][31], a[2][31], *, *, ..., *]
```

**最终 `vnLocal` 数据排布**（等价于 `[tileNA, 16]` 矩阵，按行主序存储）：

```
          col0       col1       col2       col3..15   ← 16列，只前C=3列有效
Row 0 :  a[0][0]    a[1][0]    a[2][0]    ******
Row 1 :  a[0][1]    a[1][1]    a[2][1]    ******
 ...       ...        ...        ...       ******
Row15 :  a[0][15]   a[1][15]   a[2][15]   ******
Row16 :  a[0][16]   a[1][16]   a[2][16]   ******    ← repeat=1 开始
Row17 :  a[0][17]   a[1][17]   a[2][17]   ******
 ...       ...        ...        ...       ******
Row31 :  a[0][31]   a[1][31]   a[2][31]   ******
```

> **规律**：`vnLocal[r][c] = 原输入 a[c][r]`（c < C），即行列互换。`*` 列为填充值，后续 Gather 丢弃。

**Gather 提取有效通道后**，得到最终转置结果 `[tileNA, C]` = `[32, 3]`：

```
Row 0 :  a[0][0]   a[1][0]   a[2][0]
Row 1 :  a[0][1]   a[1][1]   a[2][1]
 ...
Row31 :  a[0][31]  a[1][31]  a[2][31]
```

`TransDataTo5HD` 的输出每个 16-half block 只有前 `channelCount` 个位置有效；剩余位置是 padding。后续必须再用 `Gather` 取出有效值。

### 1.3 `Gather` 提取有效通道

```cpp
auto halfOut = halfLocal;
Gather(halfOut, vnLocal, offsetBuff, 0, validCount);
Cast(outLocal, halfOut, RoundMode::CAST_ROUND, validCount);
```

如果前面已经在 FP32 阶段完成了 in-place round，那么这里的 `Gather` / `Cast` 往往会按对齐后的 count 处理，最终 `half -> uint8` 也可以直接使用 `CAST_NONE`；有效输出范围仍然由当前 tile 的 `curN * channelCount` 决定。

这里的 `offsetBuff` 是 device 端预计算好的 byte offset 表（只需要生成一次），使用Tbuff进行管理，对应生成逻辑：

### 1.4 偏移表offsetBuff生成：Scalar → Vector 指令优化

**问题**：通用实现用 SetValue 逐元素写 offset 表，tileNA × C 次 Scalar 操作。当 tileNA=2048, C=3 时需 6144 次 Scalar 写入，小规模场景下 Scalar 占比可达 90%，成为性能瓶颈。

**关键观察**：偏移表具有周期性结构——每 16 个 p 值为一组，组间差值恒定为 16 × 16 × sizeof(half) = 512 字节：

```
组 0: offset[p*3+0] = (p*16+0)*2,  offset[p*3+1] = (p*16+1)*2,  offset[p*3+2] = (p*16+2)*2   (p=0..15)
组 1: 与组 0 完全相同，仅每个元素 +512
组 2: 与组 0 完全相同，仅每个元素 +1024
...
```

**优化方法**：Scalar 生成基础模式 + Adds 向量指令批量扩展

```
__aicore__ inline void InitOffsetTable()
{
    auto offsetI32 = offsetBuf.Get<int32_t>();
    uint32_t baseCount = 16 * C;
    // Step 1: Scalar SetValue 生成基础模式（仅 16×C 个元素）
    for (uint32_t p = 0; p < 16; ++p) {
        for (uint32_t c = 0; c < C; ++c) {
            offsetI32.SetValue(p * C + c, (p * 16 + c) * sizeof(half));
        }
    }
    // Step 2: Adds 向量指令扩展后续组（每组一次向量操作）
    uint32_t totalGroups = tileNA / 16;
    for (uint32_t g = 1; g < totalGroups; ++g) {
        AscendC::Adds(offsetI32[g * baseCount], offsetI32[0],
                      static_cast<int32_t>(g * 16 * 16 * sizeof(half)), baseCount);
    }
}
```

| 指标            | 优化前（纯 SetValue） | 优化后（Scalar+Adds） |
| ------------- | --------------- | ---------------- |
| Scalar 调用次数   | 6144            | 48               |
| Adds 向量调用次数   | 0               | 127              |
| Scalar ratio  | 90.5%           | 55.1%            |
| VEC ratio     | 11.1%           | 58.7%            |
| Task Duration | 55.6 us         | 15.3 us          |

**适用条件** :

- 偏移表具有等差数列的周期性结构
- baseCount = 16 × C 需满足 Adds 的对齐要求（32 字节，即 baseCount ≥ 8 对 int32）
- C ≤ 16（小通道 transpose 的典型场景）

**通用模式**：任何具有周期性结构的查找表（offset table、index table 等），都可以用「Scalar 生成基础模式 + Adds 向量扩展」的方式优化，将 Scalar 操作从 O(tileNA × C) 降至 O(16 × C)。

***

## 2. API 级硬约束

### 2.1 `Gather` 不直接处理 `uint8`

推荐路线是：

```text
FP32 -> half -> TransDataTo5HD -> Gather(half) -> uint8
```

不要尝试直接在 `uint8` 上做 gather 抽取。

### 2.2 `repeats == 1` 时 stride 必须置 0

```cpp
uint16_t dstRS = (repeats == 1) ? 0 : 16;
uint16_t srcRS = (repeats == 1) ? 0 : 1;
```

这是小 tile 场景的硬约束，不能省。

### 2.3 `VECOUT` depth 必须 >= 2

即便 `Compute` 逻辑看起来是“算完立刻写”，也不要把 `VECOUT` 队列缩成 1。多 tile 下 CopyOut 与后续 Compute 交错时，单槽位容易卡死流水。

### 2.4 GM读写强制使用 `DataCopyPad`，兼容32B对齐非对齐场景

输出是 `curN * channelCount` 字节。只要不是严格 32 字节对齐：

```cpp
DataCopyPad(yGm[gmOffset], outLocal, copyParams);
```

不要为了少写一个 `Pad` 路径，引入额外的尾块分支复杂度。

***

## 3. 反例与反模式

| 反模式                                      | 问题                  | 建议替代                                    |
| ---------------------------------------- | ------------------- | --------------------------------------- |
| `GetValue / SetValue` 逐元素搬运              | 标量 UB 读写，吞吐极差       | `DataCopy / DataCopyPad + vector route` |
| 逐像素 `DataCopyPad(blockLen=channelCount)` | DMA setup 成本远大于有效负载 | 按通道整段搬运，再做 `vnchwconv + Gather`         |
| 默认套用通用 transpose API                     | 小通道场景下内部开销可能远大于实际计算 | 走专门的小通道路径                               |
| 直接 `float -> half -> uint8`              | 容易出现量化 off-by-1     | 先 in-place round，再转 half                |
| 跨 tile 自己管理一次性 event                     | 容易把流水写成一次性同步死锁      | 用 `TQue` 的 `EnQue/DeQue` 管理             |

