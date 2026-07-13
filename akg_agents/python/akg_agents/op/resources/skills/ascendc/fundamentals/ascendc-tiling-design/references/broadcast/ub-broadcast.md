# Broadcast - UB Broadcast 静态接口（DAV_2201）

> **适用场景**: 合轴后多维，存在需要广播的输入。搬入原始数据 → UB 内用 `Broadcast()` 静态接口扩展 → 计算。仅支持 rank=1/2、axis=0/1。
>
> **DAV_3510** 建议使用动态接口（rank 1~9，无对齐限制），详见 [dynamic-ub-broadcast.md](dynamic-ub-broadcast.md)。

---

## 一、分支特征

| 特征 | 说明 |
|------|------|
| **合轴后维度** | > 1 维 |
| **广播方式** | 搬入未广播数据，UB 内调用 `Broadcast()` API 扩展 |
| **数据连续性** | 输入连续，但各输入 shape 不同（广播轴 dim=1） |
| **计算结果** | 与输出 tile 等大小的向量 |

---

## 二、广播方式决策

```
⚠️ 对齐判断用原始 shape 维度值 × sizeof(T)，不是 DataCopyPad 搬到 UB 后的对齐值！
   例：srcShape[1]=37, 37×4=148B → 不是 32B 倍数 → 不满足约束
   即使 DataCopyPad 搬入后 UB 上补齐到 160B，仍不满足

广播输入（stride=0 的轴）：
  │
  ├─ axis=-1（(M,1)→(M,N)）？
  │   ├─ 满足 Broadcast 静态接口约束（srcShape[0] × sizeof(T) 是 32B 倍数）？
  │   │   └─ YES → Broadcast 静态接口（§四）
  │   │
  │   └─ NO → M > 2？
  │       ├─ YES → DataCopyPad dummy 填充 + Copy + GatherMask（§3.1）
  │       └─ NO（M≤2） → 逐行 Duplicate 展开
  │
  ├─ axis=-2（(1,N)→(M,N)）？
  │   ├─ 满足 Broadcast 静态接口约束（srcShape[1] × sizeof(T) 是 32B 倍数）？
  │   │   └─ YES → Broadcast 静态接口（§四）
  │   │
  │   └─ NO → Copy 行复制 + GatherMask（§3.2）
  │            无对齐限制，DataCopyPad 搬入后 UB 已 32B 对齐
  │
  └─ 其他 → Broadcast 静态接口（§四）
```

---

## 三、搬运指令广播优化

利用 DataCopyPad + Copy + GatherMask 替代 Broadcast API，省 tmpBuffer，搬运流水完成广播。

### 3.1 axis=-1 广播：DataCopyPad dummy 填充

**适用场景**：最内维 dim=1 广播到 dim=N，如 `(13, 1)` → `(13, 37)`

**原理**：DataCopyPad 在 `blockLen` 不足 32B 对齐时，用源数据块的**首元素值**填充 dummy 数据。`blockLen = sizeof(T)`（单个元素），自动用该元素值填充到 32B（float: 1→8 个）。

**示例**（dtype=float）：

```
Step 1: DataCopyPad 搬入（blockLen=4B, blockCount=13）
  GM [0] → UB: [0,0,0,0,0,0,0,0]   ← 32B，8 个相同 float
  GM [1] → UB: [1,1,1,1,1,1,1,1]
  ...
  GM [12]→ UB: [12,12,...,12]
  得到 UB 上 (13, 8) 的数据

Step 2: Copy 扩展到目标宽度（srcStride=0 重复读同一 8 元素）
  (13, 8) → (13, 40)              ← 40 = ceil(37/8)*8

Step 3: GatherMask 裁剪到有效宽度
  (13, 40) → (13, 37)             ← 去掉尾部 3 个多余元素
```

### 3.2 axis=-2 广播：Copy 行复制

**适用场景**：倒数第二维 dim=1 广播到 dim=M，如 `(1, 37)` → `(13, 37)`

**原理**：使用 Copy 将该行复制到目标行数，再 GatherMask 裁剪。

**示例**（dtype=float）：

```
Step 1: DataCopyPad 搬入 1 行（blockLen=37*4=148B, blockCount=1）
  GM [0..36] → UB: [0,1,...,36, ?,?,?]   ← 160B（32B 对齐），40 个 float

Step 2: Copy 复制到目标行数（srcStride=0 重复读同一行）
  (1, 40) → (13, 40)

Step 3: GatherMask 裁剪每行到有效宽度
  (13, 40) → (13, 37)
```

### 涉及 API

| API | 用途 |
|-----|------|
| `DataCopyPad` | GM→UB，axis=-1 利用 dummy 填充；axis=-2 搬入单行（自动 32B 对齐） |
| `Copy` | UB 内搬运，srcStride=0 重复读实现行/列扩展 |
| `GatherMask` | 按 mask 选取有效元素，裁剪到实际宽度 |

### 优势

- 不调用 Broadcast API，省 tmpBuffer
- 搬运指令完成广播，搬运流水和计算流水可并行

---

## 四、Broadcast 静态接口

DAV_2201/DAV_3510 均可使用，但 DAV_3510 建议优先使用动态接口（见文首链接）。dim 和 axis 为编译期模板参数，仅支持 1D/2D、axis=0 或 1。

```cpp
// dim: tensor 维度（1 或 2）
// axis: 广播维度（0 或 1）
Broadcast<T, dim, axis>(dstLocal, srcLocal, dstShape, srcShape, tmpBuffer);
// 或框架自动申请临时空间版本（无需手动管理 tmpBuffer）
Broadcast<T, dim, axis>(dstLocal, srcLocal, dstShape, srcShape);

// 示例: [M, 1] → [M, K]（沿 axis=1 广播）
uint32_t dstShape[] = {M, K};
uint32_t srcShape[] = {M, 1};
Broadcast<float, 2, 1>(dstLocal, srcLocal, dstShape, srcShape, tmpBuffer);
```

### 约束

| 约束 | 说明 |
|------|------|
| **维度** | 仅 1D 和 2D |
| **axis** | 仅 0 和 1 |
| **dim=2, axis=0** | srcShape[1] 必须 32B 对齐 |
| **dim=2, axis=1** | srcShape[0] 必须 32B 对齐 |
| **地址重叠** | src 和 dst 不能重叠 |
| **dtype（DAV_2201）** | int8_t, uint8_t, half, float |

### tmpBuffer 大小

```cpp
// Host 侧获取
uint32_t maxTmpSize, minTmpSize;
GetBroadCastMaxMinTmpSize(platform, srcShape, dstShape, sizeof(T), false, maxTmpSize, minTmpSize);
// Kernel 侧预留 maxTmpSize 字节
```

---

## 五、Tiling 参数计算

### 5.1 UB 容量计算

```cpp
// bufferNum = 所有存活节点数
// maxDtypeBits = 最大 dtype 位宽
maxElemNum = (ubSize - extraSize) * 8 / (bufferNum * maxDtypeBits);
maxElemNum = floor_align(maxElemNum, 256 * 8 / minDtypeBits);  // 256B repeat 对齐
```

### 5.2 UB 切分

```cpp
// 从最内轴向外累乘输出 dims，找到放不下的轴
uint64_t curProduct = 1;
for (i = shapeLen - 1; i >= 0; i--) {
    curProduct *= outputDims[i];
    if (curProduct > maxElemNum) {
        ubSplitAxis = i;
        curProduct /= outputDims[i];
        break;
    }
}
ubFormer = maxElemNum / curProduct;
ubOuter = ceil(outputDims[ubSplitAxis] / ubFormer);
ubTail = outputDims[ubSplitAxis] - (ubOuter - 1) * ubFormer;
```

### 5.3 多核切分

```cpp
fusedProduct = ubOuter;
for (i = 0; i < ubSplitAxis; i++) {
    fusedProduct *= outputDims[i];
}
blockFormer = ceil(fusedProduct / coreNum);
blockNum = ceil(fusedProduct / blockFormer);
blockTail = fusedProduct - (blockNum - 1) * blockFormer;
```

---

## 六、Kernel 实现要点

### 6.1 数据流

```
对每个 tile (由 ubSplitAxis 切分)：

  普通输入（无广播）:
    GM → DataCopyPad → UB [tileSize]

  广播输入（stride=0 的轴）:
    GM → DataCopyPad → UB [srcShape, 未广播]
      ↓
    Broadcast(dst, src, dstShape, srcShape) → UB [dstShape, 已广播]

  计算:
    Add/Mul/Sub(output, input0, input1, eleNum)

  搬出:
    UB → DataCopyPad → GM
```

### 6.2 多维索引计算

Kernel 需要维护多维索引 `axesIndices[8]`，用于计算每个 tile 的 GM 偏移：

```cpp
// 初始化：将 blockFormer * blockIdx 展开为多维索引
void BroadcastGetAxesIndices(int64_t axesIndices[], int64_t flatIdx,
    const int64_t outputDims[], int64_t ubSplitAxis, int64_t dimProduct)
{
    for (int64_t i = 0; i < ubSplitAxis; i++) {
        dimProduct /= outputDims[i];
        axesIndices[i] = flatIdx / dimProduct;
        flatIdx %= dimProduct;
    }
    axesIndices[ubSplitAxis] = flatIdx;  // ubSplitAxis 内的索引
}

// 每次 ubLoop 后进位
void BroadcastUpdateAxesIndices(int64_t axesIndices[], const int64_t outputDims[],
    int64_t ubSplitAxis, int64_t ubOuter)
{
    axesIndices[ubSplitAxis]++;
    if (axesIndices[ubSplitAxis] >= ubOuter) {
        axesIndices[ubSplitAxis] = 0;
        // 向外进位
        for (int64_t i = ubSplitAxis - 1; i >= 0; i--) {
            axesIndices[i]++;
            if (axesIndices[i] < outputDims[i]) break;
            axesIndices[i] = 0;
        }
    }
}
```

### 6.3 GM 偏移计算

```cpp
// 普通输入的 GM 偏移
int64_t gmOffset = 0;
for (int64_t i = 0; i < ubSplitAxis; i++) {
    gmOffset += axesIndices[i] * inputStrides[i];
}
gmOffset += axesIndices[ubSplitAxis] * ubFormer * inputStrides[ubSplitAxis];

// 广播输入：stride=0 的轴不贡献偏移
// （在合轴阶段已将广播轴 stride 设为 0，上述公式自然跳过）
```

### 6.4 核心执行循环

```cpp
__aicore__ inline void Process()
{
    int64_t ubLoopNum = (GetBlockIdx() == GetBlockNum() - 1)
                        ? blockTail : blockFormer;

    int64_t axesIndices[8] = {0};
    BroadcastGetAxesIndices(axesIndices, blockFormer * GetBlockIdx(),
        outputDims, ubSplitAxis, dimProductBeforeUbInner);

    for (int64_t ubLoopIdx = 0; ubLoopIdx < ubLoopNum; ubLoopIdx++) {
        if (ubLoopIdx != 0) {
            BroadcastUpdateAxesIndices(axesIndices, outputDims, ubSplitAxis, ubOuter);
        }

        int64_t ubSplitSize = (axesIndices[ubSplitAxis] == ubOuter - 1)
                              ? ubTail : ubFormer;

        // 1. 搬入普通输入
        int64_t gmOffset0 = BroadcastGetGmOffset(axesIndices, input0Strides, ...);
        DataCopyPad(input0Local, input0Gm[gmOffset0],
            {1, inputLength0 * sizeof(T), 0, 0});

        // 2. 搬入广播输入（原始 shape）
        int64_t gmOffset1 = BroadcastGetGmOffset(axesIndices, input1Strides, ...);
        DataCopyPad(srcBrcLocal, input1Gm[gmOffset1],
            {1, srcBrcLength * sizeof(T), 0, 0});

        // 3. UB 内广播
        uint32_t dstShape[] = {ubSplitSize, innerDim};
        uint32_t srcShape[] = {1, innerDim};  // 广播轴 dim=1
        Broadcast<T, 2, 0>(dstBrcLocal, srcBrcLocal, dstShape, srcShape, tmpBuffer);

        // 4. 计算
        Add(outputLocal, input0Local, dstBrcLocal, ubSplitSize * innerDim);

        // 5. 搬出
        int64_t outOffset = BroadcastGetGmOffset(axesIndices, outputStrides, ...);
        DataCopyPad(outputGm[outOffset], outputLocal,
            {1, ubSplitSize * innerDim * sizeof(T), 0, 0});
    }
}
```

---

## 七、Buffer 规划

| Buffer | 大小 | 用途 |
|--------|------|------|
| 普通输入 × N₁ | tileSize × sizeof(T) | 无需广播的输入 |
| 广播输入源 × N₂ | srcTileSize × sizeof(T) | 广播前原始数据 |
| 广播输入展开 × N₂ | tileSize × sizeof(T) | 广播后数据 |
| 输出 | tileSize × sizeof(T) | 计算结果 |
| tmpBuffer | GetBroadCastMaxMinTmpSize | Broadcast API 临时空间 |

总 UB ≈ (N₁ + 2×N₂ + 1) × tileSize × maxDtypeBytes + tmpBufferSize

---

## 八、常见问题

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| Broadcast 结果错误 | dstShape/srcShape 传反 | dstShape 是输出 tile shape，srcShape 是输入原始 shape |
| dim=2, axis=0 报错 | srcShape[1] 未 32B 对齐 | 使用动态接口或 pad 到对齐 |
| 精度错误 | ubSplitAxis 处拆分导致广播范围不完整 | 确保广播轴不被 UB 切分拆断 |
| 多维索引越界 | axesIndices 进位逻辑错误 | 检查 UpdateAxesIndices 边界 |
