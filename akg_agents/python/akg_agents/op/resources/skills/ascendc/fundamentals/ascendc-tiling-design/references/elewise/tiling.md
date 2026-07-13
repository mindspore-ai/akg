# EleWise - Tiling 详细计算

> **适用场景**: 输入输出 Shape 完全相同，逐元素独立计算

---

## 一、多核切分（blockFormer / blockNum）

### 核心思想

确保每个核处理的数据量 >= 最小阈值，并按 512 字节对齐。

### Tiling 参数说明

| 参数 | 含义 | 计算公式 |
|------|------|---------|
| `dim0` | 元素总数量 | 所有维度相乘 |
| `coreNum` | 实际使用的核数 | `min(计算核数, 最大核数)` |
| `blockFormer` | 每个核的基础元素数（对齐到 512 元素） | `((dim0 / coreNum) + 511) / 512 * 512` |
| `blockNum` | 虚拟 block 数量 | `(dim0 + blockFormer - 1) / blockFormer` |

### 计算步骤

```cpp
// Step 1: 计算核数（保证每个核至少处理 4KB 数据）
// 注意：MIN_TILING_BITS_SIZE_PER_CORE 和 minDtypeBits 单位都是 bits
coreNum = (dim0 * minDtypeBits + MIN_TILING_BITS_SIZE_PER_CORE - 1) / 
           MIN_TILING_BITS_SIZE_PER_CORE;
coreNum = min(coreNum, availableCoreNum);

// Step 2: 每个核的基础元素数，对齐到 512 元素
blockFormer = ((dim0 + coreNum - 1) / coreNum + ELEM_ALIGN_FACTOR - 1) / 
               ELEM_ALIGN_FACTOR * ELEM_ALIGN_FACTOR;

// Step 3: 计算虚拟 block 数量
blockNum = (dim0 + blockFormer - 1) / blockFormer;
```

### 核间数据偏移

```cpp
template <typename DataType>
__aicore__ inline int64_t CalcBlockOffset() {
    return blockFormer * GetBlockIdx() * sizeof(DataType);
}
```

---

## 二、UB 切分（ubFormer / ubLoop / tail）

### 核心思想

确保 UB 处理量是 256B 的整数倍（Vector 指令最优）。

### Tiling 参数说明

| 参数 | 含义 | 计算公式 |
|------|------|---------|
| `ubFormer` | 每个 UB 块的基础大小（256B 对齐） | 对齐到 256B |
| `ubLoopOfFormerBlock` | 首块的 UB 循环次数 | `(blockFormer + ubFormer - 1) / ubFormer` |
| `ubTailOfFormerBlock` | 首块的尾部大小 | `blockFormer - (ubLoopOfFormerBlock - 1) * ubFormer` |
| `ubLoopOfTailBlock` | 尾块的 UB 循环次数 | `(blockTail + ubFormer - 1) / ubFormer` |
| `ubTailOfTailBlock` | 尾块的尾部大小 | `blockTail - (ubLoopOfTailBlock - 1) * ubFormer` |

### 计算步骤

```cpp
// Step 1: 计算 UB 能容纳的最大元素数
bufferDivisor = bufferNum * elemBytes;
maxElemNum = (ubSize - extraSize) * 8 / bufferDivisor;

// Step 2: 按 256B 对齐
alignFactor = REPEAT_BYTES * 8 / minDtypeBits;  // 如 FP32 = 64 元素
ubFormer = (maxElemNum / alignFactor) * alignFactor;

// Step 3: 计算循环次数
ubLoopOfFormerBlock = (blockFormer + ubFormer - 1) / ubFormer;
ubTailOfFormerBlock = blockFormer - (ubLoopOfFormerBlock - 1) * ubFormer;
```

---

## 三、Kernel 执行模型

### Process 循环结构

**核心逻辑**：区分首 block 和尾 block，因为它们的 UB 循环次数可能不同。

```cpp
// 1. 判断当前处理的是否是最后一个 block
bool isLastBlock = (blockIdx == blockNum - 1);

// 2. 获取当前 block 的循环次数和尾部大小
//    首 block 和尾 block 的循环次数/尾部大小可能不同
loopNum = isLastBlock ? ubLoopOfTailBlock : ubLoopOfFormerBlock;
tailNum = isLastBlock ? ubTailOfTailBlock : ubTailOfFormerBlock;

// 3. 主循环（处理完整的 UB 块）
for (uint64_t i = 0; i < loopNum - 1; i++) {
    ProcessTile(offset, ubFormer);
    offset += ubFormer;
}

// 4. 尾部处理（处理最后一个不完整的 UB 块）
ProcessTile(offset, tailNum);
```

**为什么要区分首/尾 block？**

- `blockFormer` 是按 512B 对齐的，可能略大于平均分配
- 最后一个 block 分配的原始数据量 `blockTail` 可能小于 `blockFormer`
- 导致 `ubLoopOfFormerBlock` ≠ `ubLoopOfTailBlock`，`ubTailOfFormerBlock` ≠ `ubTailOfTailBlock`

---

## 四、最小可执行 Tiling 模板

```cpp
struct TilingData {
    int64_t dim0;           // 元素总数量
    int32_t coreNum;        // 实际核数
    int64_t blockFormer;    // 每个核的基础数据量
    int64_t blockNum;       // block 数量
    int64_t ubFormer;       // UB 基础大小
    int64_t ubLoopOfFormerBlock;
    int64_t ubTailOfFormerBlock;
    int64_t ubLoopOfTailBlock;
    int64_t ubTailOfTailBlock;
};

TilingData ComputeTiling(int64_t dim0, int64_t elemBytes, int64_t ubSize, 
                         int64_t bufferNum, int64_t availableCoreNum) {
    TilingData tiling;
    
    // 常量
    constexpr int64_t MIN_TILING_BITS = 32768;       // 4KB，单位 bits
    constexpr int64_t ELEM_ALIGN_FACTOR = 512;       // 多核切分元素对齐因子
    constexpr int64_t ALIGN_256 = 256;               // UB 对齐字节数
    
    // 1. 多核切分（minDtypeBits = elemBytes * 8）
    tiling.coreNum = (dim0 * minDtypeBits + MIN_TILING_BITS - 1) / MIN_TILING_BITS;
    tiling.coreNum = std::min(tiling.coreNum, availableCoreNum);
    
    tiling.blockFormer = ((dim0 + tiling.coreNum - 1) / tiling.coreNum + ELEM_ALIGN_FACTOR - 1) / ELEM_ALIGN_FACTOR * ELEM_ALIGN_FACTOR;
    tiling.blockNum = (dim0 + tiling.blockFormer - 1) / tiling.blockFormer;
    
    // 2. UB 切分
    int64_t bufferDivisor = bufferNum * elemBytes;
    int64_t maxElemNum = (ubSize * 8) / bufferDivisor;
    int64_t alignFactor = ALIGN_256 * 8 / elemBytes;
    tiling.ubFormer = (maxElemNum / alignFactor) * alignFactor;
    
    // 3. 循环次数
    tiling.ubLoopOfFormerBlock = (tiling.blockFormer + tiling.ubFormer - 1) / tiling.ubFormer;
    tiling.ubTailOfFormerBlock = tiling.blockFormer - (tiling.ubLoopOfFormerBlock - 1) * tiling.ubFormer;
    
    int64_t blockTail = dim0 - (tiling.blockNum - 1) * tiling.blockFormer;
    tiling.ubLoopOfTailBlock = (blockTail + tiling.ubFormer - 1) / tiling.ubFormer;
    tiling.ubTailOfTailBlock = blockTail - (tiling.ubLoopOfTailBlock - 1) * tiling.ubFormer;
    
    tiling.dim0 = dim0;
    return tiling;
}
```

---

## 五、经验总结

| 经验 | 说明 | 代码模式 |
|------|------|----------|
| **最小粒度** | 每个核至少 4KB 数据，否则不值得开核 | `MIN_TILING_BITS = 32768` |
| **多核对齐** | 元素数对齐到 512 的倍数 | `blockFormer = (原始值 + 511) / 512 * 512` |
| **UB 对齐** | 按 256B 对齐，确保 Vector 指令效率 | `ubFormer = (原始值 / alignFactor) * alignFactor`，其中 alignFactor = 256 / elemBytes（如 FP32=4字节 则 alignFactor=64） |
| **跨核偏移** | 当前核 GM 偏移 = `blockFormer * blockIdx` | `CalcBlockOffset()` |

---

## 六、dtype 分支：FP16/BF16 升精度（Add/Sub）的 UB 预算

**何时启用**：patterns.md 的 Step 2 判定命中升精度分支时使用。

### 1. Buffer 规划差异

升精度分支在原 dtype Queue 之外，额外引入 **K 份 `ubFormer * sizeof(float)` 的 FP32 中间 Buffer**。K 由 API 层的别名约束决定，tiling 阶段只需将其作为参数代入 UB 预算。

| 项 | 原 dtype 直算 | 升精度分支 |
|---|--------------|-----------|
| 输入/输出 Buffer dtype | 原 dtype（half/bf16） | 原 dtype（不变） |
| FP32 中间 Buffer 份数 | — | **K**（由 API 别名约束给出） |
| 单份 FP32 中间大小 | — | `ubFormer * sizeof(float)` |
| 总 UB 占用 | `bufferNum * ubFormer * elemBytes` | `bufferNum * ubFormer * elemBytes + K * ubFormer * sizeof(float)` |

### 2. ubFormer 计算调整

升精度分支下 `bufferDivisor` 需同时包含半精度 Queue 和 FP32 中间 Buffer 两部分：

```cpp
// 原直算分支
bufferDivisor = bufferNum * elemBytes;

// 升精度分支：bufferNum 份半精度 + K 份 FP32
bufferDivisor = bufferNum * elemBytes + K * sizeof(float);

maxElemNum = (ubSize * 8) / bufferDivisor;
alignFactor = ALIGN_256 * 8 / elemBytes;       // 对齐仍按输入 dtype
ubFormer = (maxElemNum / alignFactor) * alignFactor;
```

K 的取值取决于 Compute 阶段选用 API 是否支持 dst/src 别名，该参数由 API 实现细节给出后回填。

> 升精度分支不新增 TilingData 字段，Kernel 侧按 dtype 模板参数静态选择分支即可。具体 Cast/Add/Sub 调用、RoundMode、别名写法等属 API 实现细节，tiling 阶段不涉及。

---
