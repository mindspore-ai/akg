# Broadcast - OneDim 分支

> **适用场景**: 合轴后所有维度合为一维，本质是 Elementwise（部分输入可能是标量 dim=1）

---

## 一、分支特征

| 特征 | 说明 |
|------|------|
| **合轴后维度** | 1 维 |
| **广播方式** | 标量输入优先用 TensorScalar 接口（Adds/Muls 等），无对应接口时 Duplicate 展开 |
| **数据连续性** | 所有数据连续，线性处理 |
| **计算结果** | 与输出等长的 1D 向量 |

---

## 二、Buffer 规划

```cpp
// aliveNum = 所有存活节点数（输入 + 输出 + 中间 buffer）
// maxDtypeBytes = 计算图中最大 dtype 字节数

pipe->InitBuffer(buf, ubFormer * maxDtypeBytes * aliveNum);
```

---

## 三、Tiling 参数计算

### 3.1 UB 切分

```cpp
// 先分 UB，开 DB，128B 对齐
int64_t ubFormerByte = (ubSize - extraSize) / aliveNum;
int64_t ubFormer = (ubFormerByte / CACHE_LINE) * CACHE_LINE / maxDtypeBytes;
// CACHE_LINE = 128
```

### 3.2 多核切分

```cpp
int64_t dimLength = outputDims[0];  // 合轴后只有 1 维

int64_t ubOuter = ceil(dimLength / ubFormer);
int64_t ubTail = dimLength % ubFormer;  // 0 → ubFormer
int64_t blockFormer = ceil(ubOuter / coreNum);
int64_t blockTail = ubOuter % blockFormer;  // 0 → blockFormer
int64_t blockNum = ceil(ubOuter / blockFormer);
```

### 3.3 多核优化

当 blockNum 不足 coreNum 一半时，缩小 ubFormer 使核数翻倍：

```cpp
if (blockNum < coreNum / 2 && ubFormer * maxDtypeBytes * aliveNum > 8 * 1024) {
    // 尝试按 coreNum/2 重新分配
    int64_t dimPerCore = dimLength * 2 / coreNum;
    int64_t alignDimPerCore = ceil_align(dimPerCore * maxDtypeBytes, CACHE_LINE) / maxDtypeBytes;
    ubFormer = min(ubFormer, alignDimPerCore);

    // 下限：开 DB 后每核至少 8KB
    int64_t lowestUbFormer = (8 * 1024 / aliveNum / CACHE_LINE) * CACHE_LINE / maxDtypeBytes;
    ubFormer = max(ubFormer, lowestUbFormer);

    // 重新计算分核参数
    ubOuter = ceil(dimLength / ubFormer);
    blockFormer = ceil(ubOuter / coreNum);
    blockNum = ceil(ubOuter / blockFormer);
}
```

---

## 四、Kernel 实现要点

### 4.1 数据流

```
GM → DataCopyPad → UB [ubFormer]
  ↓
  标量输入: 优先用 TensorScalar 接口（Adds/Subs/Muls 等），无需搬入
           若无对应 TensorScalar 接口，才用 Duplicate 展开为向量 + TensorTensor 接口
  非标量输入: DataCopyPad(inputGm, curLen)
  ↓
Compute (Add/Mul/Sub/... 逐元素)
  ↓
UB → DataCopyPad → GM
```

### 4.2 标量输入检测

合轴后某个输入 dim=1 → 该输入是标量。用 scalarFlag 位图标记：

```cpp
// Host 侧
int32_t scalarFlag = 0;
for (int i = 0; i < inputNum; i++) {
    if (dims[i][0] == 1) {
        scalarFlag |= (1 << i);
    }
}
```

### 4.3 核心代码模板

```cpp
__aicore__ inline void Process()
{
    int64_t blockLoopNum = (GetBlockIdx() == blockNum - 1) ? blockTail : blockFormer;
    int64_t offset = GetBlockIdx() * blockFormer * ubFormer;

    for (int64_t i = 0; i < blockLoopNum; i++) {
        int64_t curLen = (i == blockLoopNum - 1 && GetBlockIdx() == blockNum - 1)
                         ? ubTail : ubFormer;

        // CopyIn：非标量输入用 DataCopyPad
        DataCopyPad(input0Local, input0Gm[offset], {1, curLen * sizeof(T), 0, 0});

        // Compute：标量输入优先用 TensorScalar 接口
        if (scalarFlag & (1 << 1)) {
            // 方式1（推荐）：有对应 TensorScalar 接口时直接用
            Adds(outputLocal, input0Local, scalar1, curLen);
            // 方式2（兜底）：无对应 TensorScalar 接口时 Duplicate + TensorTensor
            // Duplicate<T>(input1Local, scalar1, curLen);
            // CustomOp(outputLocal, input0Local, input1Local, curLen);
        } else {
            DataCopyPad(input1Local, input1Gm[offset], {1, curLen * sizeof(T), 0, 0});
            Add(outputLocal, input0Local, input1Local, curLen);
        }

        // CopyOut
        DataCopyPad(outputGm[offset], outputLocal, {1, curLen * sizeof(T), 0, 0});

        offset += ubFormer;
    }
}
```

> **标量处理优先级**：Adds/Subs/Muls/Divs 等 TensorScalar 接口 > Duplicate + TensorTensor 接口。
> TensorScalar 接口省掉 Duplicate 操作和一个 Buffer，性能更优。

---

## 五、常见问题

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| 标量输入结果错误 | scalarFlag 计算错误 | 检查合轴后 dims[i][0] 是否为 1 |
| 核利用率低 | ubFormer 太大导致 blockNum 太少 | 启用多核优化（缩小 ubFormer） |
| 非对齐数据错误 | DataCopy 不支持非对齐 | 使用 DataCopyPad |
