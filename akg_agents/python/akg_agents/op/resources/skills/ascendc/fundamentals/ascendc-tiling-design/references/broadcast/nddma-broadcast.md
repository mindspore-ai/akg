# Broadcast - NDDMA Broadcast 分支（DAV_3510）

> **适用场景**: 合轴后多维，DAV_3510 芯片。**仅用于 GM→UB 搬入阶段**，通过 NDDMA 硬件 stride=0 配置自动广播，数据到达 UB 时已是广播后的完整 tile。不适用于 UB 内部广播（UB 内部请使用 [动态 UB Broadcast](dynamic-ub-broadcast.md)）。
>
> **DAV_2201 不支持 NDDMA**，请使用 [UB Broadcast 静态接口](ub-broadcast.md)。

---

## 一、分支特征

| 特征 | 说明 |
|------|------|
| **芯片要求** | DAV_3510（Ascend 950） |
| **合轴后维度** | > 1 维 |
| **广播方式** | GM→UB 搬运时，NDDMA 硬件根据 stride=0 自动复制数据 |
| **与 UB BRC 的区别** | 不需要 UB 内 `Broadcast()` API 调用，搬入即完成广播 |
| **NDDMA 最大维度** | 5 维。超过 5 维需要外层循环 + 多次 NDDMA 调用 |

---

## 二、核心 API：DataCopy + MultiCopyParams

NDDMA 广播的本质：配置多维 strided copy，将广播轴的 srcStride 设为 0，硬件在该轴自动复制。

### 2.1 MultiCopyParams 结构

```cpp
// NDDMA 最大支持 5 维
constexpr int64_t NDDMA_MAX_DIMS = 5;

AscendC::MultiCopyLoopInfo<NDDMA_MAX_DIMS> loopInfo;
// loopInfo.loopSize[i]      — 第 i 维的循环次数
// loopInfo.loopSrcStride[i] — 第 i 维的 GM 源跳跃（stride=0 → 该维自动复制）
// loopInfo.loopDstStride[i] — 第 i 维的 UB 目标跳跃

AscendC::MultiCopyParams<T, NDDMA_MAX_DIMS> params = {loopInfo, constValue};
```

### 2.2 调用方式

```cpp
static constexpr AscendC::MultiCopyConfig config = {false, 0, 0, false};
AscendC::DataCopy<T, NDDMA_MAX_DIMS, config>(localTensor, globalTensor[gmOffset], params);
```

### 2.3 stride=0 的广播效果

```
示例: x=[1,3,8] 广播到 out=[4,3,8]
  inputStrides  = [0, 8, 1]    ← 轴 0 stride=0，需要广播
  outputStrides = [24, 8, 1]

NDDMA 配置:
  loopSize      = [4, 3, 8]
  loopSrcStride = [0, 8, 1]    ← srcStride[0]=0，硬件在轴 0 重复读同一块数据
  loopDstStride = [24, 8, 1]

效果: 硬件自动将 [1,3,8] 的数据复制 4 次填满 [4,3,8]
```

---

## 三、两种模式

### 3.1 WithoutLoop（schMode=1）：UB 切分后剩余轴 ≤ 5

剩余维度在 NDDMA 的 5 维限制内，一次 `DataCopy` 调用完成。

```cpp
// 配置 MultiCopyParams：将 ubSplitAxis 及之后的轴映射到 NDDMA 的 5 维
MultiCopyParams<T, 5> params = BroadcastSetNddmaConfigWithoutLoop<T>(
    outputDims, outputStrides, inputStrides, shapeLen, ubSplitSize, ubSplitAxis);

// 一次搬入完成广播
DataCopy<T, 5, config>(localTensor, globalTensor[gmOffset], params);
```

**优化**：如果某个广播输入的 `inputStrides[ubSplitAxis] == outputStrides[ubSplitAxis]`（即该输入在 UB 切分轴无需广播），退化为普通 `DataCopyPad`，避免 NDDMA 开销。

### 3.2 WithLoop（schMode=2）：UB 切分后剩余轴 > 5

剩余维度超过 NDDMA 的 5 维限制。将最内 5 维交给 NDDMA，外层轴通过 Kernel 循环遍历。

```cpp
// 配置 NDDMA 处理最内 5 维
MultiCopyParams<T, 5> params = BroadcastSetNddmaConfigWithLoop<T>(
    outputDims, outputStrides, inputStrides, shapeLen, ubSplitAxis);

// 外层循环遍历剩余轴
int64_t nddmaProduct = BroadcastFuseAxes(outputDims, ubSplitAxis + 1, shapeLen - 5) * ubSplitSize;
int64_t nddmaIndices[3] = {0};

for (int64_t i = 0; i < nddmaProduct; i++) {
    if (i != 0) {
        BroadcastUpdateNddmaAxesIndices(nddmaIndices, outputDims, ubSplitAxis, ...);
    }
    int64_t nddmaGmOffset = BroadcastGetNddmaOffset(nddmaIndices, inputStrides, ...);
    int64_t nddmaUbOffset = BroadcastGetNddmaOffset(nddmaIndices, outputStrides, ...);

    DataCopy<T, 5, config>(localTensor[nddmaUbOffset],
                           globalTensor[gmOffset + nddmaGmOffset], params);
}
```

### 3.3 FuseAxis 优化（WithLoop + CopyBrcSize ≤ 4）

当 CopyBrc 节点数 ≤ 4 且 ≥ 3 时，尝试将广播模式相同的相邻轴合并，减少 NDDMA 调用次数：

```cpp
// 从最内轴向外扫描，相邻轴广播模式相同（都 stride=0 或都 stride>0）则合并
while (count > ubSplitAxis) {
    curFlag = inputStrides[count] == 0 ? 0 : 1;
    if (curFlag != oriFlag) {
        // 不同模式 → 新维度
        outputDims2[newCount] = outputDims[count];
    } else {
        // 相同模式 → 合并
        outputDims2[newCount] *= outputDims[count];
    }
}
```

---

## 四、Tiling 参数计算

与 UB Broadcast 分支完全相同（共用 `DoBrodcastTiling`），仅 schMode 不同：

```cpp
// 判定
int64_t axisInsideUB = shapeLen - ubSplitAxis;
if (axisInsideUB <= 5) {
    schMode = 1;   // WithoutLoop
} else {
    schMode = 2;   // WithLoop
}
```

---

## 五、Kernel 执行流程

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

        // 1. 广播输入：NDDMA 搬入（stride=0 轴硬件自动复制）
        //    数据到达 UB 时已是广播后的完整 tile
        BroadcastNddmaWithoutLoop/WithLoop(globalTensor, localTensor, ...);

        // 2. 普通输入：DataCopyPad 线性搬入
        DataCopyPad(inputLocal, inputGm[gmOffset], {1, inputLength * sizeof(T), 0, 0});

        // 3. 计算
        Add(outputLocal, input0Local, input1Local, tileLength);

        // 4. 搬出
        DataCopyPad(outputGm[outOffset], outputLocal, {1, tileLength * sizeof(T), 0, 0});
    }
}
```

---

## 六、与 UB Broadcast 的对比

| 维度 | UB Broadcast (DAV_2201) | NDDMA Broadcast (DAV_3510) |
|------|---------------------|----------------------|
| 广播时机 | 搬入后，UB 内调用 Broadcast API | 搬入时，硬件自动完成 |
| UB 占用 | 需要 src + dst 两块空间 | 只需 dst 空间（搬入即是结果） |
| API | `Broadcast<T, dim, axis>()` | `DataCopy<T, 5, config>()` |
| 维度限制 | 静态接口 1D/2D；动态接口 rank 1~9 | NDDMA 最大 5 维，超过需外层循环 |
| 性能 | 额外矢量指令开销 | 硬件完成，无额外指令 |
| tmpBuffer | 需要（Broadcast API 内部使用） | 不需要 |

---

## 七、约束

| 约束 | 说明 |
|------|------|
| **芯片** | 仅 DAV_3510（Ascend 950），DAV_2201 不支持 |
| **NDDMA 最大维度** | 5。超过 5 维需外层循环 |
| **stride=0 含义** | inputStrides 中 stride=0 的轴，硬件重复读取不推进地址 |
| **无广播退化** | inputStrides == outputStrides 时退化为 DataCopyPad |
