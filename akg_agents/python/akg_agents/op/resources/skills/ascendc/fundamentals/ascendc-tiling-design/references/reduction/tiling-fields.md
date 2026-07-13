# 通用 Tiling 参考

> Tiling 设计原则、tmpBufSize 公式、常用 Tiling 数据结构字段

---

## Tiling 设计原则

**直接公式计算，不用二分查找**：

1. **a0TileBase 是最小对齐单位**：`VECTOR_REG_WIDTH / sizeof(T)`（FP32=64），所有 Buffer 大小是其整数倍
2. **约束取最小**：`a0Inner = min(UB容量限制, A0维度限制, 多核均衡限制)`
3. **保守估算**：用 a0TileBase 计算 `ubPerTileBase`，实际 `tileA0Len ≤ 估算值`，不会超出 UB
4. **API 参数限制传导**：若 Reduce API 的 `repeatTimes ≤ 255`，且传入值与 R 相关，则 `R_max = min(R_max, 255)`

**全载 vs 分载判定**：全载 = 加载数据 + 计算过程全部 Buffer ≤ UB_SIZE。不同算子的中间 Buffer 不同，阈值公式因算子而异。

## tmpBufSize（sharedTmpBuffer）计算

```cpp
uint32_t ComputeReduceBufSize(uint32_t rLengthAlign, uint32_t typeSize) {
    uint32_t perRepeat = 256 / typeSize;  // 64 for FP32
    uint32_t perBlock = 32 / typeSize;    // 8 for FP32
    uint32_t repeats = (rLengthAlign + perRepeat - 1) / perRepeat;
    uint32_t tmpBufSize = ((repeats + perBlock - 1) / perBlock) * perBlock * typeSize;
    return std::max(tmpBufSize, 4096u);   // 最小 4KB
}
```

---

## 常用 Tiling 数据结构字段

## 基础归约 Tiling（ReduceOpTilingData）

| 字段 | 类型 | 含义 |
|------|------|------|
| `factorACntPerCore` | uint64 | 每核 A 轴工作量 |
| `factorATotalCnt` | uint64 | A 轴总工作单元 |
| `ubFactorA` | uint64 | UB 的 A 轴切片大小 |
| `factorRCntPerCore` | uint64 | 每核 R 轴工作量 |
| `factorRTotalCnt` | uint64 | R 轴总工作单元 |
| `ubFactorR` | uint64 | UB 的 R 轴切片大小 |
| `groupR` | uint64 | R 轴分组数（>1 触发 Group Reduce）|
| `outSize` | uint64 | 输出缓冲区大小 |
| `basicBlock` | uint64 | 输入 UB 缓冲区大小 |
| `resultBlock` | uint64 | 输出/中间缓冲区大小 |
| `coreNum` | int32 | 使用核数 |
| `useNddma` | int32 | 是否使用 NDDMA |
| `shape[8]` | uint64[] | 各维度大小 |
| `stride[8]` | int64[] | 各维度步进 |

## ArgMax 系列 Tiling

| 字段 | 类型 | 含义 |
|------|------|------|
| `aSize` | uint64 | 归约轴前所有维度之积 |
| `rSize` | uint64 | 归约维度大小 |
| `nextASize` | uint64 | 归约轴后所有维度之积 |
| `cutASize` | uint16 | UB 的 A 切片 |
| `cutRSize` | uint16 | UB 的 R 切片 |
| `cutNextASize` | uint16 | UB 的 nextA 切片 |
| `realCoreNum` | uint64 | 实际使用核数 |
| `blkFactor` | uint64 | 每核主维度块大小 |
| `blkTailFactor` | uint64 | 尾核主维度块大小 |
| `tilingKey` | uint64 | 策略选择键 |
| `aRaMode` | uint64 | ARA 子模式 (1-6) |
| `workSpaceSize` | uint64 | Group Reduce workspace |

## Norm 类 Tiling（RmsNorm/LayerNorm）

| 字段 | 类型 | 含义 |
|------|------|------|
| `num_row` | uint64 | 输入行数 (M) |
| `num_col` | uint64 | 输入列数 (N) |
| `num_col_align` | uint64 | 对齐后列数 |
| `block_factor` | uint64 | 每核行数 |
| `row_factor` | uint32 | 每次迭代处理行数 |
| `ub_factor` | uint32 | 每次迭代处理列数 |
| `reduce_mask` | uint32 | 归约 mask 配置 |
| `epsilon` | float | 数值稳定常数 |
| `avg_factor` | float | 1.0/num_col |

## Softmax 系列 Tiling

| 字段 | 类型 | 含义 |
|------|------|------|
| `a` (或 `totalA0Len`/`totalA1Len`) | uint64 | A 维大小 |
| `r` (或 `totalRLen`) | uint64 | R 维大小 |
| `rAligned` | uint64 | R 对齐后大小 |
| `ubFactor` | uint64 | UB 处理大小 |
| `aBlockFactor` | uint64 | 每核 A 行数 |
| `tilesPerCore` | uint64 | 每核 tile 数 |
| `rLoopCount` | uint64 | R / VL_FP32 |
