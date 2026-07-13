# 无 Bound 优化策略

## 判定条件

- 各硬件单元利用率均不高
- 无明显单一瓶颈
- 总吞吐量未达预期

## 仿真图分析要点

- 识别流水线气泡与等待区间
- 检查搬移-计算 overlap 度
- 标记 UnitFlag 同步点，检查是否可以进一步减少

---

## 策略 1：开 PingPong

| 操作 | 说明 |
|------|------|
| Double Buffer | 搬入与计算交替进行，隐藏搬移延迟 |
| 双路流水编排 | 使用 Set/Dst 双队列实现搬移-计算 overlap |
| PingPong 粒度调优 | 控制每次搬移量，确保计算时间 ≥ 搬移时间 |

## 策略 2：MMAD 与 Fixp 间 UnitFlag

| 操作 | 说明 |
|------|------|
| UnitFlag 信号 | MMAD 输出 → Fixp 输入通过 UnitFlag 同步，避免同步开销 |
| 减少 Barrier | 以 UnitFlag 替代显式同步指令 |

## 策略 3：减少单次搬运量

| 操作 | 说明 |
|------|------|
| 避免全量搬移 | 仅搬移当前计算所需数据 |
| 渐进式搬移 | 边算边搬，不积压数据 |

## 策略 4：Preload

| 操作 | 说明 |
|------|------|
| 提早搬入下一轮数据 | 在当前计算未完成时启动下一批数据搬移 |
| 预取指令提前发射 | 利用 MTE2 的空闲窗口 |
| 调整 Preload 窗口大小 | 匹配计算耗时，避免搬移完成但计算未完成 |

## 策略 5：指令提早发射

| 操作 | 说明 |
|------|------|
| 搬移指令提前发射 | 不等待当前计算完全结束 |
| 计算指令流水线填充 | 减少流水线气泡 |

## 策略 6：Vec 指令融合减少重复搬运

| 操作 | 说明 |
|------|------|
| 识别重复搬移模式 | 同一数据被多次搬入的场景 |
| 融合 Vec 操作 | 在寄存器内连续处理，消除中间搬出搬入 |
| 一次性搬入多次复用 | 数据驻留 UB 期间完成所有消费 |

## 策略 7：低利用率场景的排查顺序

无明显 bound 往往不是“没有瓶颈”，而是瓶颈被启动开销、同步、分支或过小工作粒度打散。建议按以下顺序排查：

1. **kernel launch / host setup 是否主导**：很多小 tensor、foreach 或 per-row 调用应合并到一次 kernel。
2. **CopyIn/Compute/CopyOut 是否没有重叠**：trace 里三段串行时，优先修流水，而不是盲目增大 tile。
3. **是否有过多小 CopyOut**：多行标量输出应先攒到 UB，再批量写回。
4. **是否每 tile 都做固定分支**：dtype、rank、broadcast mode、特殊值模式应在 host 或 `Init` 中决定。
5. **是否存在未使用 buffer 或死分支**：无 bound 场景中，清理 dead helper、unused TQue、unused include 有时能改善编译调度和寄存器压力。
6. **tile 是否过小**：若每 tile compute 太少，DMA、barrier 和 loop overhead 会吞掉吞吐。

示意：

```cpp
// 差：每行单独处理，整体看不到明显 VEC/MTE bound。
for (int32_t row = 0; row < rows; ++row) {
  CopyIn(row);
  ComputeSmallRow(row);
  CopyOut(row);
}

// 好：多行合批，让每轮有足够工作量。
for (int32_t rb = 0; rb < rows; rb += ROW_BATCH) {
  CopyInRows(rb, ROW_BATCH);
  ComputeRows(rb, ROW_BATCH);
  CopyOutRows(rb, ROW_BATCH);
}
```

## 策略 8：不要把参数微调当成结构优化

无 bound 场景常见陷阱是反复调 `TILE_LENGTH` 或 `blockDim`，但真正问题是结构性的：

| 现象 | 更可能的结构问题 |
|---|---|
| tile 调大调小都只变化 1% | 同步或 host setup 主导 |
| 所有硬件单元都有碎片化短段 | loop 粒度太小或每行单独处理 |
| MTE/VEC 都不满但总耗时高 | 分支、标量索引、queue 往返过多 |
| 小 case 远慢于库实现 | launch 次数、CopyOut 粒度或特殊值快路径缺失 |

## Tiling 修正建议

- 调整 tile 粒度以实现搬移-计算平衡
- 优化 PingPong 分段大小
- 调整 Preload 窗口大小与时间节点
- 若多次调参收益都低于噪声，停止调参，改查合批、同步、索引和死代码。
