# 访存 Bound 优化策略

## 判定条件

- MTE2 利用率高（AIC 带宽接近理论峰值）
- Vector/Cube 单元存在等待数据的空闲气泡
- 计算密度低（Ops/Byte 低于硬件能力）

## 仿真图分析要点

- 查看 MTE2 搬移时间线与计算时间线的重叠度
- 识别大数据量搬移窗口，标记可合并或优化的小搬移序列

---

## 策略 1：带宽未打满 → L2 复用

| 操作 | 说明 |
|------|------|
| 检查当前带宽利用率 | 对比 `实际带宽 / 理论峰值带宽` |
| L2 cache 驻留优化 | 调整 tile 大小使数据能驻留 L2，减少重复读 DDR |
| 多核 L2 共享 | 相邻核处理相邻数据，共享 L2 缓存行 |

## 策略 2：指令效率 / 选基本块

| 操作 | 说明 |
|------|------|
| 分析 MTE2 指令发射效率 | 检查是否有冗余搬移指令 |
| 选取基本块 | 优先使用连续地址 + 大粒度搬移指令 |
| 减少 stride 搬移 | stride 搬移效率低于连续搬移，重构数据布局 |

## 策略 3：减少小块搬运 / 合并载入

| 操作 | 说明 |
|------|------|
| 识别小块搬运 | profiling 中识别小于 threshold 的搬移操作 |
| 合并连续小块 | 将多个相邻小搬移合并为单次大搬移 |
| DataCopyPad 参数优化 | 调整对齐参数避免碎片化搬移 |
| 消除不必要搬移 | 检查 UB 内数据是否可以原地消费，减少搬入搬出 |

### 连续块优先

gather、scatter、resize、broadcast 类算子经常因为每元素标量读写变成访存 bound。只要索引在某个局部窗口内形成连续段，就优先改成连续块搬运：

```cpp
// 差：每个元素触发一次 GM 标量访问。
for (int32_t i = 0; i < len; ++i) {
  float v = xGm.GetValue(srcBase + i);
  yGm.SetValue(dstBase + i, v);
}

// 好：连续段搬到 UB，再一次写回。
DataCopy(tileLocal, xGm[srcBase], len);
DataCopy(yGm[dstBase], tileLocal, len);
```

若只有 tail 非对齐，主路径仍应使用普通 `DataCopy`，tail 单独用 `DataCopyPad`：

```cpp
int32_t aligned = len / elemsPer32B * elemsPer32B;
if (aligned > 0) {
  DataCopy(local, gm[offset], aligned);
}
if (aligned < len) {
  DataCopyPad(local[aligned], gm[offset + aligned], len - aligned, padParams);
}
```

### UB 内复用

当同一输入或系数会被多个阶段消费时，优先留在 UB 中，不要写回 GM 再读入：

```cpp
DataCopy(xLocal, xGm[rowBase], D);
DataCopy(xCache, xLocal, D);        // pass 2 复用
ReduceSum(sumLocal, Square(xLocal), tmp, D);

float inv = Rsqrt(sumLocal.GetValue(0) / D + eps);
Muls(xLocal, xCache, inv, D);
Mul(outLocal, xLocal, gammaLocal, D);
DataCopy(yGm[rowBase], outLocal, D);
```

这类复用最适合 normalization、rotary、softmax、multi-pass reduction；如果 D 超过 UB，改成分 tile cache 或只缓存小系数。

## Tiling 修正建议

- 调整 UB tile 大小以提升 L2 命中率
- 优化数据搬移粒度与对齐参数
- 调整多核切分以减少单核数据量
- 对多输出或小结果场景，先在 UB 累积多行/多块结果，再一次 CopyOut，避免每行一次小写回。
