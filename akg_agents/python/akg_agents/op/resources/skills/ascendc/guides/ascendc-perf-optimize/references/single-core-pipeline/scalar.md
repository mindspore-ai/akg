# Scalar Bound / 小 case 优化策略

## 判定条件

- 总计算量极小（如 element count < 阈值）
- Scalar 指令占比高，Vector/Cube 单元空闲
- 指令发射率低，IPC 偏低

## 仿真图分析要点

- 定位 Scalar 指令占比较大的时间窗口
- 识别 Scalar 与 Vector 间不必要的同步等待

## 优化策略

| 策略 | 操作 | 效果 |
|------|------|------|
| **Scalar 优化** | 减少冗余 scalar 计算，合并条件分支 | 降低 scalar 指令数 |
| **循环展开** | 展开小循环减少分支代价 | 提升 IPC |
| **减少循环轴** | 根据tiling最简化循环轴 | 降低scalar |
| **指令选择** | 使用高效 scalar 指令替代低效序列 | 缩短关键路径 |
| **减少标量-向量转换** | 避免不必要的 Scalar ↔ Vector 数据搬移 | 减少搬移开销 |
| **使用性能友好的API** | 使用set_flag和wait_flag代替Queue，使用LocalTensor代替Tbuffer，去除Tpipe | 减少封装带来的scalar |

## 实战模式

### 1. 将索引算术移出内层

内层出现 `div/mod`、多级 stride 乘加、shape 分支时，通常先把它们变成 batch 级变量或 host tiling 字段：

```cpp
// 差：每个元素重复解析线性下标。
int64_t n = linear / (C * H * W);
int64_t c = (linear / (H * W)) % C;
int64_t h = (linear / W) % H;
int64_t w = linear % W;

// 好：按连续行推进。
int64_t rowBase = ((n * C + c) * H + h) * W;
for (int32_t w = wStart; w < wEnd; ++w) {
  Compute(rowBase + w);
}
```

如果范围足够，内层计数器优先使用 `int32_t`/`uint32_t`，减少 64 位整数指令压力。

### 2. 批量化小结果

argmax、cross entropy、foreach norm 等场景中，每行只输出 1 个或几个元素。不要每行单独 CopyOut：

```cpp
constexpr int32_t BATCH = 32;
auto outLocal = outBuf.Get<int64_t>();

for (int32_t base = 0; base < rows; base += BATCH) {
  int32_t n = Min(BATCH, rows - base);
  for (int32_t i = 0; i < n; ++i) {
    outLocal.SetValue(i, ComputeSmallRow(base + i));
  }
  DataCopy(outGm[base], outLocal, n);
}
```

### 3. 小 D 规约可走标量

当 `D <= 32` 或 `D <= 64` 时，向量规约的同步和临时 buffer 可能比标量循环更贵：

```cpp
if (D <= smallDThreshold) {
  float maxVal = -INFINITY;
  int32_t maxIdx = 0;
  for (int32_t i = 0; i < D; ++i) {
    float v = static_cast<float>(xLocal.GetValue(i));
    if (v > maxVal) {
      maxVal = v;
      maxIdx = i;
    }
  }
  outIdxLocal.SetValue(row, maxIdx);
} else {
  ReduceMax(maxLocal, xLocal, tmpBuf, D);
}
```

### 4. 分支上移

固定模式判断不要放在 tile 内层：

```cpp
// Init 或 Process 开头决定。
bool sameShape = mode_ == MODE_SAME_SHAPE;

if (sameShape) {
  ProcessSameShape();
} else {
  ProcessGeneric();
}
```

不要在 `for tile` 或 `for element` 中反复判断 dtype、rank、broadcast 模式。

## Tiling 修正建议

- 适当增大单次处理粒度，减少循环次数
- 考虑与其他 kernel 融合以减少 launch 开销
- 对小行/小结果场景，优先搜索 `rowsPerTile`、`outputsPerCopy`、`smallDThreshold`，而不是只调 `TILE_LENGTH`。
