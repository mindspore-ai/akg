# 二分累加（Dichotomy Addition / Half-Interval）

**适用场景**: Sum 归约专用，解决顺序累加中大数吃小数的精度问题

**问题**: 顺序累加 `sum = a1 + a2 + a3 + ...` 时，当 sum 已经很大而后续元素很小，小数会因浮点精度被"吃掉"。

**原理**: 用二叉树结构折叠求和，使相近量级的数先相加。

## 核心算法

```cpp
float DichotomyReduceSum(LocalTensor<float>& src, int count) {
    // Step 1: 找到最大的 2^k ≤ count
    int powerTwo = FindNextPower2LessEqual(count);

    // Step 2: 尾部折叠
    int tail = count - powerTwo;
    if (tail > 0) {
        Add(src, src, src[powerTwo], tail);
        // 变体（Half-Interval）：用 mask 保护尾部
        // Add(src, src, src[powerTwo], GenMask(tail));
    }

    // Step 3: 二分折叠
    while (powerTwo > 64) {
        powerTwo /= 2;
        Add(src, src, src[powerTwo], powerTwo);
    }

    // Step 4: WholeReduceSum 硬件指令（≤64 元素）
    WholeReduceSum(result, src, powerTwo);
    return result;
}
```

## 与直接 ReduceSum 的对比

| 方面 | 顺序累加 ReduceSum | 二分累加 |
|------|-------------------|---------|
| 精度 | 大数吃小数 | 相近量级先加，精度更好 |
| 适用操作 | 仅 Sum | 仅 Sum（Max/Min 不受影响） |
| UB 开销 | 无额外 | 原地操作，无额外 buffer |
| 典型场景 | R ≤ VL，量级均匀 | R >> VL，量级差异大 |

## 多行版本（MergeN 模式）

利用 repeat stride 同时对多行做归约（来自 rms_norm / layer_norm 的 `reduce_common.h`）：

```cpp
void ReduceSumMultiN(LocalTensor<float>& src, int numRows,
                     int colsPerRow, int stride) {
    uint64_t rptCfg = BuildRepeatConfig(numRows, stride);
    WholeReduceSum(dst, src, rptCfg);
}
```

