# Welford Online 算法（在线单遍）

**适用场景**: 分载模式下需要流式计算两个相关统计量（如 mean + variance），单遍扫描完成。

**优势**: 单遍扫描（vs TwoPass 两遍）、数值稳定性好、支持分组并行合并。

---

## 核心更新公式

```
初始: mean = 0, M2 = 0, count = 0

对每个新元素 x:
    count += 1
    delta1 = x - mean           ← 旧偏差
    mean = mean + delta1/count  ← 增量更新均值
    delta3 = x - mean           ← 新偏差
    M2 = M2 + delta1 * delta3   ← 增量更新方差

最终: var = M2 / (count - correction)
```

## 两组合并公式

当多个核或多个分组各自计算了局部 (mean, M2, count) 后，合并为全局结果：

```
合并 (mean_a, M2_a, count_a) 和 (mean_b, M2_b, count_b):

count_total = count_a + count_b
delta = mean_b - mean_a
mean_total = mean_a + delta * count_b / count_total
M2_total = M2_a + M2_b + delta² * count_a * count_b / count_total
```

## Group Welford（分组合并）

当分载的 chunk 数很大时，每 8 个 chunk 做一次中间合并，防止浮点误差累积。

## 向量化 Welford 更新（AscendC 实现示例）

```cpp
// 对 UB 中一个 chunk 的数据做 Welford 更新
void WelfordUpdate(LocalTensor<float>& x, int curLen,
                   LocalTensor<float>& mean, LocalTensor<float>& M2,
                   int& count) {
    for (int i = 0; i < curLen; i++) {
        count++;
        float scale = 1.0f / static_cast<float>(count);

        // delta1 = x - mean（向量化：对整个 A 维）
        Sub(delta1Buf, x[i * A_aligned], mean, A_aligned);

        // mean = mean + delta1 * scale
        Muls(tmpBuf, delta1Buf, scale, A_aligned);
        Add(mean, mean, tmpBuf, A_aligned);

        // delta3 = x - mean_new
        Sub(delta3Buf, x[i * A_aligned], mean, A_aligned);

        // M2 = M2 + delta1 * delta3
        Mul(tmpBuf, delta1Buf, delta3Buf, A_aligned);
        Add(M2, M2, tmpBuf, A_aligned);
    }
}
```

## 与 TwoPass 的选择

| 条件 | 推荐 |
|------|------|
| FullLoad + 两次顺序归约 | TwoPass（第一遍求 A，第二遍用 A 求 B，实现简单） |
| 分载 + 两次相关归约 | Welford（单遍流式，省一轮 IO） |
