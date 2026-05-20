---
name: triton-ascend-verify-diagnose
description: triton-ascend 验证失败诊断：根据 verifier 的 [precision] 行、hard/outlier 分层阈值、逐维错误分布和样例坐标判断精度、mask、索引、边界、NaN/Inf 等根因
category: fix
version: "1.0.0"
dsl: triton_ascend
metadata:
  case_type: fix
  backend: ascend
  dsl: triton_ascend
---

# Triton Ascend 验证失败诊断指南

只在 verifier 已失败的 debug/fix 阶段使用本 skill。先读 `[precision]` 判断失败类型，再结合逐维位置分布和样例值定位代码区域。逐维分布是辅助线索，不是强制根因；不要因为看到某个维度范围就盲目改 mask 或 tile。

## 1. 先读 `[precision]`

格式：

```text
[precision] dtype=torch.float32 total=256 strict=2 outlier=0/0 hard=254 mere=2.549964e+00 mare=4.284246e+01
```

字段含义：

| 字段 | 含义 | 诊断优先级 |
|------|------|------------|
| `hard` | 超过放宽阈值 `relaxed_tol` 的元素数 | `hard > 0` 时优先按逻辑、索引、mask、store、累加错误处理 |
| `outlier=a/b` | 超过严格阈值但未超过放宽阈值的数量 / 允许上限 | `hard=0` 且 `a>b` 时多为轻微精度或归约误差 |
| `strict` | 通过严格阈值的元素数 | 接近 `total` 表示多数值正确 |
| `mere` | 平均相对误差 | 判断整体偏差，不要单独作为根因 |
| `mare` | 最大相对误差 | 结合样例坐标看极端错误 |

比较公式：

```text
strict_tol  = atol + rtol * abs(ref)
relaxed_tol = outlier_atol + outlier_rtol * abs(ref)
hard_fail   = abs(ref - impl) > relaxed_tol
outlier     = strict_tol < abs(ref - impl) <= relaxed_tol
```

如果 `hard > 0`，先修 hard_fail；不要先调精度阈值。

## 2. 再读逐维错误分布

典型格式：

```text
Error location per dimension ([start:end]=error index range, count/size=coverage):
  dim0: [:]  (2/2 = 100.0%)
  dim1: [:]  (3/3 = 100.0%)
  dim2: [4:5]  (1/5 = 20.0%)
  位置[0, 0, 4]: ref=5.000000e+00 impl=0.000000e+00 abs_diff=5.000000e+00 relaxed_tol=6.200000e-03
```

读取规则：

- `dim0/dim1/...` 是输出 tensor 的原始维度。
- `[start:end]` 是该维出现过错误的索引范围，左闭右开。
- `[:]` 表示该维所有索引都至少出现过错误。
- `count/size` 是该维唯一出错索引数 / 该维大小，不是错误元素个数。
- 每一维都是独立投影，不能把各维 coverage 相乘还原错误数量。
- 单例维可能被省略；例如 `[M, 1]` 的第二维定位价值很低。
- 只有一个非单例维时，逐维分布通常只比样例坐标多一点信息，应优先看 `[precision]` 和样例值。
- 所有非单例维都是 `[:]` 时，优先检查全局公式、累加、dtype、store 或 buffer 覆盖，而不是只修局部 boundary mask。
- 如果日志没有 `Error location per dimension`，不要臆测维度模式，只根据 `[precision]` 和样例值排查。

## 3. 位置模式到排查方向

| 位置模式 | 更可能的根因 | 优先检查 |
|----------|--------------|----------|
| 某一维局部，其他维 `[:]` | 该维边界或索引错误 | 对应维度 offset、stride、tail mask、broadcast |
| 单一连续边界区间，如最后一列/最后一块 | 边界 tile 或 tail 处理错误 | `tl.load`/`tl.store` mask、padding、boundary_check |
| 周期性多个区间 | tile 映射错误 | `program_id` 分解、BLOCK_M/N/K、swizzle |
| 所有非单例维都是 `[:]` 且 `hard` 多 | 全局计算或写回错误 | 公式、转置、acc dtype、store 指针、buffer 覆盖 |
| 只有 `[M]` 或 `[M,1]` | 维度定位信息弱 | 归约轴、累加顺序、样例值、重复写回 |
| `hard=0` 且 `outlier > cap` | 轻微数值误差 | fp32 累加、cast 位置、Kahan、归约顺序 |

## 4. 样例值特征

| 样例值 | 常见原因 |
|--------|----------|
| `impl=0` 且 `ref!=0` | 未计算、store mask 过严、tail 漏写、padding 被误用 |
| `impl` 量级远大于 `ref` | 指针/stride 错、读到其他 tile、acc 未清零、buffer 覆盖 |
| `impl` 与 `ref` 符号相反 | 减法方向、转置、输入顺序错误 |
| 多个位置 `impl` 重复同一值 | 写回覆盖、program_id 映射错、只计算了一个 row/tile |
| 误差很小但超过 strict | 精度/累加顺序问题，不要大改索引逻辑 |

## 5. 快速决策

1. `NaN/Inf` 位置不匹配：先修非法运算、mask 下无效 load、除零或溢出。
2. `hard > 0` 且样例值明显错误：先查逻辑、索引、mask、store。
3. `hard > 0` 但错误覆盖所有非单例维：先查全局公式、累加、dtype 和 store，不要只修边界。
4. `hard=0 && outlier > cap`：按精度问题处理，优先 fp32 accumulator、cast 位置和 Kahan。
5. 低维输出 `[M]` / `[M,1]`：不要过度依赖逐维分布；它通常只是样例索引的补充。
