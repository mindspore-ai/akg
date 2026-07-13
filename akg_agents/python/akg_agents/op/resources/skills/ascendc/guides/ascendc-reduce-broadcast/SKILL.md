---
name: ascendc-reduce-broadcast
description: "Reduce + broadcast 模式（softmax / layernorm / rmsnorm / log-softmax）的标准写法：怎么用 ReduceMax/Sum、WholeReduce、BlockReduce，怎么把 scalar 广播回 vector，怎么避免 NaN 级联和精度炸。"
category: guide
version: "1.0.0"
metadata:
  backend: ascend
  dsl: ascendc
  hardware: "Atlas A2, Atlas A3, Atlas A5"
  operator_patterns: "softmax, layernorm, rmsnorm, instance_norm, log_softmax"
---

# AscendC Reduce + Broadcast 模式

softmax、layernorm、RMSNorm 这一类算子的共同骨架是 **per-row reduce → 用 reduce 结果广播回去做 elementwise**。AscendC 把"reduce"这一步做得很灵活，但很多 fail 都来自下面三个陷阱：

1. **拿错了 reduce intrinsic**（`ReduceMax` 返回的是 LocalTensor 还是 scalar？）
2. **不会把 reduce 后的 scalar 广播回 vector 做减法/除法**
3. **数值稳定性破坏**（exp 不减 max → 上溢；layernorm 不开 fp32 累积 → 精度炸）

本 skill 给出几个公认的稳定 pattern。

## 1. AscendC 提供的 reduce intrinsic 概览

`/Ascend/cann-8.5.0/aarch64-linux/asc/include/basic_api/kernel_operator_vec_reduce_intf.h` 里有这些：

| API | 输入 shape | 输出 shape | 用途 |
|---|---|---|---|
| `BlockReduceMax/Sum` | (N,) | (N/64,) | 每 64 个 element 一组 reduce，输出仍是 LocalTensor |
| `PairReduceSum` | (N,) | (N/2,) | 相邻两两求和 |
| `WholeReduceMax/Sum` | (N,) 配 `repeatTimes` | (repeatTimes,) | **batched row reduce，常用** |
| `ReduceMax/Sum` | (N,) | (1,) 的 LocalTensor | 把整个输入归约成一个 scalar，写到 dst[0] |
| `GetReduceMaxMinCount` | — | scalar | 把上一次 ReduceMax 的标量取到 `T &` 寄存器 |

**判断用哪一个**：

- 单行 reduce → `ReduceMax/Sum`（dst 是 1 元素 LocalTensor）
- N 行 batched 同时 reduce → `WholeReduceMax/Sum` 加 `repeatTimes=N`
- 分阶段 hierarchical（先 64-wise，再 whole）→ `BlockReduceMax` 然后 `ReduceMax`

## 2. 数值稳定 softmax 的标准写法

公式：`softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))`。**不减 max 必上溢**（输入即便只到 89.0，`exp(89) ≈ 4.5e38`，fp32 极限 3.4e38）。

```cpp
template <typename T>
__aicore__ inline void SoftmaxRow(
    const LocalTensor<T>& xIn,     // (D,) 当前行输入
    const LocalTensor<T>& yOut,    // (D,) 当前行输出
    const LocalTensor<float>& fp32Buf,  // 至少 2*D 个 fp32 slot
    uint32_t D)
{
    auto x = fp32Buf;                       // [0,  D)  upcast 后的 x
    auto e = fp32Buf[D];                    // [D, 2D) exp(x - max)

    // 1) upcast 到 fp32（保证精度）
    if constexpr (std::is_same_v<T, float>) {
        AscendC::Adds(x, xIn, 0.0f, D);
    } else {
        AscendC::Cast(x, xIn, AscendC::RoundMode::CAST_NONE, D);
    }

    // 2) row-max → scalar
    AscendC::ReduceMax<float>(/*dst=*/e, /*src=*/x, /*workLocal=*/x, D, false);
    //                          ^^ 借 e[0] 作为 1 元素 dst
    AscendC::PipeBarrier<PIPE_V>();          // 等 reduce 结果落到 e[0]
    float maxVal;
    AscendC::GetReduceMaxMinCount<float>(maxVal);   // 拿 scalar 到 GPR

    // 3) x - max → e
    AscendC::Adds(e, x, -maxVal, D);

    // 4) exp
    AscendC::Exp(e, e, D);

    // 5) row-sum → scalar
    AscendC::ReduceSum<float>(x, e, x, D);   // dst 借 x[0] 用
    AscendC::PipeBarrier<PIPE_V>();
    float sumVal;
    AscendC::GetReduceMaxMinCount<float>(sumVal);

    // 6) 除以 sum
    float invSum = 1.0f / sumVal;
    AscendC::Muls(e, e, invSum, D);

    // 7) downcast 到目标 dtype
    if constexpr (std::is_same_v<T, float>) {
        AscendC::Adds(yOut, e, 0.0f, D);
    } else {
        AscendC::Cast(yOut, e, AscendC::RoundMode::CAST_RINT, D);
    }
}
```

关键不变量：
- **fp32 中间计算**，即使 IO 是 fp16/bf16。fp16 的 `exp` 极易上溢。
- **减 max 之前不能 exp**（即使写注释说"应该没事"）。
- `ReduceMax` 之后必须 `PipeBarrier<PIPE_V>()` 才能让 `GetReduceMaxMinCount` 读到正确 scalar；漏 barrier 会读到旧值或 0，进而 `exp(x - 0)` 溢出 → 全 Inf → softmax NaN（softmax 阶段最常见的"全 NaN"事故）。
- `Muls(invSum)` 比 `Div(sum)` 快 5× 以上，AscendC 的 Div 是仿真出来的。

## 3. LayerNorm 的标准写法

公式：`y = (x - mean) / sqrt(var + eps) * gamma + beta`，其中 `mean = E[x]`、`var = E[x^2] - mean^2`。

```cpp
// fp32Buf 至少需要 3*D 个 float slot
auto x   = fp32Buf;                  // [0,  D)
auto x2  = fp32Buf[D];               // [D, 2D)
auto out = fp32Buf[2*D];             // [2D,3D)

// upcast
AscendC::Cast(x, xIn, ..., D);

// mean: 用 RepeatReduceSum 在累加器里直接 1/D scale
AscendC::Muls(x2, x, 1.0f / D, D);              // x / D
AscendC::ReduceSum<float>(out, x2, x2, D);
AscendC::PipeBarrier<PIPE_V>();
float mean;  AscendC::GetReduceMaxMinCount<float>(mean);

// var = E[(x - mean)^2]
AscendC::Adds(x2, x, -mean, D);                 // x - mean
AscendC::Mul(out, x2, x2, D);                   // (x - mean)^2
AscendC::Muls(out, out, 1.0f / D, D);
AscendC::ReduceSum<float>(out, out, out, D);
AscendC::PipeBarrier<PIPE_V>();
float var;   AscendC::GetReduceMaxMinCount<float>(var);

// normalize: (x - mean) * rsqrt(var + eps)
float invStd = 1.0f / std::sqrt(var + eps);     // scalar，host-style fp32
AscendC::Muls(out, x2, invStd, D);              // x2 still holds (x - mean)

// affine: out = out * gamma + beta（gamma/beta 都是 (D,) LocalTensor）
AscendC::Mul(out, out, gammaLocal, D);
AscendC::Add(out, out, betaLocal, D);
```

**LayerNorm 的两条死法**：
- `var` 用 fp16 算 → 大输入时 (x-mean)^2 上溢得到 Inf
- `rsqrt(var + eps)` 在 var 极小时（接近零方差行）数值放大，输出 nan/inf；eps 不能省，建议 fp32 表示 `1e-5f` 起

## 4. RMSNorm（更简单一些）

公式：`y = x / sqrt(mean(x^2) + eps) * gamma`，没有减 mean 这一步。

```cpp
AscendC::Cast(x, xIn, ..., D);
AscendC::Mul(x2, x, x, D);                       // x^2
AscendC::Muls(x2, x2, 1.0f / D, D);
AscendC::ReduceSum<float>(out, x2, x2, D);
AscendC::PipeBarrier<PIPE_V>();
float meanSq;  AscendC::GetReduceMaxMinCount<float>(meanSq);

float invRms = 1.0f / std::sqrt(meanSq + eps);
AscendC::Muls(out, x, invRms, D);
AscendC::Mul(out, out, gammaLocal, D);
```

## 5. Batched 多行：用 `WholeReduce*` + `repeatTimes`

要在一个 kernel 调用里处理 B 行（B 已知或可控），**别去切子视图**（详见 [[ascendc-localtensor-subviews]]），用 `WholeReduceMax` 的 `repeatTimes` 参数让 intrinsic 自己迭代：

```cpp
// inLocal shape = (B * D,) fp32, 每行 D 个元素连续
LocalTensor<float> maxOut;  // shape >= (B,)

AscendC::WholeReduceMax<float, false>(
    /*dst=*/        maxOut,
    /*src=*/        inLocal,
    /*mask=*/       D,                // 每次 reduce 处理 D 个元素
    /*repeatTimes=*/B,                // 总共 B 行
    /*dstRepStride=*/1,               // 每次结果间隔 1 个 element
    /*srcBlkStride=*/1,
    /*srcRepStride=*/D / 8            // src 跳 D/8 个 32B block
);
```

`D` 必须能被 64 整除（或在 32B 对齐上），否则用 padded D，多余位置用 `Duplicate` 填 `-INFINITY` 后再 reduce（max 路径）或 0（sum 路径）。

## 6. Pass split：两个 kernel vs 单 kernel 单次 launch

softmax 想分两个 kernel（pass-1 算 max 写 GMEM，pass-2 读 max 算 exp/sum/divide）的动机是**减少 UB 压力**。但这条路**几乎不会赢**，原因：

- 中间 max 要走 GMEM → 多一次 DMA + 一次 launch
- pass-1 和 pass-2 之间必须 host-side `aclrtSynchronizeStream` 否则 pass-2 读到旧数据 → 全 NaN
- 单 kernel 用 `PipeBarrier<PIPE_V>` 在 UB 内部就完成同步，零额外 cost

**结论**：除非 D 大到一行都装不下 UB，否则单 kernel 单次 launch 总是更快。

## 7. 失败模式速诊

| 现象 | 诊断 | 改法 |
|---|---|---|
| `Implementation=N/N` 全 NaN | exp 上溢 → softmax 全 0 → 0/0 = NaN。检查是不是漏减 max 或 max scalar 没读到 | §2 的 `PipeBarrier<PIPE_V>` + `GetReduceMaxMinCount` 顺序 |
| 全 Inf | exp 输入未减 max，且 reduce sum 也溢成 Inf | 同上 |
| 大致对、个别行偏大 | reduce 用了 fp16 累加 → catastrophic cancellation | fp32 中间计算 |
| 输出对、但比 ref 慢 5× 以上 | 用了 `Div` 替 `Muls(1/x)`；或两 kernel split | §2 改 Muls；§6 单 kernel |
| `errno 507035` UB OOB | reduce 时输入/输出 LocalTensor 在 UB 内重叠 | calcBuf 切片必须不重叠（详见 [[ascendc-localtensor-subviews]] §4） |

## 8. 不要做的事

- 不要用 fp16/bf16 直接算 mean/var/sum，**永远 upcast 到 fp32 做累加**。
- 不要在 reduce 后直接读 `dst[0].GetValue()` —— `LocalTensor::GetValue` 在 V pipe 上不保证已 commit，必须先 `PipeBarrier<PIPE_V>` + `GetReduceMaxMinCount` 走 scalar 路径。
- 不要为了"少一次 Muls"把 `1/sum` 用 `Div(out, e, sumTensor, D)` 代替 —— AscendC `Div` 极慢。
- 不要省 eps —— 看似输入永远非零，但反传或边界 shape 上会有零方差行。
