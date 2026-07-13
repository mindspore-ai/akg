# VEC Bound 优化策略

VEC bound 是 elementwise、activation、reduction 类算子最常见的瓶颈。Vector 计算单元成为耗时主导，MTE2/MTE3 搬运单元相对空闲。

---

## 判定条件

- Vector 单元利用率高，`aiv_vec_ratio` 是耗时主导
- Vector 指令占据主要执行时间
- MTE2 和 Cube 单元相对空闲

**瓶颈严重程度分级**：

| VEC 占比 | 等级 | 优化方向 |
|----------|------|---------|
| 50–65% | 轻度 | DoubleBuffer + UB 融合有较大收益 |
| 65–80% | 中度 | 减少 Cast 或融合指令 |
| >80% | 深度 | VEC 本身已接近理论极限，优化空间有限 |

---

## 仿真图分析要点

- 识别 Vector 指令密集区
- 检查 Vector 指令间的数据依赖性，寻找可并行执行的机会

**VEC bound trace 特征**：

```
Time: 0 ---------------------------------------- 100ms

SCALAR     |##............................| 5%
SCALARLDST |##............................| 4%
MTE2       |.##....##....##....##.........|15%
VECTOR     |..############################|65%  <- 主导
MTE3       |..........................####.|10%
```

VECTOR 行持续活跃，MTE2 有明显空闲等待，说明搬运速度快于计算速度。

---

## VEC 内部指令分析

从 Chrome Trace JSON 提取 pid=5 (VECTOR) 事件，按 `name` 分类统计：

| 指令类型 | 典型指令 | 含义 |
|---------|---------|------|
| 算术类 | `vec_add`, `vec_mul`, `vec_sub` | 基本计算，低延迟 |
| 超越函数类 | `vec_exp`, `vec_log`, `vec_rec`, `vec_rsqrt` | 高延迟指令，无硬件加速 |
| 类型转换类 | `vcvt_f2f`, `vcvt_f2s`, `vcvt_s2f` | Cast 开销 |
| 归约类 | `vec_reduce_sum`, `vec_reduce_max` | reduction 开销 |

**判断规则**：
- Cast 占比 >20% → 类型转换密集场景，优化 Cast 有高收益
- 超越函数占比 >30% → 深度 VEC bound，优化空间有限

> ascend950 regbase 范式使用 **RVEC** 单元：RVECEX（执行）、RVECLD（加载）、RVECST（存储）、RVECSU（设置）。

---

## 策略 1：UB 融合

多个连续的 Vector 操作直接在 UB 中完成，不将中间结果写回 GM，消除不必要的 MTE2/MTE3 往返。

```
未融合: GM → UB → Compute1 → GM → UB → Compute2 → GM    // 6 次 GM 访问
已融合: GM → UB → Compute1 → Compute2 → UB → GM          // 2 次 GM 访问
```

**检查方法**：在 trace 中观察两个 VECTOR 活跃段之间是否插入了 MTE3（写回）+ MTE2（读入）。如果有，说明中间结果经过了 GM，未融合。

| 操作 | 说明 |
|------|------|
| 识别可融合的相邻 Vector 操作 | 消除中间搬移和暂存 |
| 链式 Vector 操作合并 | Mul+Add → MulAdd, 多个激活函数链式处理 |
| 减少中间结果写回 UB | 融合后在寄存器内完成传递 |

---

## 策略 2：减少类型转换

Cast（类型转换）是 VEC bound 中最常见的隐性开销。典型模式：`fp16 → Cast fp32 → 计算 → Cast fp16`。当计算本身只有 1–2 条指令时，Cast 可能占总 VEC 时间的 30–50%。

| 操作 | 说明 |
|------|------|
| 批量 Cast | 将多次 Cast 合并为一次大粒度操作 |
| 避免不必要的 Cast | 检查计算精度是否必须转换 |
| 选择合适的计算精度 | 全链路 fp16 或全链路 fp32，避免往返转换 |

常见修法：

```cpp
// 差：float 输入也走一遍 Cast/identity copy。
if constexpr (std::is_same_v<T, float>) {
  Adds(xf, xLocal, 0.0f, count);
} else {
  Cast(xf, xLocal, RoundMode::CAST_NONE, count);
}
ComputeFp32(xf, count);

// 好：float 直接作为计算源，非 float 才进 fp32 scratch。
if constexpr (std::is_same_v<T, float>) {
  ComputeFp32(xLocal, count);
} else {
  Cast(xf, xLocal, RoundMode::CAST_NONE, count);
  ComputeFp32(xf, count);
}
```

如果 fp16/bf16 native 误差满足参考要求，可以为半精度单独保留 native 路径：

```cpp
if constexpr (std::is_same_v<T, half>) {
  Sigmoid(yLocal, xLocal, count);  // 避免 Cast 到 fp32 再 Cast 回 half
} else {
  Cast(xf, xLocal, RoundMode::CAST_NONE, count);
  Sigmoid(yf, xf, count);
  Cast(yLocal, yf, RoundMode::CAST_RINT, count);
}
```

---

## 策略 3：融合指令

使用融合指令减少 VEC 指令数：

| 指令 | 等价操作 | 说明 |
|------|---------|------|
| VMULA | VMUL + VADD | 乘加融合 |
| VMULS | VMUL + VSUB | 乘减融合 |
| VMADD | 累加模式 | 单指令累加 |

典型替换：

```cpp
// 差：两个 vector pass。
Muls(tmp, x, scale, count);
Adds(y, tmp, bias, count);

// 好：支持时用融合乘加，或把 bias/scale 折进上一阶段。
Mad(y, x, scaleLocal, biasLocal, count);
```

对于 activation 链，优先把中间 buffer 压到最少：

```cpp
// softplus-like: log(1 + exp(-abs(x))) + max(x, 0)
Abs(tmp0, x, count);
Muls(tmp0, tmp0, -1.0f, count);
Exp(tmp0, tmp0, count);
Adds(tmp0, tmp0, 1.0f, count);
Log(tmp0, tmp0, count);
Max(tmp1, x, zero, count);
Add(y, tmp0, tmp1, count);
```

能原地覆盖的阶段不要另开一个 float buffer；先画 live-range，再决定 calcBuf 段数。

---

## 策略 4：低延迟归约

对于 reduction 操作，优先使用硬件树形归约指令（ReduceSum / ReduceMax / ReduceMin），避免手动 for 循环逐元素归约。

例外：当 reduce dim 很小（例如 2、4、8、16、32）且每行都要做一次 barrier 时，硬件规约同步成本可能超过标量循环。此时可以按 D 分桶：

```cpp
if (D <= 32) {
  float acc = 0.0f;
  for (int32_t i = 0; i < D; ++i) {
    float v = static_cast<float>(xLocal.GetValue(i));
    acc += v;
  }
  outLocal.SetValue(0, acc);
} else {
  ReduceSum(sumLocal, xLocal, tmpBuf, D);
  PipeBarrier<PIPE_V>();
  outLocal.SetValue(0, sumLocal.GetValue(0));
}
```

---

## 策略 5：RegBase 访存

| 操作 | 说明 |
|------|------|
| 数据布局调整为 RegBase 友好 | 连续读取、对齐访问 |
| 使用 RegBase 加载指令 | 减少地址计算开销 |
| 减少 vload/vstore 次数 | RegBase 大粒度搬移优势 |

---

## DoubleBuffer 检查

在 Chrome Trace 中观察 MTE2 和 VECTOR 行的时间重叠：

| 模式 | 特征 | 含义 |
|------|------|------|
| **DB 生效** | MTE2 和 VECTOR 交替出现 | 搬入与计算重叠，流水健康 |
| **DB 未生效** | MTE2 全在前，VECTOR 全在后 | 串行执行，需开启或修复 DoubleBuffer |

---

## Tiling 修正建议

- 调整 UB 布局以支持更高效的 RegBase 访问模式
- 调整 tile 粒度匹配 Vector 融合窗口
- 增大 tile 尺寸减少循环次数，降低流水启停开销（UB 容量：192KB / 910B2，248KB / 950）
- 使能 DoubleBuffer 时，实际可用 UB 需除以 2
