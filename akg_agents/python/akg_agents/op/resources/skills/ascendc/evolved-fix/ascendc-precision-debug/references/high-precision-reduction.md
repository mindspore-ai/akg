# 高精度归约：FP32 累加器不够用时

> 承接 [common-traps.md](common-traps.md) 陷阱4（Reduce 精度损失）。
> 那里的结论是"用 FP32 累加器"。本文讲**FP32 累加器仍不够**的场景与手段。

## 何时需要本文

FP32 累加器（约 6-7 位有效）在下面三种情况仍会失手：

1. **严格对标 fp64 参考**：评测按 MERE/MARE 对 fp64 golden，小值域/相消位置要求
   NPU 误差 ≤ 2×CPU-fp32 误差（甚至恰好 0 错）。fp32 累加与 CPU 累加**顺序不同**，
   近零点落位不同就挂。
2. **大 K 归约**：K≈1e3~1e4 时，单个 fp32 累加器把这么多项加起来会丢 ~1e-3
   （与输入位宽无关，是累加器本身的舍入）。
3. **灾难性抵消**：大项相消到近零，绝对误差被放大成巨大相对误差。

判断顺序：先确认是不是累加器不够（把 K 调小/换 fp64 参考对比），再上手段。

---

## 手段1：补偿求和（Kahan → Neumaier → guarded）

补偿求和用一个 `comp` 变量追回每步加法丢失的低位，把 fp32 累加拉到 ~fp64。

**别用裸 Kahan**：`t - acc` 在 inf 处产生 `inf - inf = NaN`，会毒化整条链。
用 **Neumaier**（按操作数大小分支），并**加 finite 守卫**——非有限步跳过补偿，
纯加把 inf/nan 按正确语义带过去。

```cpp
// guarded-Neumaier 标量累加：acc + comp 表示 ~fp64 的和
float acc = 0.0f, comp = 0.0f;
for (int i = 0; i < n; ++i) {
    float v = ToFloat(x[i]);          // 逐项
    float t = acc + v;
    // 丢失的低位：按 |acc| 与 |v| 谁大分支
    float lo = (Abs(acc) >= Abs(v)) ? (acc - t) + v : (v - t) + acc;
    if (IsFinite(lo)) comp += lo;     // 守卫：inf/nan 步不补偿，靠纯加带过去
    acc = t;
}
float result = acc + comp;
```

- 只修累加顺序损失，**不修单项的积/casts 误差**（那是手段2）。
- Host 侧 ATen 组合同理：`where(isfinite(lo), lo, 0)` 做守卫，逐元素张量版本。

## 手段2：TwoProduct —— 让积也精确（无需 FMA）

当**积**本身超出 fp32 尾数（24 bit）时，补偿求和救不了，因为丢失发生在乘法：

| 场景 | 积的有效位 | 是否需要 |
|---|---|---|
| fp16 × int8 | ≤19 | 否（fp32 精确） |
| fp16 × fp16 | ≤22 | 否 |
| fp16 × fp32(满) | 可达 ~29 | **是** |
| 大值 fp32 × fp32 | 可达 ~46 | **是** |

Veltkamp 拆分（分裂因子 `4097 = 2^12+1`，fp32 专用，**不依赖 FMA**）把每个积
拆成 `p + e`，`p+e` 等于精确积，`e` 是丢失的低位，喂进 `comp`：

```cpp
// TwoProduct(a,b) -> (p, e)，p+e == a*b 精确
inline void TwoProduct(float a, float b, float& p, float& e) {
    const float S = 4097.0f;          // 2^12 + 1
    p = a * b;
    float ca = a * S, ah = ca - (ca - a), al = a - ah;
    float cb = b * S, bh = cb - (cb - b), bl = b - bh;
    e = ((ah * bh - p) + ah * bl + al * bh) + al * bl;
}
```

## 手段3：小归约 = TwoProduct + guarded-Neumaier → fp64-exact

**小归约**（conv 单点 9~125 tap、RoPE 2 项点积、小 stencil）：每个积用 TwoProduct，
和用 guarded-Neumaier，`comp` 里同时收积的低位 `e` 和加的低位 `lo`：

```cpp
float acc = bias, comp = 0.0f;
for (每个 tap) {
    float p, e; TwoProduct(x_tap, w_tap, p, e);
    float t = acc + p;
    float lo = (Abs(acc) >= Abs(p)) ? (acc - t) + p : (p - t) + acc;
    float delta = lo + e;
    if (IsFinite(delta)) comp += delta;
    acc = t;
}
return acc + comp;                     // ~fp64，近零点也能对齐 golden
```

实测：fp32 depthwise conv 的近零相消 case（大 weight 值域）由此从系统性 outlier
压到 MERE ~1e-14；RoPE fp32 大值相消同法过关。

## 手段4：大归约（matmul / conv-backward，K~1e3-1e4）

积无法全部 materialize（[M,K,N] 内存爆），改**分块 + 块间补偿**：

```
把 K 切成 CHUNK=256 的块；
每块做一次子 matmul（子块内 fp32 累加，项少损失小）；
块间用 Neumaier 补偿合并块的部分和。
```

```cpp
// 伪代码：chunked-K + Neumaier（Host ATen 组合，device 亦可）
acc = matmul(A[:, 0:CH], B[0:CH, :]); comp = zeros_like(acc);
for (k0 = CH; k0 < K; k0 += CH) {
    p = matmul(A[:, k0:k0+CH], B[k0:k0+CH, :]);
    t = acc + p;
    comp += where(abs(acc) >= abs(p), (acc - t) + p, (p - t) + acc);
    acc = t;
}
y = acc + comp;
```

- 块内仍有 fp32 舍入，故这是**逼近**不是精确；能把大 K 的 ~1e-3 压到接近阈值、
  抬高均值，但最深相消点仍可能差一线（见"天花板"）。
- 若积也超 24 bit（如权重反量化 fp32），叠手段2 拆右操作数。

## HF32 陷阱：fp32 matmul/conv 默认不是全精度

Ascend Cube 对 **fp32 matmul 默认走 HF32**——把输入尾数舍到 ~11 位再乘，
比真 fp32 差一个量级。**conv 的 HF32 默认也是开的**。对标 fp64 参考时先关掉：

```python
import torch, torch_npu
torch.npu.matmul.allow_hf32 = False    # 全精度 fp32 matmul
# torch.npu.conv.allow_hf32 = False    # conv 同理（视版本）
```

- 关 HF32 是**精度模式开关**，不是换 kernel，不算绕过反作弊。
- HF32 只影响**输入舍入**，不影响 fp32 累加器本身（累加器对可精确表示的整数结果
  是精确的）——所以关了 HF32 还不够时，问题在累加器（回手段3/4）。

## 天花板：什么时候该停

**深相消 + fp16/bf16 输出舍入**的组合，需要真 **fp64 累加**——910B 硬件没有。
现象：输出是两个大项之差的近零值，fp16/bf16 的 ULP 在该量级极小，要正确舍入
必须把和算到 >24 bit。

已证伪的"以为能过"的路（省得重试）：
- **Ozaki / error-free matmul**（指数对齐切片使块和无舍入）：受控整数测例 error-free，
  但真 case 的对齐部分和逼近 2^24、多 matmul 反而更频繁踩边界，均值不升反降。
- **单纯减小 chunk / 加切片**：非单调，最深点随舍入 knife-edge 抖，不收敛。

结论：小归约能做到 fp64-exact 稳过；大归约能显著改善但最深相消 case 是
**fp32 硬件 + 窄输出舍入的物理天花板**，靠软件补偿逼近但不保证过线。判断出是这类，
就停手，别在 fp32 路上无限试。

## 复用决策树

```
归约精度不达标（对标 fp64/严格 MERE-MARE）
├─ 先确认累加器是不是瓶颈（调小 K / 对 fp64 参考）
├─ matmul/conv 且 fp32 输入？→ 先关 HF32（allow_hf32=False）
├─ 小归约（≤~125 项）？→ TwoProduct + guarded-Neumaier → fp64-exact（能过）
├─ 大归约（K~1e3-1e4）？→ chunked-K + Neumaier(+积拆分) → 逼近，抬均值
└─ 深相消 + 窄输出(fp16/bf16)？→ 需 fp64 累加，硬件无 → 判定天花板，停手
```
