# 精度转换与混合精度指南

Cast API 使用规范和混合精度计算模式。

---

## 目录

1. [Cast RoundMode 选择](#cast-roundmode-选择)
2. [混合精度计算模式（FP16 输入）](#混合精度计算模式fp16-输入)
3. [MX 块量化格式精度路径](#mx-块量化格式精度路径mxfp8--mxfp4-等)

---

## Cast RoundMode 选择

### 选择规则

| 转换方向 | RoundMode | 原因 |
|---------|-----------|------|
| **half → float** | `CAST_NONE` | 低精度→高精度，无精度损失 |
| **float → half** | `CAST_ROUND` | 高精度→低精度，有精度损失 |
| half → int32_t | `CAST_ROUND` / `CAST_CEIL` | 量化场景，根据需求选择 |
| int32_t → float | `CAST_NONE` | 整数→浮点，无精度损失 |

### 正确用法

```cpp
// ✅ half → float：低精度到高精度
AscendC::LocalTensor<float> xFloat = workBuf.Get<float>();
AscendC::Cast<float, half>(xFloat, xHalf, AscendC::RoundMode::CAST_NONE, count);

// ✅ float → half：高精度到低精度
AscendC::LocalTensor<half> yHalf = outQueue.AllocTensor<half>();
AscendC::Cast<half, float>(yHalf, xFloat, AscendC::RoundMode::CAST_ROUND, count);
```

---

## 混合精度计算模式（FP16 输入）

### 适用场景

当输入输出为 FP16，但需要 FP32 精度进行中间计算时（如 Softmax、LayerNorm）。

### 计算流程

```
half 输入 → Cast(FP32) → 中间计算(FP32) → Cast(half) → half 输出
```

### 为什么需要 FP32 中间计算？

1. **ReduceMax/Exp/ReduceSum** 在 FP32 上精度更稳定
2. **避免 FP16 数值溢出**：Exp 结果可能超出 FP16 表示范围
3. **累积误差控制**：多次运算的累积误差在 FP32 下更小

### 加减法场景示例

半精度加减法默认升 FP32；仅当 spec 明确"输入同量级"（如 mask 叠加、已归一化概率相加）时才允许直接 `Add/Sub<half>`。BF16 与 FP16 适用同一规则，仅临界比值不同（BF16=128，FP16=1024）。

> 完整示例、决策表与 Kernel 集成要点见 [api-arithmetic.md → 场景3](api-arithmetic.md#场景3半精度加减法精度优化)。

---

## MX 块量化格式精度路径（mxfp8 / mxfp4 等）

### 适用场景

输入或输出走 MX 类块量化格式（mxfp8 / mxfp4 / mxfp6 等）：每 32 元素一组共享一个 E8M0 scale，数据本体走低精度 dtype（如 fp8_e4m3 / fp8_e5m2 / fp4）。

### 数据通路概览

```
高精度 fp32 张量 → 沿量化轴每 32 元素一组计算 amax
                ↓
            E8M0 scale 生成
                ↓
            Cast<量化 dtype, fp32>(x / scale) → 低精度数据 + 配套 scale
```

### E8M0 scale 编码规则

E8M0 scale 编码 **必须使用 ceil 偏移**：

```cpp
e8m0_byte = (biased_exp_amax - emax_quant_dtype) + 1;  // ✅ ceil
```

参数：
- `biased_exp_amax`：amax 的 fp32 biased exponent（0-255）
- `emax_quant_dtype`：目标量化 dtype 的 max exponent（fp8_e4m3=8、fp8_e5m2=15、fp4_e2m1=2）
- `+1`：ceil 偏移，确保 `amax / decoded_scale ≤ quant_dtype_max`

### 反模式：floor 偏移导致 NaN

```cpp
❌ e8m0_byte = biased_exp_amax - emax_quant_dtype;  // 缺失 +1，落入 floor 区间
```

floor 偏移下 `amax / decoded_scale` 可能落在 `[quant_dtype_max, 2 × quant_dtype_max)` 区间（对 e4m3 即 `[448, 896)`）。Cast<量化 dtype, fp32, RINT> 对超过 dtype_max 的输入会产出 NaN（对 fp8_e4m3 为 `0x7F`）。

### Cast<fp8_e8m0_t> 的 API 路径

MX 类格式的 scale dtype 是 `fp8_e8m0_t`，需要把 fp32 的 biased exponent 部分编码为 e8m0：

| 路径 | 说明 |
|------|------|
| **MemBase Cast API** | 不提供 `fp8_e8m0_t` 重载 |
| **Reg-API Cast** | 提供 `bfloat16_t ↔ fp8_e8m0_t` 重载（查 `Cast-45.md` 等 Reg-API 变体），但需引入 Regbase 编程范式 |
| **手写位运算** | 通过 `ReinterpretCast<uint32_t>(fp32_tensor)` 拿到位表示，`ShiftRight + And + Sub + Add` 提取 biased exp 并应用 ceil 偏移 |

主路径选择按算子整体的编程范式决定：纯 MemBase 路径用手写位运算，Reg-API 路径可直接用 Cast。

### 验证清单

新增 MX 量化路径必须验证：
- [ ] amax 沿正确的轴归约
- [ ] E8M0 scale 公式用 ceil 偏移
- [ ] Cast 之后输出无 NaN（有 NaN → 多半是 floor 偏移）
- [ ] `amax / decoded_scale` 始终 ≤ 量化 dtype max

### 编译错诊断速查

| 编译错 | 真根因 | 修复 |
|--------|-------|------|
| `Mmad` does not accept fp8 ptr | Mmad fm/filter 模板参数同型约束未满足 | 显式指定 `Mmad<fp32, fp8_e4m3_t, fp8_e4m3_t>` 等同型组合 |
| `Cast<fp8_e8m0_t, fp32>` unresolved | MemBase Cast 不支持 e8m0 直接 cast | 用手写位运算 或 切换 Reg-API 路径 |

> MX 格式的 scale 轴选择（沿哪个张量维度量化）属于设计层决策，涉及与 matmul reduction 轴的对齐 —— 仅 Cube/matmul 类算子相关，本仓库（vector / reduction）不展开。
