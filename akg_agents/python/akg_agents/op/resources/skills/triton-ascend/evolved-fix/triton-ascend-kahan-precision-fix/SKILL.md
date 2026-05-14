---
name: triton-ascend-kahan-precision-fix
description: triton-ascend 大 K 归约精度修复：Kahan 补偿求和替代简单累加，消除 NPU Cube 引擎 FP32 仿真路径与 Triton 顺序累加路径的精度差异
category: fix
version: "1.0.0"
metadata:
  case_type: fix
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3, Atlas A5"
---

# Kahan 补偿求和精度修复

## 触发条件

- matmul / reduction kernel 在大 K 维度（K ≥ 4096）下验证 hard_fail > 0
- mere 正常（< 1e-4）但 mare 偏大（> 1e-2），说明少数点误差极大

## 修复：Kahan 补偿求和

```python
# 错误：简单累加，K 大时误差 O(K × eps)
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for k in range(0, K, BLOCK_K):
    a = tl.load(...)
    b = tl.load(...)
    acc += tl.dot(a, b)

# 修复：Kahan 补偿，误差降为 O(eps)
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
comp = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for k in range(0, K, BLOCK_K):
    a = tl.load(...)
    b = tl.load(...)
    partial = tl.dot(a, b)
    y = partial - comp
    t = acc + y
    comp = (t - acc) - y
    acc = t
```

### 原理

浮点加法不满足结合律：`(a + b) + c ≠ a + (b + c)`。当 `acc` 很大而 `partial` 很小时，`acc + partial` 会丢失 `partial` 的低位精度。Kahan 算法通过一个补偿变量 `comp` 追踪每次加法丢失的精度，下次累加时补回。

逐步拆解：

```
partial = tl.dot(a, b)       # 本次 dot 结果
y = partial - comp           # 减去上次丢失的精度（补偿）
t = acc + y                  # 累加
comp = (t - acc) - y         # 捕获本次丢失的精度
acc = t                      # 更新累加器
```

- `y = partial - comp`：把上次丢失的部分补回来
- `t = acc + y`：执行实际累加
- `comp = (t - acc) - y`：`(t - acc)` 是实际加入 acc 的值，减去 `y` 得到本次丢失的低位
- 代数上 `comp` 恒为零，但浮点运算中它捕获了舍入误差

## 适用范围

| 场景 | 是否适用 |
|------|---------|
| matmul K ≥ 4096 | ✅ |
| reduction（sum/mean）大维度 | ✅ |
| matmul K < 4096 | ❌ 简单累加足够 |
| elementwise / 无归约 | ❌ 无累加误差 |

## Quick Checklist

1. **hard_fail > 0 + mare > 1e-2 + K ≥ 4096** → 加 Kahan 补偿（§修复）
2. **Kahan 后 NPU vs NPU 仍有 hard_fail** → 检查 kernel 逻辑（非累加问题）
