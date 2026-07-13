---
name: ascendc-api-best-practices
description: Ascend C API 使用最佳实践和黑名单。涵盖算术 / 归约 / 数据搬运 / Buffer / 精度转换 / 流水线同步 / Transpose 等核心 API 的正确用法、参数限制和避坑指南。触发：写算子时遇到 API 用法不确定、API 参数报错（对齐 / repeatTimes）、需要查询黑名单和限制时。
---

# Ascend C API 最佳实践

## API 类别索引

| API 类别 | 涵盖 API | 文档 | 典型场景 |
|---|---|---|---|
| **算术** | Add, Sub, Mul, Div, Adds, Muls | [api-arithmetic.md](references/api-arithmetic.md) | Softmax, LayerNorm, 广播 |
| **归约** | ReduceMax, ReduceSum, WholeReduce*, BlockReduce* | [api-reduce.md](references/api-reduce.md), [api-reduce-pattern.md](references/api-reduce-pattern.md) | Softmax, LayerNorm, ReduceMean |
| **数据搬运** | DataCopy, DataCopyPad | [api-datacopy.md](references/api-datacopy.md) | 非对齐处理、多维搬运 |
| **Transpose / Gather** | TransDataTo5HD, Gather | [api-transpose.md](references/api-transpose.md) | 小通道 transpose、gather/permute |
| **Buffer 管理** | TBuf, TQue | [api-buffer.md](references/api-buffer.md) | DoubleBuffer、UB 规划 |
| **精度转换** | Cast | [api-precision.md](references/api-precision.md) | FP16/BF16/FP32 混合精度 |
| **流水线同步** | EnQue, DeQue, SetFlag, WaitFlag | [api-pipeline.md](references/api-pipeline.md) | 多级流水线、事件同步 |
| **repeatTimes 限制** | 所有 vector intrinsic | [api-repeat-limits.md](references/api-repeat-limits.md) | repeatTimes ≤ 255，分批处理 |
| **API 限制 / 对齐** | Compare 等 | [api-restrictions.md](references/api-restrictions.md) | 256B 对齐约束、禁用 API |
| **Host Runtime** | aclrtSetDevice, aclrtGetDeviceInfo | [api-host-runtime.md](references/api-host-runtime.md) | 设备初始化、核数查询 |
| **快速参考** | 所有上述 API 的参数速查表 | [api-quickref.md](references/api-quickref.md) | — |

## 典型场景索引

| 场景 | 关联文档 | 关键技巧 |
|---|---|---|
| **Softmax / LayerNorm** | api-reduce, api-reduce-pattern, api-arithmetic | Reduce 结果广播回 vector，Adds/Muls 替 Div |
| **逐行处理（AR 模板）** | api-arithmetic | Adds/Muls、UB 节省 |
| **多行广播（ARA 模板）** | api-arithmetic | `BinaryRepeatParams.src1RepStride=0`、分批 |
| **半精度加减（FP16/BF16）** | api-arithmetic, api-precision | 默认升精度 FP32，除非 spec 说同量级 |
| **非对齐数据** | api-datacopy | DataCopyPad，避免 DataCopy 撞 32B |
| **混合精度** | api-precision | FP16 输入 + FP32 计算 |
| **流水线优化** | api-pipeline, api-buffer | DoubleBuffer + EnQue/DeQue 配对 |

## ⛔ API 黑名单（绝对禁止）

| API | 禁止原因 | 替代方案 |
|---|---|---|
| `GlobalTensor::SetValue()` | 效率极低（单值 DMA） | `DataCopyPad` |
| `GlobalTensor::GetValue()` | 同上 | `DataCopyPad` |
| `DataCopy(GM↔UB)` 非 32B 对齐 | 越界访问导致 UB OOB | `DataCopyPad` |

调试时可用：`AscendC::printf` 单点验证（仅调试构建，生产移除）。

## 性能反模式速查

这些写法通常能跑通，但在 profiling 中会表现为 Scalar bound、访存碎片或 Vector 指令膨胀：

| 反模式 | 典型表现 | 优先替换 |
|---|---|---|
| GM 每元素 `GetValue/SetValue` | gather/scatter/resize 极慢 | UB staging + 连续 `DataCopy` |
| 每行一个小 `CopyOut` | 小规约、多输出算子 launch 后仍慢 | UB 中攒多行结果后批量写回 |
| fp32 输入仍做 `Adds(x, 0)` | VEC 中 identity add/cast 占比高 | 直接把输入 LocalTensor 作为计算源 |
| fp16/bf16 简单算子强制往返 fp32 | Cast 占主要耗时 | 精度允许时加 native dtype 路径 |
| 每 tile 重复 `div/mod` | Scalar 指令密集 | host tiling 预计算 stride/base/mode |
| 无依赖也成对 `SetFlag/WaitFlag` | Vector 与 Scalar 等待多 | `PipeBarrier` 或删除多余同步 |
| 所有 broadcast 都走 generic index | 小 broadcast 也慢 | same-shape/scalar/last-dim 快路径 |
| DataCopyPad 覆盖主路径 | MTE 指令碎片化 | aligned 主路径用 `DataCopy`，tail 才 pad |

示例：把标量 GM 读写改成连续块搬运。

```cpp
// 差：每个元素一次 GM 标量访问。
for (int32_t i = 0; i < len; ++i) {
    auto v = xGm.GetValue(base + i);
    yGm.SetValue(outBase + i, v);
}

// 好：先搬到 UB，再批量写回。
auto local = buf.Get<T>();
DataCopy(local, xGm[base], len);
DataCopy(yGm[outBase], local, len);
```

示例：aligned 主路径和 tail pad 分离。

```cpp
int32_t mainLen = len / elemsPer32B * elemsPer32B;
if (mainLen > 0) {
    DataCopy(local, gm[offset], mainLen);
}
if (mainLen < len) {
    DataCopyPad(local[mainLen], gm[offset + mainLen], len - mainLen, padParams);
}
```
