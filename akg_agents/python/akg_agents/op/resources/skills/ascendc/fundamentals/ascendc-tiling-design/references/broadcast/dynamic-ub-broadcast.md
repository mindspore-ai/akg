# Broadcast - 动态 UB Broadcast（DAV_3510）

> **适用场景**: 合轴后多维，DAV_3510 芯片。使用动态 Broadcast API（rank 1~9）在 UB 内广播，无 32B 对齐限制。
>
> **DAV_2201** 请使用静态接口（rank=1/2），详见 [ub-broadcast.md](ub-broadcast.md)。

---

## 一、与静态接口的对比

| 维度 | 静态接口 (DAV_2201/DAV_3510) | 动态接口 (DAV_3510) |
|------|-------------------------------|---------------|
| 芯片 | DAV_2201/DAV_3510 通用 | 仅 DAV_3510 |
| rank | 仅 1D/2D | **1~9** |
| axis | 仅 0/1（编译期） | **任意轴**（运行时） |
| 对齐 | dim=2,axis=0 需 srcShape[1] 32B 对齐 | **无对齐限制** |
| tmpBuffer | 需要手动管理或框架申请 | Tiling 内部管理 |
| dtype | int8/uint8/half/float | int8/uint8/int16/uint16/half/bfloat16/int32/uint32/float/int64/uint64 |

---

## 二、API

```cpp
// 1. Kernel 侧计算 Tiling
BroadcastTiling tiling;
GetBroadcastTilingInfo<T>(rank, dstShape, srcShape, false, tiling);

// 2. 执行广播
Broadcast<T>(dstLocal, srcLocal, dstShape, srcShape, &tiling);
```

**参数说明**：

| 参数 | 说明 |
|------|------|
| rank | 维度数，[1, 9] |
| dstShape | 输出 shape，uint32_t 数组，长度 = rank |
| srcShape | 输入 shape，uint32_t 数组，长度 = rank。srcShape[i]=1 且 dstShape[i]>1 时该轴广播 |
| srcInnerPad | 最后一维是否 32B 对齐，当前仅支持 false |
| tiling | `GetBroadcastTilingInfo` 的输出，传给 `Broadcast` |

**示例**：

```cpp
// [2, 1, 4] → [2, 3, 4]（沿 axis=1 广播，rank=3）
uint32_t dstShape[] = {2, 3, 4};
uint32_t srcShape[] = {2, 1, 4};
BroadcastTiling tiling;
GetBroadcastTilingInfo<float>(3, dstShape, srcShape, false, tiling);
Broadcast<float>(dstLocal, srcLocal, dstShape, srcShape, &tiling);

// [1, 8] → [4, 8]（沿 axis=0 广播，rank=2，无 32B 对齐要求）
uint32_t dstShape2[] = {4, 8};
uint32_t srcShape2[] = {1, 8};
BroadcastTiling tiling2;
GetBroadcastTilingInfo<half>(2, dstShape2, srcShape2, false, tiling2);
Broadcast<half>(dstLocal, srcLocal, dstShape2, srcShape2, &tiling2);
```

---

## 三、约束

| 约束 | 说明 |
|------|------|
| **芯片** | 仅 DAV_3510 |
| **rank** | [1, 9] |
| **广播条件** | srcShape[i]=1 且 dstShape[i]>1 |
| **地址重叠** | src 和 dst 不能重叠 |
| **srcInnerPad** | 当前仅支持 false |

---

## 四、数据流

与静态 UB Broadcast 相同，区别仅在 Broadcast API 调用：

```
GM → DataCopyPad → UB [srcShape, 未广播]
  ↓
GetBroadcastTilingInfo + Broadcast → UB [dstShape, 已广播]
  ↓
Compute → UB → DataCopyPad → GM
```

Tiling 参数计算、多核切分、多维索引管理与 [ub-broadcast.md](ub-broadcast.md) 完全相同。
