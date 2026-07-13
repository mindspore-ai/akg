---
name: ascendc-localtensor-subviews
description: "LocalTensor 子视图与索引规则：哪些子视图是合法的、哪些会触发 UB 越界，以及在做多行 batching / per-row 处理时的合法替代写法。"
category: fundamental
version: "1.0.0"
metadata:
  backend: ascend
  dsl: ascendc
  hardware: "Atlas A2, Atlas A3, Atlas A5"
  operator_patterns: "elementwise, reduction, normalize, softmax"
---

# AscendC LocalTensor 子视图与索引规则

在 kernel 内对一个 LocalTensor 取片（subview）非常常见 —— 把 batch 后的 UB 缓冲按行拆开、把双 buffer 拆成前后两半、把 reduce 结果取第一个元素。**绝大多数 UB 越界（VEC instruction error / UB address out of bounds / errno 507035 vector core exception）都来自一种被违反的不变量：子视图的偏移必须在 kernel 编译期可知。**

本 skill 在你打算做"多行 batching"、"双 buffer 切分"、"per-row 处理"、"reduce 结果取第 i 个 scalar"之前必读。

## 1. 一句话规则

**`tensor[const_offset]` 合法；`tensor[runtime_var]` 在大多数 vector intrinsic 入参上会越界。**

```cpp
// ✅ 合法：偏移是编译期常量
LocalTensor<float> half2 = calcBuf.Get<float>()[TILE_LENGTH];  // TILE_LENGTH 是 constexpr
AscendC::Exp(half2, half1, count);

// ❌ 非法：偏移在 kernel 跑起来才算
uint32_t row_off = block_idx * tiling->paddedD;       // runtime
LocalTensor<float> row_view = inBuf.Get<float>()[row_off];   // 编译过得了
AscendC::ReduceMax(maxBuf, row_view, count, ...);              // 运行时 UB OOB
```

LocalTensor 上的 `operator[]` 接受 runtime 表达式（C++ 层面）——**编译能过**，但 vector intrinsic 在解码描述符时需要静态 UB 起始地址，运行时偏移会让硬件抓到非法 UB 地址，触发 errno 507035。

## 2. 真正合法的子视图来源

只有以下来源的偏移在 vector intrinsic 上安全：

| 偏移来源 | 合法 | 说明 |
|---|---|---|
| `constexpr` / `template` 模板常量 | ✅ | `tensor[TILE_LENGTH]`、`tensor[BUFFER_SIZE / 2]` |
| `#define` 宏 | ✅ | `tensor[HALF_TILE]` |
| 循环展开后的常量索引 | ✅ | `#pragma unroll` 展开 |
| `__aicore__ inline` 函数的编译期 const 参数 | ✅ | 模板参数传入 |
| `tiling->fieldName` 从 GMEM 读取的值 | ❌ | 运行时变量 |
| `block_idx`、`GetBlockIdx()` | ❌ | 运行时变量 |
| 循环变量 `for (i = 0; i < N; ++i)` 中的 `i` | ⚠️ | 仅当 N 是 constexpr 且编译器展开循环才安全 |

判断方法：如果偏移依赖任何 GM 读到的字段、kernel 入参、`GetBlockIdx()` 或非展开循环变量，**它就是 runtime 变量**。

## 3. 多行 batching 的反例与正解

软最大化、layernorm、RMSNorm 这类 per-row 操作常想"一次 launch 处理 B 行摊薄开销"。常见错误写法：

```cpp
// ❌ 反例：runtime-offset 子视图导致 UB OOB
__aicore__ inline void Process() {
    auto inLocal = inQueue.DeQue<float>();  // shape = (B, paddedD)
    for (uint32_t r = 0; r < B; ++r) {
        auto row = inLocal[r * paddedD];               // runtime offset
        AscendC::ReduceMax(maxTmp, row, paddedD, ...); // ☠ 运行时 OOB
        // ...
    }
}
```

合法的三种替代：

### 方法 A: per-row 独立 Alloc/Free（最简单，性能略低）

每行单独走一次完整 queue 周期，从根上避免 batch tensor + 子视图：

```cpp
for (uint32_t r = 0; r < rows_this_core; ++r) {
    auto in = inQueue.AllocTensor<T>();
    DataCopy(in, xGm[r * D], paddedD);
    inQueue.EnQue(in);
    auto x = inQueue.DeQue<T>();
    // 现在 x 的起始地址就是 UB 0 偏移，所有 intrinsic 安全
    AscendC::ReduceMax(maxBuf, x, D, ...);
    // ...
    inQueue.FreeTensor(x);
}
```

### 方法 B: 把所有行连成一段连续数据（仅当 reduce 轴是最后一维）

如果你的 reduce 在 last-dim，B 行的数据本来就是 GM 上连续的 `B * D` 个元素，把整段当一个 `(1, B*D)` LocalTensor 处理 —— **依然是一个 reduce 调用**，但用 `mask` / `repeatTimes` 让 intrinsic 自己按 D 一段段处理：

```cpp
// WholeReduceMax 支持 repeat：一次处理 repeatTimes 个长度为 D 的 row
AscendC::WholeReduceMax<float, false>(
    maxOut,          // shape = (B,) 的 LocalTensor
    inLocal,         // shape = (B*D,) 的扁平 LocalTensor
    /*mask=*/D,
    /*repeatTimes=*/B,
    /*dstRepStride=*/1,         // 每次 reduce 结果连续写
    /*srcBlkStride=*/1,
    /*srcRepStride=*/D / 8      // src 每行跳 D/8 个 block (32B)
);
```

这条路是真正能加速 batched softmax / layernorm 的姿势 —— 用 `repeatTimes` + stride 让 intrinsic 自己迭代多行，**不需要写任何子视图**。

### 方法 C: 编译期已知的固定 B（少见但偶尔有用）

只有当 B 在 kernel 编译期常量（template 参数或宏）时，循环展开后的 `r * paddedD` 才可能被编译器折叠成常量。这种写法脆弱，**不推荐作为首选**，但可以作为针对单一 shape 的特化路径。

## 4. UB 内 aliasing 规则

vector intrinsic 的输入和输出操作数**必须在 UB 内不重叠**，否则硬件不会报编译期错误而是产生静默数据 corruption，下一个用到结果的 op 才会以"精度错"或"NaN"形式爆出来。

```cpp
// ❌ aliasing：c 既当 src 又当 dst
AscendC::Adds(c, c, 0.0f, count);  // 等价于 c += 0，硬件不保证语义

// ❌ 重叠：dst 覆盖了 src 后半段
auto a = buf.Get<float>();
auto b = buf.Get<float>()[count - 8];  // 故意重叠 8 elements
AscendC::Mul(b, a, c, count);            // ☠ 数据撞车

// ✅ in-place 必须用同地址同形状的"指定 in-place"intrinsic
//    或者 dst 完全错开 src 的整段
```

特别注意 `calcBuf` 上手动切片的几个 LocalTensor，**只要它们在某次 intrinsic 调用里同时出现，就必须完全不重叠**。常见两个区间设置：

```cpp
// calcBuf 总大小 2 * TILE_LENGTH * sizeof(float)
auto a = calcBuf.Get<float>();                  // [0,                 TILE_LENGTH)
auto b = calcBuf.Get<float>()[TILE_LENGTH];     // [TILE_LENGTH, 2*TILE_LENGTH)
// 任意 count <= TILE_LENGTH 时 a 和 b 都不重叠
```

## 5. 失败模式速诊

| 报错 | 实际原因 | 改法 |
|---|---|---|
| `VEC instruction error: the ub address out of bounds` / errno 507035 | 子视图偏移是 runtime 变量；或 src/dst 在 UB 内 aliasing；或 mask/repeat 越过 UB 边界 | 改成方法 A 或方法 B（见 §3） |
| 编译时 `error: no matching function for call to '...'` | LocalTensor 取片返回类型不匹配 vector intrinsic 的入参签名 | 多半是子视图返回的不是 `LocalTensor<T>` 而是 `LocalTensor<T> &&`，把它先存到具名变量再传 |
| 精度全 NaN / 全 Inf，但 kernel 不报错 | UB aliasing 导致静默 corruption | 在 calcBuf 上用 `[offset]` 切的所有片画一个区间表，确认无重叠 |
| 第一次跑对、改了某行后变错（同 shape）| 拿了一个 runtime 偏移子视图，碰巧某些 shape 下偏移 0 而显得正常 | 用 §3 方法 A 重写 |

## 6. 不要做的事

- 不要为了"batch B 行"而引入 `inLocal[r * stride]` 模式，没有例外。
- 不要用 `*(__ubuf__ T*)((__ubuf__ uint8_t*)inLocal.GetPhyAddr() + offset)` 这种手动指针运算来绕过子视图限制 —— 编译期能过，但破坏 pipe sync 推断，后续 intrinsic 会异常。
- 不要在 `Process()` 调用中给同一个 LocalTensor 反复 reassign 到 `buf[...]` 不同偏移 —— 编译器对每个 LocalTensor 跟踪的 UB 起始地址是绑定式的，重新指向会让 pipeline 调度出错。
