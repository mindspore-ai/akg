# Scalar 高性能编码原则

本文给出 9 条编写 Scalar 高性能算子的编码原则。每条原则的格式统一为：

- **原理**：为什么这样写会让 Scalar 流水跑得更快
- **识别要点**：在已有代码中识别该反模式的关键信号
- **反例 / 正例**：可直接对照的代码模板（非真实算子代码，用于模式识别）

> 所有原则的共同目标：**帮助编译器更好地进行寄存器分配、别名分析和常量传播，从而减少 Load/Store 指令**。

---

## P1 结构体内慎用数组

### 原理

结构体内的数组成员，如果通过**动态下标**（运行时变量）访问，会让编译器的**别名分析（Alias Analysis）**失败：编译器无法确定下标的具体值，必须考虑最坏情况——下标可能经过计算后恰好指向结构体中的其他成员。为了保证正确性，编译器会**放弃对整个结构体所有成员的常量传播**，改为从内存重新读取。该污染会沿继承/包含关系向上传播。

### 识别要点

- 类/结构体中同时存在数组成员和需要常量优化的其他成员
- 通过运行时变量（如 ping-pong 索引、循环变量）下标访问数组成员
- profiling 中观察到「常量参数本应折叠却仍走 Load」的现象

### 反例

```cpp
class Block {
    event_t eventIds_[2] = {EVENT_ID0, EVENT_ID1};   // 数组成员
    event_t eventIdsM_[2] = {EVENT_ID2, EVENT_ID3};
    uint16_t pingPongId_ = 0;                         // 运行时下标
    uint32_t tileM_;                                  // 同结构体内其他成员被一起污染
};

void Block::Step() {
    SetFlag<HardEvent::V_MTE2>(eventIds_[pingPongId_]);  // 动态下标
    pingPongId_ ^= 1;
    // 即使 tileM_ 在编译期可推断为常量，也会被保守地走 Load
}
```

### 正例

```cpp
class Block {
    event_t eventId_ = EVENT_ID0;                     // 拆为独立标量
    event_t eventIdM_ = EVENT_ID2;
    uint16_t pingPongId_ = 0;
    uint32_t tileM_;
};

void Block::Step() {
    SetFlag<HardEvent::V_MTE2>(eventId_);
    eventId_   = (eventId_   == EVENT_ID0) ? EVENT_ID1 : EVENT_ID0;
    eventIdM_  = (eventIdM_  == EVENT_ID2) ? EVENT_ID3 : EVENT_ID2;
    pingPongId_ ^= 1;
}
```

---

## P2 循环主尾块分离

### 原理

热路径循环（如 Cube 类算子的 K-Reduce 循环）若在单一 `for` 内通过 `if` 处理首块、周期性事件、尾块，则**每轮迭代都执行 3 次分支判断**。Scalar 的分支预测器较简单，对"大多数迭代为 false 但偶尔为 true"的模式预测效果差；热路径中每次预测失败的代价被放大。

### 识别要点

- 热路径循环内出现 `if (k == 0)`、`if (k == maxK - 1)`、`if (k % N == 0)` 等模式
- 循环次数大（数十~数百），且循环体含 MTE2 周期性 load 或首尾特殊处理

### 反例

```cpp
for (uint64_t k = 0; k < maxK; ++k) {
    if (k == 0)               { SetInitialParams(); }
    if (k % loadStride == 0)  { MTE2_LoadAL1(); }
    if (k == maxK - 1)        { SetFinalParams(); }
    UpdateParams(k);
    Compute();
}
```

### 正例

```cpp
// Prologue：处理首块
SetInitialParams();
UpdateParams(0);
Compute();

// Hot Loop：零分支，仅纯计算与参数更新，周期事件被外层包住
uint64_t k = 1;
uint64_t segmentEnd = loadStride;
while (k < maxK - 1) {
    if unlikely(k == segmentEnd) {
        MTE2_LoadAL1();
        segmentEnd += loadStride;
    }
    uint64_t end = (segmentEnd < maxK - 1) ? segmentEnd : maxK - 1;
    for (; k < end; ++k) {           // 内层 for 内零分支
        UpdateParams(k);
        Compute();
    }
}

// Epilogue：处理尾块
UpdateParams(maxK - 1);
SetFinalParams();
Compute();
```

> 适用边界：循环次数 < 5 次时，三段式带来的代码膨胀可能触发 I-Cache miss，需权衡。

---

## P3 显式编写循环代码

### 原理

`for` 循环的回跳走**专用循环硬件**，不经过分支预测器，不会发生 flush；同时编译器对显式循环更容易做循环展开、循环无关变量外提等优化。
反之，通过 `while(Iterate())` 状态机隐式实现多维嵌套循环时，每次 Tile 切换都会触发大量进位判断（一次切换可能引发 3–5 次连续 mispredict）；状态机函数为处理各维度 Tail 与边界，代码体积通常很大（数百行），容易触发 I-Cache miss。

### 识别要点

- 看到 `while(Iterate(self))`、`while(self->Next())` 等隐式状态机驱动循环
- `Iterate` / `Next` 函数内部级联多个 `if (counter == limit) { counter = 0; ...higher_dim++ }` 进位判断
- 函数体长（>200 行）、内部存在多层嵌套 `if` 处理维度切换

### 反例

```cpp
while (IterateMFirstMMode(self)) {     // 隐式状态机
    LoadAL1(self);
    IterateK(self);
    FreeTensor();
}

// IterateMFirstMMode 内部：
template <class Intf>
inline bool IterateMFirstMMode(Intf* self) {
    if (IterateL0MFirstMMode(self)) return true;
    self->ctx.mAL1Iter++;
    if (self->ctx.mAL1Iter != self->ctx.loopM) { CalcMVar(self); return true; }
    self->ctx.mAL1Iter = 0;
    self->ctx.batchIter++;
    if (self->ctx.batchIter != self->ctx.loopBatch) return true;
    self->ctx.batchIter = 0;
    // ... 多层级联 ...
    return false;
}
```

### 正例

```cpp
for (uint64_t nBL1 = 0; nBL1 < self->ctx.loopN; ++nBL1) {
    for (uint64_t batch = 0; batch < self->ctx.loopBatch; ++batch) {
        for (uint64_t mAL1 = 0; mAL1 < self->ctx.loopM; ++mAL1) {
            self->ctx.nBL1Iter   = nBL1;
            self->ctx.batchIter  = batch;
            self->ctx.mAL1Iter   = mAL1;
            CalcMVar(self);

            for (uint64_t nL0 = 0; nL0 < self->ctx.l12l0LoopN; ++nL0) {
                for (uint64_t mL0 = 0; mL0 < self->ctx.l12l0LoopM; ++mL0) {
                    // L0 层计算
                }
            }
        }
    }
}
```

---

## P4 尽量使用局部变量

### 原理

局部变量与成员变量在编译器优化层面有三方面本质差异：

1. **寄存器分配**：未取地址的局部变量可能从不进入内存；成员变量必须通过 `this`+offset 访问，每次读写至少一次内存访问。
2. **别名分析**：未取地址的局部变量，编译器知道无其它指针能指向它，可大胆做寄存器复用；成员变量则因可能的指针别名，每次读取都可能需要重新 Load。
3. **跨函数调用**：未取地址的局部变量可跨函数调用保持在寄存器中；成员变量被外部函数调用后须重新从内存加载。

### 识别要点

- 类中存在仅在单个函数（或少数内联函数）中频繁访问的成员变量
- 该成员变量在 kernel 生命周期内的值实际上不需要跨函数共享
- 含 `this->xxx` 的热路径表达式特别多

### 反例

```cpp
template <typename T>
class FlashAttentionKernel {
protected:
    ConstParam constParam_;                          // 类成员
public:
    inline void Init() {
        for (int i = 0; i < constParam_.iters; ++i) {       // 每轮 Load constParam_
            DoStep(constParam_.scale);                       // this->constParam_.scale
        }
    }
};
```

### 正例

```cpp
template <typename T>
class FlashAttentionKernel {
public:
    inline void Init() {
        ConstParam constParam;                       // 局部变量，可全程驻留寄存器
        // ... 初始化 constParam ...
        for (int i = 0; i < constParam.iters; ++i) {
            DoStep(constParam.scale);                 // 无 this 间接、无别名顾虑
        }
    }
};
```

---

## P5 变量定义贴近使用位置

### 原理

编译器把变量映射到寄存器。变量的**活跃范围（Live Range）**——从定义到最后一次使用的代码范围——越长，该变量占用寄存器的时间越长。长活跃范围会挤压其他变量的寄存器空间，迫使编译器进行 Spill。贴近使用位置定义则可能让变量直接使用临时寄存器、甚至被优化掉。

### 识别要点

- 函数顶部集中定义大量局部变量
- 某个变量被定义后，跨越了大段（数十行以上）无关代码后才被使用
- 函数中段有大量函数调用或重计算（寄存器压力大）

### 反例

```cpp
void Foo() {
    int a = 1;              // 活跃范围横跨整个函数
    int b = 2;
    int c = 3;

    DoHeavyWork();          // a/b/c 与重计算的临时值争寄存器
    CallSubKernel();
    DoMoreWork();

    Print(a, b, c);         // 真正使用在末尾
}
```

### 正例

```cpp
void Foo() {
    DoHeavyWork();
    CallSubKernel();
    DoMoreWork();

    int a = 1;              // 紧邻使用，活跃范围极短
    int b = 2;
    int c = 3;
    Print(a, b, c);
}
```

---

## P6 避免多级指针解引用

### 原理

多级指针带来三方面问题：

1. **访存延迟串行累加**：每一级解引用本质上是一条 Load 指令，且这些 Load 之间存在数据依赖（前一条 Load 的结果是下一条 Load 的地址）。
2. **破坏数据局部性**：多级指针的各级地址可能散落在堆内存不同位置，每级解引用都可能触发 D-Cache miss。
3. **编译器优化困难**：指针别名问题让编译器不确定指针的某一级内容是否会被其他写操作修改，每次解引用都必须从内存重新加载。

### 识别要点

- 类成员是指针类型，且通过该指针访问被指对象的成员
- 形如 `obj_->member->field` 或 `tiling_->params->Kb` 的访问链
- tiling/config 类参数通过指针长期持有

### 反例

```cpp
class Block {
    const TCubeTiling* tiling_;     // 指针类型成员
public:
    bool IsTail(int kInner) {
        return kInner + stepK_ >= tiling_->Kb;   // 2 次 Load：先 Load tiling_，再 Load tiling_->Kb
    }
};
```

### 正例

```cpp
class Block {
    AscendC::Shape<int64_t, int64_t, int64_t> problemShape_;   // 值类型聚合体 (M, N, K)
public:
    bool IsTail(int kInner) {
        return kInner + stepK_ >= Get<MNK_K>(problemShape_);    // 1 次 Load
    }
};
```

---

## P7 避免使用超大结构体

### 原理

1. **缓存局部性差**：D-Cache cacheline 大小为 64 字节。结构体过大时，访问不同成员可能落在不同 cacheline 上，导致 D-Cache miss 增多。建议结构体 ≤ 64B（1 个 cacheline），最大不超过 128B。
2. **寄存器溢出**：编译器无法将超大结构体完全放入寄存器，必须分配到栈内存上，每次访问成员都生成 Load/Store 指令。
3. **指针传递引入别名**：为避免拷贝开销，大结构体常通过指针传递，但这引入别名问题——编译器无法确定两个指针是否指向同一内存，无法激进优化。

### 识别要点

- 单个 struct/class 的成员超过 10 个、估算尺寸 > 64B
- "上帝结构体"：把不同计算阶段（L1/L0/事件/调试）的所有字段堆在同一个对象里
- 接口签名出现 `void f(BigParam* a, BigParam* b, ...)`（多个同类型指针，别名风险）

### 反例

```cpp
struct GodParams {                  // ~200B，跨多个 cacheline
    // L1 阶段
    uint64_t l1A, l1B, l1ScaleA, l1ScaleB;
    // L0 阶段
    uint64_t l0A, l0B, l0C;
    // 事件
    event_t evV2MTE2, evV2MTE3, evMTE1_M, evM_FIX;
    // tiling、调试标志
    TilingParams tiling;
    uint32_t debugFlags;
    // ...
};

void Compute(GodParams* a, GodParams* b) {
    a->l1A = ...;
    b->l1A = ...;                   // 若 a == b，前一条赋值被覆盖
    int x = a->l1A;                 // 编译器必须重新 Load
    int y = a->l0C;                 // 与 l1A 在不同 cacheline
}
```

### 正例

```cpp
struct L1Stage { uint64_t a, b, scaleA, scaleB; };           // 32B
struct L0Stage { uint64_t a, b, c; };                         // 24B
struct EventIds { event_t v2mte2, v2mte3, mte1_m, m_fix; };   // ≤ 32B

void Compute(L1Stage l1, L0Stage l0) {                        // 按值传递，无别名
    l1.a = ...;
    l0.c = ...;
}
```

---

## P8 用 constexpr / 模板参数承载编译期常量

### 原理

`const` 修饰的成员变量、或在构造函数中赋值的运行时常量，对编译器来说**仍是 runtime 值**——必须通过 Load 读取，且无法参与常量折叠（Constant Folding）和循环展开（Loop Unrolling）。
改为 `constexpr` 或模板非类型参数后，该值在编译期就嵌入到生成代码中，进而能链式触发 P1 类常量传播优化（编译器一旦确认某个值为常量，就更愿意把相关变量也保持在寄存器中）。

### 识别要点

- 类中含 `const` 成员变量、值在构造时确定但程序生命周期内不变
- 在 host 侧本就以模板/编译期方式生成的 tiling 参数，却被以 runtime 入参方式传入 kernel
- 形如 `for (int i = 0; i < kernel.tileM_; ++i)` 的循环，循环上限本应在编译期可知

### 反例

```cpp
class Kernel {
    const uint32_t tileM_;               // const 成员，对编译器仍是 runtime Load
    const uint32_t tileK_;
public:
    Kernel(uint32_t m, uint32_t k) : tileM_(m), tileK_(k) {}
    void Run() {
        for (uint32_t i = 0; i < tileM_; ++i) {        // 无法展开
            for (uint32_t j = 0; j < tileK_; ++j) {
                Compute(i, j);
            }
        }
    }
};
```

### 正例

```cpp
// 方案 A：模板非类型参数——值在类型中固化
template <uint32_t TILE_M, uint32_t TILE_K>
class Kernel {
public:
    void Run() {
        for (uint32_t i = 0; i < TILE_M; ++i) {        // 可被完全展开
            for (uint32_t j = 0; j < TILE_K; ++j) {
                Compute(i, j);
            }
        }
    }
};

// 方案 B：constexpr static——全局编译期常量
class Kernel {
    static constexpr uint32_t TILE_M = 128;
    static constexpr uint32_t TILE_K = 64;
public:
    void Run() {
        for (uint32_t i = 0; i < TILE_M; ++i) {
            for (uint32_t j = 0; j < TILE_K; ++j) {
                Compute(i, j);
            }
        }
    }
};
```

---

## P9 Hot Loop 内不构造对象、不取地址

### 原理

1. **构造/析构展开后是大量 Store**：在热循环内构造对象，每轮迭代都会展开为对成员的逐项初始化 Store；带析构语义的对象还会在迭代末尾插入清理代码。
2. **取地址迫使编译器 Spill**：对局部变量取地址（`&x`）意味着该地址可能传递到外部代码，编译器必须保守地假设变量在调用之后可能被修改，因此必须把变量放到栈上（Spill），不能保持在寄存器中。
3. **取地址也阻碍别名分析**：被取地址的变量与其他指针之间无法证明非别名，污染周围代码的优化。

### 识别要点

- 热循环内出现非平凡类型（如 `LocalTensor`、临时聚合体）的就地构造
- 循环内出现 `Api(&x)` 形式调用，且 `x` 本可按值传递
- 循环内对中间变量频繁取地址传给辅助函数

### 反例

```cpp
// 反例 #1：循环内构造 LocalTensor，每轮触发多个成员的初始化 Store
for (int i = 0; i < N; ++i) {
    LocalTensor<half> tmp = buf.Get<half>();      // 每轮重新构造
    Compute(tmp, i);
}

// 反例 #2：循环内对局部变量取地址，强制 Spill 到栈
for (int i = 0; i < N; ++i) {
    uint32_t idx = i * 2;
    LegacyApi(&idx);                              // &idx 触发 Spill，idx 无法驻留寄存器
}
```

### 正例

```cpp
// 正例 #1：构造外提，循环内仅复用
LocalTensor<half> tmp = buf.Get<half>();
for (int i = 0; i < N; ++i) {
    Compute(tmp, i);
}

// 正例 #2：能按值传递就按值传递；必须传地址时，把变量提升为循环不变量
for (int i = 0; i < N; ++i) {
    uint32_t idx = i * 2;
    NewApi(idx);                                  // 按值传，idx 全程在寄存器
}
```
