# Scalar 性能优化策略索引

Scalar 单元负责标量计算、地址计算、指令参数构造与指令分发。当它本身成为瓶颈、无法及时把指令发到 Cube / Vector / MTE 的 Issue 队列时，其他流水线会出现 bubble，整个算子被拖慢。这种现象称为 **ScalarBound**。

Scalar 优化的核心结论：**主因不是 Scalar 计算指令多，而是 Load/Store 指令多**（典型占比超过 30%），其根源是编译器寄存器 Spill。因此本类目下所有原则的共同目标是：**帮助编译器更好地完成寄存器分配、别名分析和常量传播，从而减少 Load/Store**。

## 9 条编码原则速查

| 编号 | 原则 | 一句话核心 |
|------|------|-----------|
| [P1](coding_principles.md#p1-结构体内慎用数组) | 结构体内慎用数组 | 动态下标的结构体数组会让编译器放弃整个结构体的常量传播 |
| [P2](coding_principles.md#p2-循环主尾块分离) | 循环主尾块分离 | Prologue / Hot Loop（零分支）/ Epilogue 三段式，避免热路径每轮 3 次分支判断 |
| [P3](coding_principles.md#p3-显式编写循环代码) | 显式编写循环代码 | 用嵌套 `for` 替代隐式状态机；`for` 的回跳走专用硬件，不进分支预测 |
| [P4](coding_principles.md#p4-尽量使用局部变量) | 尽量使用局部变量 | 成员变量受别名分析压制；跨函数调用后必须从内存重新 Load |
| [P5](coding_principles.md#p5-变量定义贴近使用位置) | 变量定义贴近使用位置 | 缩短活跃范围 = 缩短寄存器占用 = 降低 Spill 概率 |
| [P6](coding_principles.md#p6-避免多级指针解引用) | 避免多级指针解引用 | 每级解引用都是带依赖的 Load，用值类型聚合体替代 |
| [P7](coding_principles.md#p7-避免使用超大结构体) | 避免使用超大结构体 | > 64B 跨 cacheline；超寄存器容量必 Spill；指针传递引入别名 |
| [P8](coding_principles.md#p8-用-constexpr--模板参数承载编译期常量) | 用 `constexpr` / 模板参数承载编译期常量 | `const` 成员对编译器仍是 runtime Load；进入常量折叠才能链式带动 P1 类优化 |
| [P9](coding_principles.md#p9-hot-loop-内不构造对象不取地址) | Hot Loop 内不构造对象、不取地址 | 构造/析构展开后是大量初始化 Store；取地址会让编译器假设变量可被外部修改、强制 Spill |

## 与其他算子族优化的协同

Scalar 优化与算子族优化**正交**：算子族优化决定"做什么计算和搬运"，Scalar 优化决定"这些指令能不能被及时分发出去"。两者都要做，但顺序上以算子族优化（tiling / MTE 策略）为先。

各算子族在通用 9 条原则之上还有场景特化的 Scalar 优化策略：

| 算子族 | 场景 | 核心思路 | 指南 |
|--------|------|---------|------|
| FA | — | 📋 规划中 | — |
