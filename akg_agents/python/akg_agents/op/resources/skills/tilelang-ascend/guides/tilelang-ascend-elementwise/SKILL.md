---
name: tilelang-ascend-elementwise
description: "TileLang Ascend 逐元素算子编码指南。涵盖 T.Parallel 符号 API 和 T.tile.xxx 扩展原语两种编程范式选择、广播模式、列切分。当生成逐元素类算子时参考此指南。"
category: guide
version: "1.0.0"
metadata:
  backend: ascend
  dsl: tilelang_ascend
  hardware: "Atlas A2, Atlas A3"
  operator_type: "elementwise"
---

# TileLang Ascend 逐元素算子编码指南

---

## 决策树：element-wise 算子路径

**重要**：`T.tile.*` 在 Developer 和 Expert 模式下**都可使用**。模式选择取决于是否需要手动控制内存层级和同步，而非使用了哪个 API。

```
纯 element-wise（逐元素运算）
├─ 单步运算 → Developer 模式
│   API: T.Parallel + 算术符号
│   内存: T.alloc_shared（编译器映射到 UB）
│
└─ 多步运算
    ├─ 需精细 buffer 控制 → Expert 模式
    └─ 无需精细控制 → Developer 模式
```

