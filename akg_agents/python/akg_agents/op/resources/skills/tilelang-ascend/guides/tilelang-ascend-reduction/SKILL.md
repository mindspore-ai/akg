---
name: tilelang-ascend-reduction
description: "TileLang Ascend 归约类算子编码指南，当生成等含归约维度的算子时参考此指南。"
category: guide
version: "1.0.0"
metadata:
  backend: ascend
  dsl: tilelang_ascend
  hardware: "Atlas A2, Atlas A3"
  operator_type: "reduction"
---

# TileLang Ascend 归约类算子编码指南

---

## 决策树：归约算子路径

**重要**：`T.reduce_sum/max/min` 和 `T.tile.*` 在 Developer 和 Expert 模式下**都可使用**。模式选择取决于是否需要手动控制内存层级和同步，而非使用了哪个 API。

```
含归约（reduce_sum / reduce_max / reduce_min）
    必须正确判断归约维度：归约第一根轴（列归约）；归约最后一根轴（行归约）；归约所有轴；跳轴归约
    选择合适的 API: T.reduce_sum / T.reduce_max / T.reduce_min / T.atomic_add
    内存: T.alloc_shared → UB
```

